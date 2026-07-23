"""
XGBoost — Nested Cross-Validation (estudo de viés).

Objetivo: estimar o F1-macro sem viés de seleção de hiperparâmetros,
comparando com o F1-macro do CV plano (0.9082) obtido em
train_xgboost_window_class.py.

Estrutura do Nested CV:
  Loop externo  — GroupKFold(K_out=5): mede desempenho de forma imparcial.
  Loop interno  — RandomizedSearchCV(GroupKFold(K_in=3), 20 configs):
                  seleciona hiperparâmetros dentro de cada fold externo.
  Total de treinos: K_out × K_in × N_ITER = 5 × 3 × 20 = 300.

Viés estimado = F1_flat (0.9082) − F1_nested_mean.
  Positivo → CV plano era otimista.
  Negativo → CV plano era pessimista (improvável).

Nenhum modelo final é salvo — este script é exclusivamente para avaliação.

Execução:
    python scripts/train_xgboost_nested_cv.py [--filter {gaussian,statistical,none}]

Saídas (filtro gaussiano):
    results/metrics/xgboost_nested_cv_metrics.json
    results/figures/confusion_matrix_xgboost_nested_cv_estado_operacional.png
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["PYTHONIOENCODING"] = "utf-8"

from config import (
    FEATURES_BY_FILTER,
    FIGURES_DIR,
    METRICS_DIR,
    N_ITER_SEARCH,
    N_JOBS,
    N_SPLITS_CV,
    RANDOM_STATE,
    WINDOW_CLASSES,
)
from src.evaluation import plot_confusion_matrix, print_classification_report


def _impute_per_instance(X_raw: np.ndarray, groups: np.ndarray, feature_cols: list) -> np.ndarray:
    """Preenche NaN com a mediana da própria instância para cada feature.

    Como GroupKFold nunca divide janelas de uma mesma instância entre folds,
    isso é equivalente à imputação por fold — sem vazamento de informação.
    Fallback para 0.0 quando a feature é inteiramente NaN na instância
    (valor neutro no espaço z-score normalizado).
    """
    df = pd.DataFrame(X_raw, columns=feature_cols)
    df["_gid"] = groups
    inst_med = df.groupby("_gid")[feature_cols].transform("median")
    df[feature_cols] = df[feature_cols].fillna(inst_med).fillna(0.0)
    return df[feature_cols].values


# F1-macro do CV plano para calcular o viés
F1_FLAT_OOF = 0.9082

# Folds internos (menos que o externo para reduzir custo computacional)
K_IN = 3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filter", choices=["gaussian", "statistical", "none"],
        default="gaussian", dest="filter_type",
        help="Tipo de filtro aplicado ao sinal (padrão: gaussian)",
    )
    args = parser.parse_args()
    filter_type = args.filter_type
    suffix = f"_{filter_type}" if filter_type != "gaussian" else ""

    data_path = FEATURES_BY_FILTER[filter_type]

    print("=" * 65)
    print(f"  XGBoost — Nested CV (estudo de viés)")
    print(f"  Filtro: {filter_type}")
    print(f"  Estrutura: K_out={N_SPLITS_CV}, K_in={K_IN}, N_iter={N_ITER_SEARCH}")
    print(f"  Total de treinos: {N_SPLITS_CV * K_IN * N_ITER_SEARCH}")
    print(f"  Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # ── Carregar features ─────────────────────────────────────────────────────
    print("\n[1/4] Carregando features...")
    if not data_path.exists():
        print(f"ERRO: {data_path} nao encontrado.")
        print("Execute primeiro: python scripts/run_pipeline_window_class.py")
        sys.exit(1)

    df = pd.read_parquet(data_path)
    META_COLS = ["instance_id", "fault_class", "window_label", "source_type", "window_start"]
    feature_cols = [c for c in df.columns if c not in META_COLS]

    X      = df[feature_cols].values
    y_orig = df["window_label"].values
    groups = df["instance_id"].values

    classes_orig, counts = np.unique(y_orig, return_counts=True)
    print(f"  Janelas: {len(df):,} | Features: {len(feature_cols)}")
    print(f"  Classes unicas: {len(classes_orig)} — {sorted(classes_orig.tolist())}")
    print(f"  Distribuicao:")
    for c, n in zip(classes_orig, counts):
        pct = 100 * n / len(y_orig)
        print(f"    {c:4d} ({WINDOW_CLASSES.get(c,'?'):<22}): {n:>8,} ({pct:.1f}%)")

    # ── Imputar NaN e codificar labels ───────────────────────────────────────
    print("\n[2/4] Imputando NaN e codificando labels...")
    X = _impute_per_instance(X, groups, feature_cols)

    le = LabelEncoder()
    y_enc = le.fit_transform(y_orig)
    n_classes = len(le.classes_)
    print(f"  LabelEncoder: {n_classes} classes em [0, {n_classes-1}]")

    sample_weight = compute_sample_weight("balanced", y_orig)

    param_grid = {
        "n_estimators":     [100, 200, 300, 500],
        "max_depth":        [3, 4, 6, 8],
        "learning_rate":    [0.01, 0.05, 0.1, 0.2],
        "subsample":        [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
    }

    # ── Nested CV ─────────────────────────────────────────────────────────────
    print(f"\n[3/4] Nested CV...")
    outer_cv = GroupKFold(n_splits=N_SPLITS_CV)
    inner_cv = GroupKFold(n_splits=K_IN)

    fold_scores   = []
    best_params_per_fold = []
    y_true_all    = []
    y_pred_all    = []

    for fold, (train_idx, test_idx) in enumerate(
        outer_cv.split(X, y_enc, groups), 1
    ):
        print(f"\n  ── Fold externo {fold}/{N_SPLITS_CV} ──────────────────────────────")
        t0 = datetime.now()

        X_train, X_test       = X[train_idx], X[test_idx]
        y_train, y_test       = y_enc[train_idx], y_enc[test_idx]
        groups_train          = groups[train_idx]
        sw_train              = sample_weight[train_idx]

        # Busca de hiperparâmetros apenas nos dados de treino do fold externo
        xgb = XGBClassifier(
            objective="multi:softmax",
            num_class=n_classes,
            tree_method="hist",
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            verbosity=0,
        )
        search = RandomizedSearchCV(
            estimator=xgb,
            param_distributions=param_grid,
            n_iter=N_ITER_SEARCH,
            cv=inner_cv,
            scoring="f1_macro",
            n_jobs=1,
            random_state=RANDOM_STATE,
            verbose=0,
            refit=True,
        )
        search.fit(X_train, y_train,
                   groups=groups_train,
                   sample_weight=sw_train)

        best_params_fold = search.best_params_
        best_inner_f1    = search.best_score_
        best_params_per_fold.append(best_params_fold)

        # Avaliação no fold externo — dados nunca vistos durante busca interna
        y_pred_enc = search.best_estimator_.predict(X_test)

        y_true_fold_orig = le.inverse_transform(y_test)
        y_pred_fold_orig = le.inverse_transform(y_pred_enc)

        f1_fold = f1_score(y_true_fold_orig, y_pred_fold_orig,
                           average="macro", zero_division=0)
        fold_scores.append(f1_fold)

        y_true_all.extend(y_true_fold_orig)
        y_pred_all.extend(y_pred_fold_orig)

        elapsed = int((datetime.now() - t0).total_seconds())
        print(f"    Melhores params internos: {best_params_fold}")
        print(f"    F1-macro interno (CV):    {best_inner_f1:.4f}")
        print(f"    F1-macro externo (fold):  {f1_fold:.4f}  ({elapsed // 60}m {elapsed % 60}s)")

    # ── Métricas finais ───────────────────────────────────────────────────────
    print(f"\n[4/4] Metricas finais...")

    y_true_arr = np.array(y_true_all)
    y_pred_arr = np.array(y_pred_all)

    f1_macro_nested    = float(np.mean(fold_scores))
    f1_macro_nested_std = float(np.std(fold_scores))
    f1_macro_concat    = f1_score(y_true_arr, y_pred_arr, average="macro",    zero_division=0)
    f1_weighted_concat = f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)
    acc_concat         = accuracy_score(y_true_arr, y_pred_arr)

    vies = F1_FLAT_OOF - f1_macro_nested

    print(f"\n  Resultados do Nested CV:")
    print(f"  F1-macro por fold:     {[round(s,4) for s in fold_scores]}")
    print(f"  F1-macro (media):      {f1_macro_nested:.4f} ± {f1_macro_nested_std:.4f}")
    print(f"  F1-macro (concat OOF): {f1_macro_concat:.4f}")
    print(f"  F1-weighted (concat):  {f1_weighted_concat:.4f}")
    print(f"  Accuracy (concat):     {acc_concat:.4f}")
    print(f"\n  ── Estudo de vies ──────────────────────────────────────────")
    print(f"  F1 CV plano (OOF): {F1_FLAT_OOF:.4f}")
    print(f"  F1 Nested CV:      {f1_macro_nested:.4f} ± {f1_macro_nested_std:.4f}")
    print(f"  Vies estimado:     {vies:+.4f}")
    if abs(vies) < 0.005:
        print(f"  Interpretacao: vies negligenciavel (< 0.5 p.p.)")
    elif vies > 0:
        print(f"  Interpretacao: CV plano era otimista em {vies*100:.1f} p.p.")
    else:
        print(f"  Interpretacao: CV plano era pessimista em {abs(vies)*100:.1f} p.p.")

    model_label = f"XGBoost Nested CV Estado Operacional ({filter_type})"
    print_classification_report(y_true_arr, y_pred_arr, model_name=model_label)

    # ── Matriz de confusão ────────────────────────────────────────────────────
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(
        y_true_arr, y_pred_arr,
        model_name=f"xgboost_nested_cv{suffix}_estado_operacional",
        label_map=WINDOW_CLASSES,
        save=True,
    )

    # ── Salvar métricas ───────────────────────────────────────────────────────
    report = classification_report(
        y_true_arr, y_pred_arr,
        labels=sorted(classes_orig.tolist()),
        target_names=[WINDOW_CLASSES.get(c, str(c)) for c in sorted(classes_orig.tolist())],
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "model": f"xgboost_nested_cv{suffix}",
        "filter": filter_type,
        "trained_at": datetime.now().isoformat(),
        "dataset": {
            "n_windows":   int(len(df)),
            "n_features":  int(len(feature_cols)),
            "n_instances": int(len(np.unique(groups))),
            "n_classes":   int(len(classes_orig)),
        },
        "nested_cv": {
            "k_out":           N_SPLITS_CV,
            "k_in":            K_IN,
            "n_iter":          N_ITER_SEARCH,
            "total_trainings": N_SPLITS_CV * K_IN * N_ITER_SEARCH,
            "scoring":         "f1_macro",
        },
        "per_fold_f1": [round(s, 4) for s in fold_scores],
        "best_params_per_fold": best_params_per_fold,
        "metrics_mean_folds": {
            "f1_macro_mean": round(f1_macro_nested, 4),
            "f1_macro_std":  round(f1_macro_nested_std, 4),
        },
        "metrics_concat_folds": {
            "f1_macro":    round(float(f1_macro_concat), 4),
            "f1_weighted": round(float(f1_weighted_concat), 4),
            "accuracy":    round(float(acc_concat), 4),
        },
        "bias_study": {
            "f1_flat_cv":      F1_FLAT_OOF,
            "f1_nested_cv":    round(f1_macro_nested, 4),
            "vies_estimado":   round(vies, 4),
            "interpretacao":   (
                "negligenciavel" if abs(vies) < 0.005
                else ("CV plano otimista" if vies > 0 else "CV plano pessimista")
            ),
        },
        "per_class": {
            str(c): {
                "name":      WINDOW_CLASSES.get(c, str(c)),
                "precision": round(report[WINDOW_CLASSES.get(c, str(c))]["precision"], 4),
                "recall":    round(report[WINDOW_CLASSES.get(c, str(c))]["recall"], 4),
                "f1":        round(report[WINDOW_CLASSES.get(c, str(c))]["f1-score"], 4),
                "support":   int(report[WINDOW_CLASSES.get(c, str(c))]["support"]),
            }
            for c in sorted(classes_orig.tolist())
        },
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = METRICS_DIR / f"xgboost_nested_cv{suffix}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\n  Metricas salvas: {metrics_path}")

    print("\n" + "=" * 65)
    print("  Nested CV concluido!")
    print(f"  Fim: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)


if __name__ == "__main__":
    main()
