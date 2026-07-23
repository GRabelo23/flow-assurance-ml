"""
Treinamento do XGBoost — Estado Operacional (17 classes).

Usa features_window_class.parquet onde cada janela e rotulada com a moda
da coluna 'class' dentro da janela:
  0        -> Normal
  1-9      -> Evento ativo do tipo correspondente
  101-109  -> Transiente do tipo correspondente

NOTA: XGBoost exige labels contiguos comecando em 0. Os labels 101-109
nao sao contiguos, entao e aplicado um LabelEncoder antes do treino.
O encoder e salvo em imputer_xgb_window_class.joblib para uso futuro.

Execução:
    python scripts/train_xgboost_window_class.py [--filter {gaussian,statistical,none}]

Saídas (sem --filter ou --filter gaussian):
    results/models/xgboost_window_class.joblib
    results/models/imputer_xgb_window_class.joblib
    results/models/label_encoder_window_class.joblib
    results/metrics/xgboost_window_class_metrics.json
    results/figures/confusion_matrix_xgboost_estado_operacional.png
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import joblib
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
    MODELS_DIR,
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

    print("=" * 60)
    print(f"  Treinamento: XGBoost — Estado Operacional (17 classes)")
    print(f"  Filtro: {filter_type}")
    print(f"  Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ── Carregar features ─────────────────────────────────────────────────────
    print("\n[1/5] Carregando features...")
    if not data_path.exists():
        print(f"ERRO: {data_path} nao encontrado.")
        print("Execute primeiro: python scripts/run_pipeline_window_class.py")
        sys.exit(1)

    df = pd.read_parquet(data_path)
    META_COLS = ["instance_id", "fault_class", "window_label", "source_type", "window_start"]
    feature_cols = [c for c in df.columns if c not in META_COLS]

    X      = df[feature_cols].values
    y_orig = df["window_label"].values   # labels originais: 0,1-9,101-109
    groups = df["instance_id"].values

    classes_orig, counts = np.unique(y_orig, return_counts=True)
    print(f"  Janelas: {len(df):,} | Features: {len(feature_cols)}")
    print(f"  Classes unicas: {len(classes_orig)} — {sorted(classes_orig.tolist())}")
    print(f"  Distribuicao:")
    for c, n in zip(classes_orig, counts):
        pct = 100 * n / len(y_orig)
        print(f"    {c:4d} ({WINDOW_CLASSES.get(c,'?'):<22}): {n:>8,} ({pct:.1f}%)")

    # ── Imputar NaN ───────────────────────────────────────────────────────────
    print("\n[2/5] Imputando NaN com mediana por instancia...")
    X = _impute_per_instance(X, groups, feature_cols)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Codificar labels para intervalo [0, n_classes-1] ─────────────────────
    # XGBoost exige labels contiguos. Os originais 101-109 causariam erro.
    le = LabelEncoder()
    y_enc = le.fit_transform(y_orig)   # ex: 101 -> 10, 102 -> 11, ...
    joblib.dump(le, MODELS_DIR / f"label_encoder_window_class{suffix}.joblib")
    n_classes = len(le.classes_)
    print(f"  LabelEncoder: {len(classes_orig)} classes mapeadas para [0, {n_classes-1}]")
    print(f"  Mapeamento: {dict(zip(le.classes_.tolist(), range(n_classes)))}")

    # ── Pesos de amostra ──────────────────────────────────────────────────────
    # Calculados sobre y_orig para refletir desbalanceamento real das classes.
    sample_weight = compute_sample_weight("balanced", y_orig)

    # ── Busca de hiperparâmetros ──────────────────────────────────────────────
    print(f"\n[3/5] Busca de hiperparametros (RandomizedSearchCV)...")
    print(f"  N_ITER={N_ITER_SEARCH} | N_SPLITS={N_SPLITS_CV} | N_JOBS={N_JOBS}")

    param_grid = {
        "n_estimators":    [100, 200, 300, 500],
        "max_depth":       [3, 4, 6, 8],
        "learning_rate":   [0.01, 0.05, 0.1, 0.2],
        "subsample":       [0.7, 0.8, 1.0],
        "colsample_bytree":[0.7, 0.8, 1.0],
        "min_child_weight":[1, 3, 5],
    }

    xgb = XGBClassifier(
        objective="multi:softmax",
        num_class=n_classes,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        verbosity=0,
    )

    gkf = GroupKFold(n_splits=N_SPLITS_CV)
    search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_grid,
        n_iter=N_ITER_SEARCH,
        cv=gkf,
        scoring="f1_macro",
        n_jobs=1,
        random_state=RANDOM_STATE,
        verbose=2,
        refit=True,
        return_train_score=True,
    )

    search.fit(X, y_enc, groups=groups, sample_weight=sample_weight)
    best_xgb    = search.best_estimator_
    best_params = search.best_params_
    best_cv_f1  = search.best_score_

    print(f"\n  Melhor F1-macro (CV): {best_cv_f1:.4f}")
    print(f"  Melhores parametros: {best_params}")

    # ── Salvar modelo ─────────────────────────────────────────────────────────
    model_path = MODELS_DIR / f"xgboost_window_class{suffix}.joblib"
    joblib.dump(best_xgb, model_path)
    print(f"  Modelo salvo: {model_path}")

    # ── Predições out-of-fold ─────────────────────────────────────────────────
    print(f"\n[4/5] Gerando predicoes out-of-fold para metricas...")
    y_true_enc_all = []
    y_pred_enc_all = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y_enc, groups), 1):
        xgb_fold = XGBClassifier(
            **best_params,
            objective="multi:softmax",
            num_class=n_classes,
            tree_method="hist",
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            verbosity=0,
        )
        xgb_fold.fit(
            X[train_idx], y_enc[train_idx],
            sample_weight=sample_weight[train_idx],
        )
        y_pred_enc = xgb_fold.predict(X[test_idx])
        y_true_enc_all.extend(y_enc[test_idx])
        y_pred_enc_all.extend(y_pred_enc)

        # F1 do fold nos labels originais (mais legivel)
        y_true_fold = le.inverse_transform(y_enc[test_idx])
        y_pred_fold = le.inverse_transform(y_pred_enc)
        f1_fold = f1_score(y_true_fold, y_pred_fold, average="macro", zero_division=0)
        print(f"  Fold {fold}: F1-macro = {f1_fold:.4f}")

    # Converter de volta para labels originais para todas as metricas
    y_true_all = le.inverse_transform(np.array(y_true_enc_all))
    y_pred_all = le.inverse_transform(np.array(y_pred_enc_all))

    f1_macro    = f1_score(y_true_all, y_pred_all, average="macro",    zero_division=0)
    f1_weighted = f1_score(y_true_all, y_pred_all, average="weighted", zero_division=0)
    acc         = accuracy_score(y_true_all, y_pred_all)

    print(f"\n  F1-macro   : {f1_macro:.4f}")
    print(f"  F1-weighted: {f1_weighted:.4f}")
    print(f"  Accuracy   : {acc:.4f}")

    model_label = f"XGBoost Estado Operacional ({filter_type})" if filter_type != "gaussian" else "XGBoost Estado Operacional"
    print_classification_report(y_true_all, y_pred_all, model_name=model_label)

    # ── Matriz de confusão ────────────────────────────────────────────────────
    print("\n[5/5] Plotando matriz de confusao...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(
        y_true_all, y_pred_all,
        model_name=f"xgboost_window_class{suffix}_estado_operacional",
        label_map=WINDOW_CLASSES,
        save=True,
    )

    # ── Salvar métricas em JSON ───────────────────────────────────────────────
    report = classification_report(
        y_true_all, y_pred_all,
        labels=sorted(classes_orig.tolist()),
        target_names=[WINDOW_CLASSES.get(c, str(c)) for c in sorted(classes_orig.tolist())],
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "model": f"xgboost_window_class{suffix}",
        "filter": filter_type,
        "trained_at": datetime.now().isoformat(),
        "dataset": {
            "n_windows":   int(len(df)),
            "n_features":  int(len(feature_cols)),
            "n_instances": int(len(np.unique(groups))),
            "n_classes":   int(len(classes_orig)),
        },
        "cv": {
            "strategy":      f"GroupKFold(n_splits={N_SPLITS_CV})",
            "scoring":       "f1_macro",
            "best_score_cv": round(float(best_cv_f1), 4),
        },
        "best_params": best_params,
        "metrics_concat_folds": {
            "f1_macro":    round(float(f1_macro), 4),
            "f1_weighted": round(float(f1_weighted), 4),
            "accuracy":    round(float(acc), 4),
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
    metrics_path = METRICS_DIR / f"xgboost_window_class{suffix}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"  Metricas salvas: {metrics_path}")

    pd.DataFrame(search.cv_results_).to_csv(
        METRICS_DIR / f"xgboost_window_class{suffix}_cv_results.csv", index=False
    )

    print("\n" + "=" * 60)
    print("  Treinamento concluido!")
    print(f"  Fim: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
