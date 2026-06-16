"""
XGBoost — Previsão de Falha a partir de Operação Normal (10 classes).

Usa apenas janelas com window_label=0 (operação normal) do parquet sem filtro,
rotuladas com a fault_class da instância. A tarefa é responder:
"dado que o poço está em operação normal, qual falha ele vai desenvolver?"

  0       → Poço genuinamente normal (nunca terá falha)
  1–9     → Poço que desenvolverá a falha do tipo correspondente

Nota: fault_class é 0-9 (contíguo), portanto NÃO é necessário LabelEncoder.

Execução:
    python scripts/train_xgboost_fault_prediction.py

Saídas:
    results/models/xgboost_fault_prediction.joblib
    results/models/imputer_xgb_fault_prediction.joblib
    results/metrics/xgboost_fault_prediction_metrics.json
    results/metrics/xgb_gain_fault_prediction.json
    results/metrics/shap_xgboost_fault_prediction.json
    results/figures/confusion_matrix_xgboost_fault_prediction.png
    results/figures/xgb_gain_fault_prediction.png
    results/figures/xgb_weight_fault_prediction.png
    results/figures/xgb_cover_fault_prediction.png
    results/figures/shap_bar_xgboost_fault_prediction.png
    results/figures/shap_beeswarm_xgboost_fault_prediction.png
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["PYTHONIOENCODING"] = "utf-8"

from config import (
    FAULT_CLASSES,
    FEATURES_NOFILTER_PATH,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    N_ITER_SEARCH,
    N_JOBS,
    N_SPLITS_CV,
    RANDOM_STATE,
)
from src.evaluation import plot_confusion_matrix, print_classification_report



def main():
    print("=" * 60)
    print("  Treinamento: XGBoost — Previsao de Falha (10 classes)")
    print("  Entrada: apenas janelas de operacao normal (window_label=0)")
    print(f"  Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ── Carregar e filtrar features ───────────────────────────────────────────
    print("\n[1/6] Carregando features (sem filtro de sinal)...")
    if not FEATURES_NOFILTER_PATH.exists():
        print(f"ERRO: {FEATURES_NOFILTER_PATH} nao encontrado.")
        print("Gere com: python scripts/run_pipeline_window_class.py --filter none")
        sys.exit(1)

    df = pd.read_parquet(FEATURES_NOFILTER_PATH)
    print(f"  Total de janelas no parquet: {len(df):,}")

    df = df[df["window_label"] == 0].copy()
    print(f"  Janelas de operacao normal (window_label=0): {len(df):,}")

    META_COLS = ["instance_id", "fault_class", "window_label", "source_type", "window_start"]
    feature_cols = [c for c in df.columns if c not in META_COLS]

    X      = df[feature_cols].values
    y      = df["fault_class"].values
    groups = df["instance_id"].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)

    classes, counts = np.unique(y, return_counts=True)
    print(f"  Features: {len(feature_cols)}")
    print(f"  Classes unicas: {len(classes)} — {sorted(classes.tolist())}")
    print(f"  Distribuicao por classe de falha:")
    for c, n in zip(classes, counts):
        pct = 100 * n / len(y)
        print(f"    {c:2d} ({FAULT_CLASSES.get(c, '?'):<35}): {n:>8,} ({pct:.1f}%)")

    # ── Imputar NaN ───────────────────────────────────────────────────────────
    print("\n[2/6] Imputando NaN com mediana...")
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(imputer, MODELS_DIR / "imputer_xgb_fault_prediction.joblib")
    joblib.dump(le, MODELS_DIR / "label_encoder_xgb_fault_prediction.joblib")

    sample_weight = compute_sample_weight("balanced", y_enc)

    # ── Busca de hiperparâmetros ──────────────────────────────────────────────
    print(f"\n[3/6] Busca de hiperparametros (RandomizedSearchCV)...")
    print(f"  N_ITER={N_ITER_SEARCH} | N_SPLITS={N_SPLITS_CV} | N_JOBS={N_JOBS}")

    param_grid = {
        "n_estimators":     [100, 200, 300, 500],
        "max_depth":        [3, 4, 6, 8],
        "learning_rate":    [0.01, 0.05, 0.1, 0.2],
        "subsample":        [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
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

    model_path = MODELS_DIR / "xgboost_fault_prediction.joblib"
    joblib.dump(best_xgb, model_path)
    print(f"  Modelo salvo: {model_path}")

    # ── Predições out-of-fold ─────────────────────────────────────────────────
    print(f"\n[4/6] Gerando predicoes out-of-fold para metricas...")
    y_true_all = []
    y_pred_all = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
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
        y_pred_fold = le.inverse_transform(xgb_fold.predict(X[test_idx]))
        y_true_all.extend(y[test_idx])
        y_pred_all.extend(y_pred_fold)

        f1_fold = f1_score(y[test_idx], y_pred_fold, average="macro", zero_division=0)
        print(f"  Fold {fold}: F1-macro = {f1_fold:.4f}")

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    f1_macro    = f1_score(y_true_all, y_pred_all, average="macro",    zero_division=0)
    f1_weighted = f1_score(y_true_all, y_pred_all, average="weighted", zero_division=0)
    acc         = accuracy_score(y_true_all, y_pred_all)

    print(f"\n  F1-macro   : {f1_macro:.4f}")
    print(f"  F1-weighted: {f1_weighted:.4f}")
    print(f"  Accuracy   : {acc:.4f}")

    print_classification_report(y_true_all, y_pred_all, model_name="XGBoost Previsao de Falha")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(
        y_true_all, y_pred_all,
        model_name="xgboost_fault_prediction",
        label_map=FAULT_CLASSES,
        save=True,
    )

    report = classification_report(
        y_true_all, y_pred_all,
        labels=sorted(classes.tolist()),
        target_names=[FAULT_CLASSES.get(c, str(c)) for c in sorted(classes.tolist())],
        output_dict=True,
        zero_division=0,
    )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {
        "model": "xgboost_fault_prediction",
        "trained_at": datetime.now().isoformat(),
        "dataset": {
            "n_windows":   int(len(df)),
            "n_features":  int(len(feature_cols)),
            "n_instances": int(len(np.unique(groups))),
            "n_classes":   int(n_classes),
            "filter":      "none",
            "window_filter": "window_label == 0 (operacao normal)",
        },
        "cv": {
            "strategy":      f"GroupKFold(n_splits={N_SPLITS_CV})",
            "scoring":       "f1_macro",
            "best_score_cv": round(float(best_cv_f1), 4),
        },
        "best_params": best_params,
        "metrics_oof": {
            "f1_macro":    round(float(f1_macro), 4),
            "f1_weighted": round(float(f1_weighted), 4),
            "accuracy":    round(float(acc), 4),
        },
        "per_class": {
            str(c): {
                "name":      FAULT_CLASSES.get(c, str(c)),
                "precision": round(report[FAULT_CLASSES.get(c, str(c))]["precision"], 4),
                "recall":    round(report[FAULT_CLASSES.get(c, str(c))]["recall"], 4),
                "f1":        round(report[FAULT_CLASSES.get(c, str(c))]["f1-score"], 4),
                "support":   int(report[FAULT_CLASSES.get(c, str(c))]["support"]),
            }
            for c in sorted(classes.tolist())
        },
    }
    metrics_path = METRICS_DIR / "xgboost_fault_prediction_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"  Metricas salvas: {metrics_path}")

    pd.DataFrame(search.cv_results_).to_csv(
        METRICS_DIR / "xgboost_fault_prediction_cv_results.csv", index=False
    )

    # ── Explicabilidade: Importância Nativa XGBoost ───────────────────────────
    print("\n[5/6] Calculando importancia de features...")
    print("  [5a] Importancias nativas do XGBoost (weight, gain, cover)...")
    booster = best_xgb.get_booster()
    booster.feature_names = feature_cols  # evita nomes genéricos f0/f1/...

    for imp_type in ["weight", "gain", "cover"]:
        scores = booster.get_score(importance_type=imp_type)
        full_scores = {f: scores.get(f, 0.0) for f in feature_cols}
        xgb_imp = pd.Series(full_scores).sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(10, 8))
        xgb_imp.head(20).plot(kind="barh", ax=ax)
        ax.set_title(f"XGBoost Importance — {imp_type} (top 20 features)")
        ax.set_xlabel(f"Score ({imp_type})")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"xgb_{imp_type}_fault_prediction.png", dpi=150, bbox_inches="tight")
        plt.close()

        if imp_type == "gain":
            with open(METRICS_DIR / "xgb_gain_fault_prediction.json", "w") as f:
                json.dump(xgb_imp.to_dict(), f, indent=2)
    print(f"    Importancias nativas salvas")

    # ── Explicabilidade: SHAP ─────────────────────────────────────────────────
    print("  [5b] SHAP TreeExplainer (amostra de ate 5000 janelas)...")
    X_sample = X[:5000] if len(X) > 5000 else X
    explainer = shap.TreeExplainer(best_xgb)
    shap_expl = explainer(X_sample)
    # shap_expl.values: [n_samples, n_features, n_classes]
    shap_arr = np.abs(shap_expl.values)  # [n_samples, n_features, n_classes]
    feat_scores = shap_arr.mean(axis=(0, 2))  # [n_features] — média por feature

    # Plot de barras: importância global via matplotlib
    shap_series = pd.Series(feat_scores, index=feature_cols).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 8))
    shap_series.head(20).plot(kind="barh", ax=ax)
    ax.set_title("XGBoost — SHAP Global Importance (top 20 features)")
    ax.set_xlabel("Media de |SHAP value|")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_bar_xgboost_fault_prediction.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot beeswarm: distribuição de impacto usando média entre classes
    shap_mean_2d = shap_arr.mean(axis=2)  # [n_samples, n_features]
    shap.summary_plot(
        shap_mean_2d, X_sample,
        feature_names=feature_cols,
        max_display=20,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_beeswarm_xgboost_fault_prediction.png", dpi=150, bbox_inches="tight")
    plt.close()

    shap_ranking = dict(zip(feature_cols, feat_scores.tolist()))
    shap_ranking = dict(sorted(shap_ranking.items(), key=lambda x: x[1], reverse=True))
    with open(METRICS_DIR / "shap_xgboost_fault_prediction.json", "w") as f:
        json.dump(shap_ranking, f, indent=2)
    print(f"    SHAP salvo")

    print("\n[6/6] Concluido!")
    print("=" * 60)
    print("  Treinamento concluido!")
    print(f"  Fim: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  F1-macro OOF: {f1_macro:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
