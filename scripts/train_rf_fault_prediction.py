"""
Random Forest — Previsão de Falha a partir de Operação Normal (10 classes).

Usa apenas janelas com window_label=0 (operação normal) do parquet sem filtro,
rotuladas com a fault_class da instância. A tarefa é responder:
"dado que o poço está em operação normal, qual falha ele vai desenvolver?"

  0       → Poço genuinamente normal (nunca terá falha)
  1–9     → Poço que desenvolverá a falha do tipo correspondente

Execução:
    python scripts/train_rf_fault_prediction.py

Saídas:
    results/models/rf_fault_prediction.joblib
    results/models/imputer_rf_fault_prediction.joblib
    results/metrics/rf_fault_prediction_metrics.json
    results/metrics/mdi_rf_fault_prediction.json
    results/metrics/perm_rf_fault_prediction.json
    results/metrics/shap_rf_fault_prediction.json
    results/figures/confusion_matrix_rf_fault_prediction.png
    results/figures/mdi_rf_fault_prediction.png
    results/figures/perm_rf_fault_prediction.png
    results/figures/shap_bar_rf_fault_prediction.png
    results/figures/shap_beeswarm_rf_fault_prediction.png
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, cross_val_predict

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
    print("  Treinamento: Random Forest — Previsao de Falha (10 classes)")
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
    joblib.dump(imputer, MODELS_DIR / "imputer_rf_fault_prediction.joblib")

    # ── Busca de hiperparâmetros ──────────────────────────────────────────────
    print(f"\n[3/6] Busca de hiperparametros (RandomizedSearchCV)...")
    print(f"  N_ITER={N_ITER_SEARCH} | N_SPLITS={N_SPLITS_CV} | N_JOBS={N_JOBS}")

    param_grid = {
        "n_estimators":     [100, 200, 300],
        "max_depth":        [None, 10, 20, 30],
        "min_samples_leaf": [1, 2, 4],
        "max_features":     ["sqrt", "log2"],
    }

    rf = RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )

    gkf = GroupKFold(n_splits=N_SPLITS_CV)
    search = RandomizedSearchCV(
        estimator=rf,
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

    search.fit(X, y, groups=groups)
    best_rf     = search.best_estimator_
    best_params = search.best_params_
    best_cv_f1  = search.best_score_

    print(f"\n  Melhor F1-macro (CV): {best_cv_f1:.4f}")
    print(f"  Melhores parametros: {best_params}")

    model_path = MODELS_DIR / "rf_fault_prediction.joblib"
    joblib.dump(best_rf, model_path)
    print(f"  Modelo salvo: {model_path}")

    # ── Métricas out-of-fold ──────────────────────────────────────────────────
    print(f"\n[4/6] Gerando predicoes out-of-fold para metricas...")
    y_pred = cross_val_predict(best_rf, X, y, cv=gkf, groups=groups, n_jobs=N_JOBS, verbose=1)

    f1_macro    = f1_score(y, y_pred, average="macro",    zero_division=0)
    f1_weighted = f1_score(y, y_pred, average="weighted", zero_division=0)
    acc         = accuracy_score(y, y_pred)

    print(f"\n  F1-macro   : {f1_macro:.4f}")
    print(f"  F1-weighted: {f1_weighted:.4f}")
    print(f"  Accuracy   : {acc:.4f}")

    print_classification_report(y, y_pred, model_name="RF Previsao de Falha")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(
        y, y_pred,
        model_name="rf_fault_prediction",
        label_map=FAULT_CLASSES,
        save=True,
    )

    report = classification_report(
        y, y_pred,
        labels=sorted(classes.tolist()),
        target_names=[FAULT_CLASSES.get(c, str(c)) for c in sorted(classes.tolist())],
        output_dict=True,
        zero_division=0,
    )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {
        "model": "rf_fault_prediction",
        "trained_at": datetime.now().isoformat(),
        "dataset": {
            "n_windows":   int(len(df)),
            "n_features":  int(len(feature_cols)),
            "n_instances": int(len(np.unique(groups))),
            "n_classes":   int(len(classes)),
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
    metrics_path = METRICS_DIR / "rf_fault_prediction_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"  Metricas salvas: {metrics_path}")

    pd.DataFrame(search.cv_results_).to_csv(
        METRICS_DIR / "rf_fault_prediction_cv_results.csv", index=False
    )

    # ── Explicabilidade: Importância Nativa (MDI) ─────────────────────────────
    print("\n[5/6] Calculando importancia de features...")
    print("  [5a] MDI (Mean Decrease in Impurity)...")
    mdi_series = pd.Series(best_rf.feature_importances_, index=feature_cols).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    mdi_series.head(20).plot(kind="barh", ax=ax)
    ax.set_title("RF — Importancia MDI (top 20 features)")
    ax.set_xlabel("Importancia media (reducao de impureza)")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "mdi_rf_fault_prediction.png", dpi=150, bbox_inches="tight")
    plt.close()
    with open(METRICS_DIR / "mdi_rf_fault_prediction.json", "w") as f:
        json.dump(mdi_series.to_dict(), f, indent=2)
    print(f"    MDI salvo")

    print("  [5b] Permutation Importance (10 repeticoes, pode demorar)...")
    perm_result = permutation_importance(
        best_rf, X, y,
        n_repeats=10,
        scoring="f1_macro",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )
    perm_series = pd.Series(perm_result.importances_mean, index=feature_cols).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    perm_series.head(20).plot(kind="barh", ax=ax)
    ax.set_title("RF — Permutation Importance (top 20 features)")
    ax.set_xlabel("Queda no F1-macro apos embaralhamento")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "perm_rf_fault_prediction.png", dpi=150, bbox_inches="tight")
    plt.close()
    with open(METRICS_DIR / "perm_rf_fault_prediction.json", "w") as f:
        json.dump(perm_series.to_dict(), f, indent=2)
    print(f"    Permutation Importance salva")

    # ── Explicabilidade: SHAP ─────────────────────────────────────────────────
    print("  [5c] SHAP TreeExplainer (amostra de ate 5000 janelas)...")
    X_sample = X[:5000] if len(X) > 5000 else X
    explainer = shap.TreeExplainer(best_rf)
    shap_expl = explainer(X_sample)
    # shap_expl.values: [n_samples, n_features, n_classes]
    shap_arr = np.abs(shap_expl.values)  # [n_samples, n_features, n_classes]
    feat_scores = shap_arr.mean(axis=(0, 2))  # [n_features] — média por feature

    # Plot de barras: importância global via matplotlib
    shap_series = pd.Series(feat_scores, index=feature_cols).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 8))
    shap_series.head(20).plot(kind="barh", ax=ax)
    ax.set_title("RF — SHAP Global Importance (top 20 features)")
    ax.set_xlabel("Media de |SHAP value|")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_bar_rf_fault_prediction.png", dpi=150, bbox_inches="tight")
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
    plt.savefig(FIGURES_DIR / "shap_beeswarm_rf_fault_prediction.png", dpi=150, bbox_inches="tight")
    plt.close()

    shap_ranking = dict(zip(feature_cols, feat_scores.tolist()))
    shap_ranking = dict(sorted(shap_ranking.items(), key=lambda x: x[1], reverse=True))
    with open(METRICS_DIR / "shap_rf_fault_prediction.json", "w") as f:
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
