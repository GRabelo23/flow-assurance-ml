"""
Treinamento do Random Forest com rotulagem por estado operacional (19 classes).

Usa features_window_class.parquet onde cada janela é rotulada com a moda
da coluna 'class' dentro da janela:
  0        → Normal
  1–9      → Evento ativo do tipo correspondente
  101–109  → Transiente do tipo correspondente

Execução:
    python scripts/train_rf_window_class.py

Saídas:
    results/models/rf_window_class.joblib
    results/models/imputer_window_class.joblib
    results/metrics/rf_window_class_metrics.json
    results/figures/confusion_matrix_rf_window_class.png
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, cross_val_predict

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["PYTHONIOENCODING"] = "utf-8"

from config import (
    FEATURES_WINDOW_PATH,
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


def main():
    print("=" * 60)
    print("  Treinamento: Random Forest — Estado Operacional (19 classes)")
    print(f"  Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ── Carregar features ─────────────────────────────────────────────────────
    print("\n[1/5] Carregando features...")
    if not FEATURES_WINDOW_PATH.exists():
        print(f"ERRO: {FEATURES_WINDOW_PATH} nao encontrado.")
        print("Execute primeiro: python scripts/run_pipeline_window_class.py")
        sys.exit(1)

    df = pd.read_parquet(FEATURES_WINDOW_PATH)
    META_COLS = ["instance_id", "fault_class", "window_label", "source_type", "window_start"]
    feature_cols = [c for c in df.columns if c not in META_COLS]

    X      = df[feature_cols].values
    y      = df["window_label"].values
    groups = df["instance_id"].values

    classes, counts = np.unique(y, return_counts=True)
    print(f"  Janelas: {len(df):,} | Features: {len(feature_cols)}")
    print(f"  Classes unicas: {len(classes)} — {sorted(classes.tolist())}")
    print(f"  Distribuicao:")
    for c, n in zip(classes, counts):
        pct = 100 * n / len(y)
        print(f"    {c:4d} ({WINDOW_CLASSES.get(c, '?'):<22}): {n:>8,} ({pct:.1f}%)")

    # ── Imputar NaN ───────────────────────────────────────────────────────────
    print("\n[2/5] Imputando NaN com mediana...")
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(imputer, MODELS_DIR / "imputer_window_class.joblib")

    # ── Busca de hiperparâmetros ──────────────────────────────────────────────
    print(f"\n[3/5] Busca de hiperparametros (RandomizedSearchCV)...")
    print(f"  N_ITER={N_ITER_SEARCH} | N_SPLITS={N_SPLITS_CV} | N_JOBS={N_JOBS}")

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth":    [None, 10, 20, 30],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }

    rf = RandomForestClassifier(
        class_weight="balanced",   # compensa desbalanceamento das 19 classes
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

    # ── Salvar modelo ─────────────────────────────────────────────────────────
    model_path = MODELS_DIR / "rf_window_class.joblib"
    joblib.dump(best_rf, model_path)
    print(f"  Modelo salvo: {model_path}")

    # ── Métricas detalhadas via cross_val_predict ─────────────────────────────
    print(f"\n[4/5] Gerando predicoes out-of-fold para metricas...")
    y_pred = cross_val_predict(best_rf, X, y, cv=gkf, groups=groups, n_jobs=N_JOBS, verbose=1)

    f1_macro    = f1_score(y, y_pred, average="macro",    zero_division=0)
    f1_weighted = f1_score(y, y_pred, average="weighted", zero_division=0)
    acc         = accuracy_score(y, y_pred)

    print(f"\n  F1-macro   : {f1_macro:.4f}")
    print(f"  F1-weighted: {f1_weighted:.4f}")
    print(f"  Accuracy   : {acc:.4f}")

    print_classification_report(y, y_pred, model_name="RF Estado Operacional")

    # ── Matriz de confusão ────────────────────────────────────────────────────
    print("\n[5/5] Plotando matriz de confusao...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(
        y, y_pred,
        model_name="RF Estado Operacional",
        label_map=WINDOW_CLASSES,
        save=True,
    )

    # ── Salvar métricas em JSON ───────────────────────────────────────────────
    report = classification_report(
        y, y_pred,
        labels=sorted(classes.tolist()),
        target_names=[WINDOW_CLASSES.get(c, str(c)) for c in sorted(classes.tolist())],
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "model": "rf_window_class",
        "trained_at": datetime.now().isoformat(),
        "dataset": {
            "n_windows":   int(len(df)),
            "n_features":  int(len(feature_cols)),
            "n_instances": int(len(np.unique(groups))),
            "n_classes":   int(len(classes)),
        },
        "cv": {
            "strategy":       f"GroupKFold(n_splits={N_SPLITS_CV})",
            "scoring":        "f1_macro",
            "best_score_cv":  round(float(best_cv_f1), 4),
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
            for c in sorted(classes.tolist())
        },
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = METRICS_DIR / "rf_window_class_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"  Metricas salvas: {metrics_path}")

    pd.DataFrame(search.cv_results_).to_csv(
        METRICS_DIR / "rf_window_class_cv_results.csv", index=False
    )

    print("\n" + "=" * 60)
    print("  Treinamento concluido!")
    print(f"  Fim: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
