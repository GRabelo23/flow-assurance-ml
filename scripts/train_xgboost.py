"""
Treinamento do XGBoost — Diagnóstico de falha (10 classes).

Usa features.parquet com rotulagem por instância (fault_class 0-9).

Execução:
    python scripts/train_xgboost.py

Saídas:
    results/models/xgboost.joblib
    results/models/imputer_xgb.joblib
    results/metrics/xgboost_metrics.json
    results/figures/confusion_matrix_xgboost.png
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["PYTHONIOENCODING"] = "utf-8"

from config import (
    FAULT_CLASSES,
    FEATURES_DATA_PATH,
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
    print("  Treinamento: XGBoost — Diagnostico (10 classes)")
    print(f"  Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ── Carregar features ─────────────────────────────────────────────────────
    print("\n[1/5] Carregando features...")
    if not FEATURES_DATA_PATH.exists():
        print(f"ERRO: {FEATURES_DATA_PATH} nao encontrado.")
        sys.exit(1)

    df = pd.read_parquet(FEATURES_DATA_PATH)
    META_COLS = ["instance_id", "fault_class", "source_type", "window_start"]
    feature_cols = [c for c in df.columns if c not in META_COLS]

    X      = df[feature_cols].values
    y      = df["fault_class"].values
    groups = df["instance_id"].values

    classes, counts = np.unique(y, return_counts=True)
    print(f"  Janelas: {len(df):,} | Features: {len(feature_cols)}")
    print(f"  Instancias: {len(np.unique(groups)):,} | Classes: {sorted(classes.tolist())}")
    print(f"  Distribuicao:")
    for c, n in zip(classes, counts):
        print(f"    {c} ({FAULT_CLASSES.get(c,'?'):<35}): {n:>8,} ({100*n/len(y):.1f}%)")

    # ── Imputar NaN ───────────────────────────────────────────────────────────
    print("\n[2/5] Imputando NaN com mediana...")
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(imputer, MODELS_DIR / "imputer_xgb.joblib")

    # ── Pesos de amostra (substitui class_weight='balanced') ─────────────────
    # XGBoost nao tem class_weight — compensamos com sample_weight por amostra.
    sample_weight = compute_sample_weight("balanced", y)

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
        num_class=len(classes),
        tree_method="hist",       # mais rapido em datasets grandes
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

    # sample_weight e subsetado automaticamente pelo sklearn para cada fold
    search.fit(X, y, groups=groups, sample_weight=sample_weight)
    best_xgb    = search.best_estimator_
    best_params = search.best_params_
    best_cv_f1  = search.best_score_

    print(f"\n  Melhor F1-macro (CV): {best_cv_f1:.4f}")
    print(f"  Melhores parametros: {best_params}")

    # ── Salvar modelo ─────────────────────────────────────────────────────────
    model_path = MODELS_DIR / "xgboost.joblib"
    joblib.dump(best_xgb, model_path)
    print(f"  Modelo salvo: {model_path}")

    # ── Predições out-of-fold (fold a fold, com sample_weight) ───────────────
    print(f"\n[4/5] Gerando predicoes out-of-fold para metricas...")
    y_true_all = []
    y_pred_all = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
        xgb_fold = XGBClassifier(
            **best_params,
            objective="multi:softmax",
            num_class=len(classes),
            tree_method="hist",
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            verbosity=0,
        )
        xgb_fold.fit(
            X[train_idx], y[train_idx],
            sample_weight=sample_weight[train_idx],
        )
        y_pred = xgb_fold.predict(X[test_idx])
        y_true_all.extend(y[test_idx])
        y_pred_all.extend(y_pred)
        f1_fold = f1_score(y[test_idx], y_pred, average="macro", zero_division=0)
        print(f"  Fold {fold}: F1-macro = {f1_fold:.4f}")

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    f1_macro    = f1_score(y_true_all, y_pred_all, average="macro",    zero_division=0)
    f1_weighted = f1_score(y_true_all, y_pred_all, average="weighted", zero_division=0)
    acc         = accuracy_score(y_true_all, y_pred_all)

    print(f"\n  F1-macro   : {f1_macro:.4f}")
    print(f"  F1-weighted: {f1_weighted:.4f}")
    print(f"  Accuracy   : {acc:.4f}")

    print_classification_report(y_true_all, y_pred_all, model_name="XGBoost Diagnostico")

    # ── Matriz de confusão ────────────────────────────────────────────────────
    print("\n[5/5] Plotando matriz de confusao...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(
        y_true_all, y_pred_all,
        model_name="XGBoost Diagnostico",
        save=True,
    )

    # ── Salvar métricas em JSON ───────────────────────────────────────────────
    report = classification_report(
        y_true_all, y_pred_all,
        labels=sorted(classes.tolist()),
        target_names=[FAULT_CLASSES[c] for c in sorted(classes.tolist())],
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "model": "xgboost",
        "trained_at": datetime.now().isoformat(),
        "dataset": {
            "n_windows":   int(len(df)),
            "n_features":  int(len(feature_cols)),
            "n_instances": int(len(np.unique(groups))),
            "n_classes":   int(len(classes)),
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
                "name":      FAULT_CLASSES[c],
                "precision": round(report[FAULT_CLASSES[c]]["precision"], 4),
                "recall":    round(report[FAULT_CLASSES[c]]["recall"], 4),
                "f1":        round(report[FAULT_CLASSES[c]]["f1-score"], 4),
                "support":   int(report[FAULT_CLASSES[c]]["support"]),
            }
            for c in sorted(classes.tolist())
        },
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = METRICS_DIR / "xgboost_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"  Metricas salvas: {metrics_path}")

    pd.DataFrame(search.cv_results_).to_csv(
        METRICS_DIR / "xgboost_cv_results.csv", index=False
    )

    print("\n" + "=" * 60)
    print("  Treinamento concluido!")
    print(f"  Fim: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
