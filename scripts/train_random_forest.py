"""
Treinamento do Random Forest com todas as instâncias do 3W Dataset.

Execução:
    python scripts/train_random_forest.py

Saídas:
    results/models/random_forest.joblib   <- modelo treinado
    results/metrics/rf_cv_results.csv     <- resultados detalhados do CV
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
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, cross_validate

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    FAULT_CLASSES,
    FEATURES_DATA_PATH,
    METRICS_DIR,
    MODELS_DIR,
    N_ITER_SEARCH,
    N_JOBS,
    N_SPLITS_CV,
    RANDOM_STATE,
)

os.environ["PYTHONIOENCODING"] = "utf-8"


def main():
    print("=" * 60)
    print("  Treinamento: Random Forest")
    print(f"  Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ── Carregar features ─────────────────────────────────────────────────────
    print("\n[1/4] Carregando features...")
    df = pd.read_parquet(FEATURES_DATA_PATH)
    META_COLS = ["instance_id", "fault_class", "source_type", "window_start"]
    feature_cols = [c for c in df.columns if c not in META_COLS]

    X = df[feature_cols].values
    y = df["fault_class"].values
    groups = df["instance_id"].values

    print(f"  Janelas: {len(df):,}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Instancias (grupos): {len(np.unique(groups))}")
    print(f"  Classes: {sorted(np.unique(y).tolist())}")

    # ── Imputar NaN com mediana ───────────────────────────────────────────────
    print("\n[2/4] Imputando NaN com mediana...")
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)
    print(f"  NaN restantes: {np.isnan(X).sum()}")

    # Salvar imputer para reutilizar na avaliacao
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(imputer, MODELS_DIR / "imputer.joblib")

    # ── Busca de hiperparâmetros ──────────────────────────────────────────────
    print(f"\n[3/4] Busca de hiperparametros (RandomizedSearchCV)...")
    print(f"  N_ITER_SEARCH = {N_ITER_SEARCH} | N_SPLITS_CV = {N_SPLITS_CV} | N_JOBS = {N_JOBS}")

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
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
        n_jobs=1,          # paralelismo interno ao RF (n_jobs no modelo)
        random_state=RANDOM_STATE,
        verbose=2,
        refit=True,
        return_train_score=True,
    )

    search.fit(X, y, groups=groups)

    best_rf = search.best_estimator_
    best_params = search.best_params_
    best_cv_f1 = search.best_score_

    print(f"\n  Melhor F1-macro (CV): {best_cv_f1:.4f}")
    print(f"  Melhores parametros: {best_params}")

    # ── Salvar modelo ─────────────────────────────────────────────────────────
    model_path = MODELS_DIR / "random_forest.joblib"
    joblib.dump(best_rf, model_path)
    print(f"\n  Modelo salvo em: {model_path}")

    # ── Métricas detalhadas por classe (cross_validate manual) ────────────────
    print("\n[4/4] Calculando metricas detalhadas por classe...")

    all_y_true = []
    all_y_pred = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        rf_fold = RandomForestClassifier(**best_params,
                                         class_weight="balanced",
                                         random_state=RANDOM_STATE,
                                         n_jobs=N_JOBS)
        rf_fold.fit(X[train_idx], y[train_idx])
        y_pred = rf_fold.predict(X[test_idx])
        all_y_true.extend(y[test_idx])
        all_y_pred.extend(y_pred)
        f1_fold = f1_score(y[test_idx], y_pred, average="macro")
        print(f"  Fold {fold+1}: F1-macro = {f1_fold:.4f}")

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    f1_macro   = f1_score(all_y_true, all_y_pred, average="macro")
    f1_weighted = f1_score(all_y_true, all_y_pred, average="weighted")
    acc         = accuracy_score(all_y_true, all_y_pred)

    print(f"\n  F1-macro   (concat folds): {f1_macro:.4f}")
    print(f"  F1-weighted(concat folds): {f1_weighted:.4f}")
    print(f"  Accuracy   (concat folds): {acc:.4f}")

    report = classification_report(
        all_y_true, all_y_pred,
        target_names=[FAULT_CLASSES[i] for i in sorted(FAULT_CLASSES)],
        output_dict=True,
    )

    print("\n  Relatorio por classe:")
    print(classification_report(
        all_y_true, all_y_pred,
        target_names=[FAULT_CLASSES[i] for i in sorted(FAULT_CLASSES)],
    ))

    # ── Salvar métricas em JSON ───────────────────────────────────────────────
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = {
        "model": "random_forest",
        "trained_at": datetime.now().isoformat(),
        "dataset": {
            "n_windows": int(len(df)),
            "n_features": int(len(feature_cols)),
            "n_instances": int(len(np.unique(groups))),
        },
        "cv": {
            "strategy": f"GroupKFold(n_splits={N_SPLITS_CV})",
            "scoring": "f1_macro",
            "best_score_cv": round(float(best_cv_f1), 4),
        },
        "best_params": best_params,
        "metrics_concat_folds": {
            "f1_macro": round(float(f1_macro), 4),
            "f1_weighted": round(float(f1_weighted), 4),
            "accuracy": round(float(acc), 4),
        },
        "per_class": {
            str(cls): {
                "name": FAULT_CLASSES[cls],
                "precision": round(report[FAULT_CLASSES[cls]]["precision"], 4),
                "recall": round(report[FAULT_CLASSES[cls]]["recall"], 4),
                "f1": round(report[FAULT_CLASSES[cls]]["f1-score"], 4),
                "support": int(report[FAULT_CLASSES[cls]]["support"]),
            }
            for cls in sorted(FAULT_CLASSES)
        },
    }

    metrics_path = METRICS_DIR / "random_forest_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"  Metricas salvas em: {metrics_path}")

    # Salvar resultados do CV em CSV
    cv_df = pd.DataFrame(search.cv_results_)
    cv_df.to_csv(METRICS_DIR / "rf_cv_results.csv", index=False)

    print("\n" + "=" * 60)
    print("  Treinamento concluido!")
    print(f"  Fim: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
