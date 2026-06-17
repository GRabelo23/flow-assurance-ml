"""
XGBoost — Previsão de Falha a partir de Operação Normal (10 classes).

Usa apenas janelas com window_label=0 (operação normal) do parquet sem filtro,
rotuladas com a fault_class da instância. A tarefa é responder:
"dado que o poço está em operação normal, qual falha ele vai desenvolver?"

  0       → Poço genuinamente normal (nunca terá falha)
  1–9     → Poço que desenvolverá a falha do tipo correspondente

Nota: fault_class é [0,1,2,5,6,7,8,9] (não contíguo) → LabelEncoder mapeia para [0..7].

Execução:
    python scripts/train_xgboost_fault_prediction.py

Saídas:
    results/models/xgboost_fault_prediction.joblib
    results/models/imputer_xgb_fault_prediction.joblib
    results/models/label_encoder_xgb_fault_prediction.joblib
    results/metrics/xgboost_fault_prediction_metrics.json
    results/metrics/xgboost_fault_prediction_cv_results.csv
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["PYTHONIOENCODING"] = "utf-8"

from config import (
    FAULT_CLASSES,
    FEATURES_NOFILTER_PATH,
    METRICS_DIR,
    MODELS_DIR,
    N_ITER_SEARCH,
    N_JOBS,
    N_SPLITS_CV,
    RANDOM_STATE,
)
from src.evaluation import print_classification_report



def main():
    print("=" * 60)
    print("  Treinamento: XGBoost — Previsao de Falha (10 classes)")
    print("  Entrada: apenas janelas de operacao normal (window_label=0)")
    print(f"  Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ── Carregar e filtrar features ───────────────────────────────────────────
    print("\n[1/4] Carregando features (sem filtro de sinal)...")
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

    X_raw  = df[feature_cols].values
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

    sample_weight = compute_sample_weight("balanced", y_enc)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Busca de hiperparâmetros ──────────────────────────────────────────────
    print(f"\n[2/4] Busca de hiperparametros (RandomizedSearchCV)...")
    print(f"  N_ITER={N_ITER_SEARCH} | N_SPLITS={N_SPLITS_CV} | N_JOBS={N_JOBS}")

    # Prefixo clf__ garante que o imputer seja fit apenas nos dados de treino
    # em cada fold, evitando vazamento de dados (data leakage)
    param_grid = {
        "clf__n_estimators":     [100, 200, 300, 500],
        "clf__max_depth":        [3, 4, 6, 8],
        "clf__learning_rate":    [0.01, 0.05, 0.1, 0.2],
        "clf__subsample":        [0.7, 0.8, 1.0],
        "clf__colsample_bytree": [0.7, 0.8, 1.0],
        "clf__min_child_weight": [1, 3, 5],
    }

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", XGBClassifier(
            objective="multi:softmax",
            num_class=n_classes,
            tree_method="hist",
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            verbosity=0,
        )),
    ])

    gkf = GroupKFold(n_splits=N_SPLITS_CV)
    search = RandomizedSearchCV(
        estimator=pipe,
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

    search.fit(X_raw, y_enc, groups=groups, clf__sample_weight=sample_weight)
    best_pipe   = search.best_estimator_
    best_xgb    = best_pipe.named_steps["clf"]
    best_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
    best_cv_f1  = search.best_score_

    print(f"\n  Melhor F1-macro (CV): {best_cv_f1:.4f}")
    print(f"  Melhores parametros: {best_params}")

    model_path = MODELS_DIR / "xgboost_fault_prediction.joblib"
    joblib.dump(best_pipe, model_path)
    joblib.dump(best_pipe.named_steps["imputer"], MODELS_DIR / "imputer_xgb_fault_prediction.joblib")
    joblib.dump(le, MODELS_DIR / "label_encoder_xgb_fault_prediction.joblib")
    print(f"  Modelo salvo: {model_path}")

    # ── Predições out-of-fold ─────────────────────────────────────────────────
    print(f"\n[3/4] Gerando predicoes out-of-fold para metricas...")
    y_true_all = []
    y_pred_all = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_raw, y, groups), 1):
        pipe_fold = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", XGBClassifier(
                **best_params,
                objective="multi:softmax",
                num_class=n_classes,
                tree_method="hist",
                eval_metric="mlogloss",
                random_state=RANDOM_STATE,
                n_jobs=N_JOBS,
                verbosity=0,
            )),
        ])
        pipe_fold.fit(
            X_raw[train_idx], y_enc[train_idx],
            clf__sample_weight=sample_weight[train_idx],
        )
        y_pred_fold = le.inverse_transform(pipe_fold.predict(X_raw[test_idx]))
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

    print("\n[4/4] Concluido!")
    print("=" * 60)
    print("  Treinamento concluido!")
    print(f"  Fim: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  F1-macro OOF: {f1_macro:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
