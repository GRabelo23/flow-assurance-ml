"""
XGBoost — Estado Operacional com filtro estatístico adaptativo (sigma=0.5).

O filtro estatístico (FiltroPID) aplica suavização adaptativa:
  alpha = erf(|x_anterior - u| / (2*sqrt(2)*sigma))
  saida = (1 - alpha)*x_anterior + alpha*u

Com sigma=0.5 (threshold ≈ 1.41 z-score), o filtro:
  - Suaviza fortemente variações pequenas (ruído de sensor)
  - Preserva transientes grandes (mudanças > 1.4 z-score entre pontos)

Execução:
    python scripts/train_xgboost_statistical.py

Saídas:
    data/processed/features_statistical_window_class.parquet
    results/models/xgboost_statistical_window_class.joblib
    results/models/imputer_statistical_window_class.joblib
    results/metrics/xgboost_statistical_metrics.json
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
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["PYTHONIOENCODING"] = "utf-8"

from config import (
    CLEANED_DATA_PATH,
    METRICS_DIR,
    MODELS_DIR,
    N_ITER_SEARCH,
    N_JOBS,
    N_SPLITS_CV,
    PROCESSED_DIR,
    RANDOM_STATE,
    STATISTICAL_SIGMA,
    WINDOW_CLASSES,
)
from src.evaluation import plot_confusion_matrix, print_classification_report
from src.feature_engineering import run_pipeline_from_cleaned

FEATURES_STAT_PATH = PROCESSED_DIR / "features_statistical_window_class.parquet"
MODEL_PATH         = MODELS_DIR / "xgboost_statistical_window_class.joblib"
IMPUTER_PATH       = MODELS_DIR / "imputer_statistical_window_class.joblib"
METRICS_PATH       = METRICS_DIR / "xgboost_statistical_metrics.json"


def gerar_features():
    print(f"\n[1/5] Gerando features com filtro estatístico (sigma={STATISTICAL_SIGMA})...")
    print(f"  Lendo: {CLEANED_DATA_PATH}")
    run_pipeline_from_cleaned(
        cleaned_path=CLEANED_DATA_PATH,
        features_path=FEATURES_STAT_PATH,
        label_strategy="window",
        smooth_filter="statistical",
        verbose=True,
    )
    print(f"  Features salvas: {FEATURES_STAT_PATH}")


def main():
    print("=" * 65)
    print(f"  XGBoost — Estado Operacional  |  filtro estatístico (sigma={STATISTICAL_SIGMA})")
    print(f"  Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # ── Etapa 1: gerar features ───────────────────────────────────────────────
    if FEATURES_STAT_PATH.exists():
        print(f"\n[1/5] Features ja existem: {FEATURES_STAT_PATH}")
        print("  Delete o arquivo para regerar. Usando versao existente.")
    else:
        gerar_features()

    # ── Etapa 2: carregar e preparar dados ────────────────────────────────────
    print("\n[2/5] Carregando features...")
    df = pd.read_parquet(FEATURES_STAT_PATH)

    META_COLS    = ["instance_id", "fault_class", "window_label", "source_type", "window_start"]
    feature_cols = [c for c in df.columns if c not in META_COLS]

    X      = df[feature_cols].values
    y_orig = df["window_label"].values
    groups = df["instance_id"].values

    classes_orig, counts = np.unique(y_orig, return_counts=True)
    print(f"  Janelas: {len(df):,} | Features: {len(feature_cols)}")
    print(f"  Classes: {len(classes_orig)} — {sorted(classes_orig.tolist())}")
    for c, n in zip(classes_orig, counts):
        print(f"    {c:4d} ({WINDOW_CLASSES.get(c,'?'):<22}): {n:>8,}  ({100*n/len(y_orig):.1f}%)")

    print("\n  Imputando NaN com mediana...")
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(imputer, IMPUTER_PATH)

    le = LabelEncoder()
    y_enc = le.fit_transform(y_orig)
    n_classes = len(le.classes_)
    print(f"  LabelEncoder: {n_classes} classes -> [0, {n_classes-1}]")

    sample_weight = compute_sample_weight("balanced", y_orig)

    # ── Etapa 3: busca de hiperparâmetros ─────────────────────────────────────
    print(f"\n[3/5] Busca de hiperparametros (N_ITER={N_ITER_SEARCH}, N_SPLITS={N_SPLITS_CV})...")

    param_grid = {
        "n_estimators":     [100, 200, 300, 500],
        "max_depth":        [3, 4, 6, 8],
        "learning_rate":    [0.01, 0.05, 0.1, 0.2],
        "subsample":        [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
    }

    xgb_base = XGBClassifier(
        objective="multi:softmax",
        num_class=n_classes,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=3,
        verbosity=0,
    )

    gkf = GroupKFold(n_splits=N_SPLITS_CV)
    search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=param_grid,
        n_iter=N_ITER_SEARCH,
        cv=gkf,
        scoring="f1_macro",
        n_jobs=4,
        random_state=RANDOM_STATE,
        verbose=2,
        refit=True,
    )
    search.fit(X, y_enc, groups=groups, sample_weight=sample_weight)

    best_xgb    = search.best_estimator_
    best_params = search.best_params_
    best_cv_f1  = search.best_score_

    print(f"\n  Melhor F1-macro (CV): {best_cv_f1:.4f}")
    print(f"  Melhores parametros: {best_params}")

    joblib.dump(best_xgb, MODEL_PATH)
    print(f"  Modelo salvo: {MODEL_PATH}")

    # ── Etapa 4: predições out-of-fold ────────────────────────────────────────
    print(f"\n[4/5] Predicoes out-of-fold...")
    y_true_enc_all, y_pred_enc_all = [], []

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
        xgb_fold.fit(X[train_idx], y_enc[train_idx],
                     sample_weight=sample_weight[train_idx])
        y_pred_enc = xgb_fold.predict(X[test_idx])
        y_true_enc_all.extend(y_enc[test_idx])
        y_pred_enc_all.extend(y_pred_enc)

        y_true_fold = le.inverse_transform(y_enc[test_idx])
        y_pred_fold = le.inverse_transform(y_pred_enc)
        f1_fold = f1_score(y_true_fold, y_pred_fold, average="macro", zero_division=0)
        print(f"  Fold {fold}: F1-macro = {f1_fold:.4f}")

    y_true_all = le.inverse_transform(np.array(y_true_enc_all))
    y_pred_all = le.inverse_transform(np.array(y_pred_enc_all))

    f1_macro    = f1_score(y_true_all, y_pred_all, average="macro",    zero_division=0)
    f1_weighted = f1_score(y_true_all, y_pred_all, average="weighted", zero_division=0)
    acc         = accuracy_score(y_true_all, y_pred_all)

    print(f"\n  F1-macro   : {f1_macro:.4f}")
    print(f"  F1-weighted: {f1_weighted:.4f}")
    print(f"  Accuracy   : {acc:.4f}")

    # ── Comparação com os outros dois modelos ─────────────────────────────────
    print("\n  Comparacao com outros filtros:")
    for nome, path in [("Gaussiano ", "xgboost_window_class_metrics.json"),
                       ("Sem filtro", "xgboost_nofilter_metrics.json")]:
        ref_path = METRICS_DIR / path
        if ref_path.exists():
            with open(ref_path, encoding="utf-8") as f:
                ref = json.load(f)
            ref_f1 = ref["metrics_concat_folds"]["f1_macro"]
            delta  = round(f1_macro - ref_f1, 4)
            sinal  = "+" if delta >= 0 else ""
            print(f"    {nome}: {ref_f1:.4f}  ->  Estatístico: {f1_macro:.4f}  (delta={sinal}{delta})")

    print_classification_report(y_true_all, y_pred_all,
                                 model_name="XGBoost Filtro Estatístico")

    # ── Etapa 5: salvar métricas ───────────────────────────────────────────────
    print("\n[5/5] Salvando metricas e figuras...")

    report = classification_report(
        y_true_all, y_pred_all,
        labels=sorted(classes_orig.tolist()),
        target_names=[WINDOW_CLASSES.get(c, str(c)) for c in sorted(classes_orig.tolist())],
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "model":            "xgboost_statistical_window_class",
        "filter":           "statistical",
        "statistical_sigma": STATISTICAL_SIGMA,
        "trained_at":       datetime.now().isoformat(),
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
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"  Metricas salvas: {METRICS_PATH}")

    plot_confusion_matrix(
        y_true_all, y_pred_all,
        model_name="XGBoost Filtro Estatístico",
        label_map=WINDOW_CLASSES,
        save=True,
    )

    print("\n" + "=" * 65)
    print("  Concluido!")
    print(f"  Fim: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)


if __name__ == "__main__":
    main()
