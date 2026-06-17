"""
Gera gráficos de importância de features e matrizes de confusão para a abordagem
de previsão de falha a partir de operação normal (10 classes, window_label == 0).

  [1] RF  — Feature Importance (MDI)
  [2] RF  — Permutation Importance (boxplot)
  [3] RF  — SHAP Global Importance
  [4] RF  — Matriz de Confusão
  [5] XGB — Gain / Weight / Cover (figura única com 3 subplots, top 15 por Gain)
  [6] XGB — SHAP Global Importance
  [7] XGB — Matriz de Confusão

Carrega os modelos já treinados — não re-treina.

Execução:
    python fault_prediction/scripts/plot_feature_importance.py

Saídas:
    fault_prediction/figures/rf/rf_mdi_fault_prediction.png
    fault_prediction/figures/rf/rf_perm_boxplot_fault_prediction.png
    fault_prediction/figures/rf/rf_shap_fault_prediction.png
    fault_prediction/figures/rf/rf_confusion_matrix_fault_prediction.png
    fault_prediction/figures/xgb/xgb_importance_fault_prediction.png
    fault_prediction/figures/xgb/xgb_shap_fault_prediction.png
    fault_prediction/figures/xgb/xgb_confusion_matrix_fault_prediction.png
"""

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold, cross_val_predict

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    FAULT_CLASSES,
    FEATURES_NOFILTER_PATH,
    MODELS_DIR,
    N_JOBS,
    N_SPLITS_CV,
    RANDOM_STATE,
)

FIG_RF  = Path(__file__).parent.parent / "figures" / "rf"
FIG_XGB = Path(__file__).parent.parent / "figures" / "xgb"

TOP_N        = 15
PERM_SAMPLE  = 10_000
PERM_REPEATS = 10
SHAP_SAMPLE  = 5_000

META_COLS = ["instance_id", "fault_class", "window_label", "source_type", "window_start"]

_TITLE_KW  = dict(fontsize=13, fontweight="bold", pad=10)
_XLABEL_KW = dict(fontsize=10)


def load_data():
    print("Carregando features (sem filtro, apenas window_label == 0)...")
    df = pd.read_parquet(FEATURES_NOFILTER_PATH)
    df = df[df["window_label"] == 0].copy()
    feature_cols = [c for c in df.columns if c not in META_COLS]
    X_raw  = df[feature_cols].values
    y      = df["fault_class"].values
    groups = df["instance_id"].values
    print(f"  {len(df):,} janelas | {len(feature_cols)} features | {len(np.unique(y))} classes")
    return X_raw, y, groups, feature_cols

def plot_rf_mdi(rf, feature_cols):
    print("\n[1/7] RF — MDI (Mean Decrease in Impurity)...")
    mdi = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    top = mdi.head(TOP_N)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.Blues_r(np.linspace(0.25, 0.80, TOP_N))
    top[::-1].plot(kind="barh", ax=ax, color=colors[::-1])

    ax.set_title("RF — Feature Importance (MDI)", **_TITLE_KW)
    ax.set_xlabel("Importância média (redução de impureza Gini)", **_XLABEL_KW)
    ax.tick_params(axis="y", labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()

    out = FIG_RF / "rf_mdi_fault_prediction.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Salvo: {out}")

def plot_rf_perm_boxplot(rf, X_imp, y, feature_cols):
    print(f"\n[2/7] RF — Permutation Importance "
          f"({PERM_REPEATS} repeats, amostra={PERM_SAMPLE:,})...")

    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X_imp), size=min(PERM_SAMPLE, len(X_imp)), replace=False)

    result = permutation_importance(
        rf, X_imp[idx], y[idx],
        n_repeats=PERM_REPEATS,
        scoring="f1_macro",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )

    mean_series  = pd.Series(result.importances_mean, index=feature_cols)
    top_features = mean_series.nlargest(TOP_N).index.tolist()
    feat_to_idx  = {f: i for i, f in enumerate(feature_cols)}
    imp_matrix   = result.importances[[feat_to_idx[f] for f in top_features], :]

    fig, ax = plt.subplots(figsize=(9, 6))
    bp = ax.boxplot(
        imp_matrix[::-1].T,
        vert=False,
        patch_artist=True,
        tick_labels=top_features[::-1],
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=3, alpha=0.5),
    )
    colors = plt.cm.Blues_r(np.linspace(0.25, 0.80, TOP_N))
    for patch, color in zip(bp["boxes"], colors[::-1]):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    ax.set_title("RF — Permutation Importance", **_TITLE_KW)
    ax.set_xlabel("Queda no F1-macro após embaralhamento", **_XLABEL_KW)
    ax.tick_params(axis="y", labelsize=9)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()

    out = FIG_RF / "rf_perm_boxplot_fault_prediction.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Salvo: {out}")

def plot_rf_shap(rf, X_imp, feature_cols):
    print(f"\n[3/7] RF — SHAP Global Importance (amostra={SHAP_SAMPLE:,})...")

    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X_imp), size=min(SHAP_SAMPLE, len(X_imp)), replace=False)
    X_sample = X_imp[idx]

    explainer   = shap.TreeExplainer(rf)
    shap_expl   = explainer(X_sample)
    shap_arr    = np.abs(shap_expl.values)    # [n_samples, n_features, n_classes]
    feat_scores = shap_arr.mean(axis=(0, 2))  # [n_features]

    shap_series = pd.Series(feat_scores, index=feature_cols).sort_values(ascending=False)
    top = shap_series.head(TOP_N)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.Blues_r(np.linspace(0.25, 0.80, TOP_N))
    top[::-1].plot(kind="barh", ax=ax, color=colors[::-1])

    ax.set_title("RF — SHAP Global Importance", **_TITLE_KW)
    ax.set_xlabel("Média de |SHAP value|", **_XLABEL_KW)
    ax.tick_params(axis="y", labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()

    out = FIG_RF / "rf_shap_fault_prediction.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Salvo: {out}")

def plot_confusion_matrix_custom(y_true, y_pred, title, out_path):
    classes     = sorted(np.unique(y_true).tolist())
    cm_abs      = confusion_matrix(y_true, y_pred, labels=classes)
    cm_norm     = cm_abs.astype(float) / cm_abs.sum(axis=1, keepdims=True)
    class_names = [FAULT_CLASSES.get(c, str(c)) for c in classes]
    n = len(classes)

    fig, ax = plt.subplots(figsize=(max(8, n * 0.9), max(6, n * 0.8)))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predito", **_XLABEL_KW)
    ax.set_ylabel("Real", **_XLABEL_KW)
    ax.set_title(title, **_TITLE_KW)

    thresh = 0.5
    for i in range(n):
        for j in range(n):
            val = cm_norm[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if val > thresh else "black")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Salvo: {out_path}")

def plot_xgb_combined(xgb, feature_cols):
    print("\n[5/7] XGBoost — Gain / Weight / Cover (figura única)...")
    booster = xgb.get_booster()
    booster.feature_names = feature_cols

    scores = {}
    for imp_type in ["gain", "weight", "cover"]:
        raw = booster.get_score(importance_type=imp_type)
        scores[imp_type] = pd.Series({f: raw.get(f, 0.0) for f in feature_cols})

    top_features = scores["gain"].nlargest(TOP_N).index.tolist()

    xlabels = {
        "gain":   "Ganho médio por divisão",
        "weight": "Frequência de uso",
        "cover":  "Amostras cobertas em média",
    }

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    fig.suptitle("XGB — Feature Importance", fontsize=14, fontweight="bold")

    colors = plt.cm.Oranges_r(np.linspace(0.25, 0.80, TOP_N))

    for col, (ax, imp_type) in enumerate(zip(axes, ["gain", "weight", "cover"])):
        vals = scores[imp_type][top_features][::-1]
        vals.plot(kind="barh", ax=ax, color=colors[::-1])

        ax.set_title(imp_type.capitalize(), fontsize=12, fontweight="bold")
        ax.set_xlabel(xlabels[imp_type], fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        ax.tick_params(axis="y", labelsize=8)

        if col > 0:
            ax.set_yticklabels([])

    plt.tight_layout()

    out = FIG_XGB / "xgb_importance_fault_prediction.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Salvo: {out}")

def plot_xgb_shap(xgb, X_imp, feature_cols):
    print(f"\n[6/7] XGB — SHAP Global Importance (amostra={SHAP_SAMPLE:,})...")

    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X_imp), size=min(SHAP_SAMPLE, len(X_imp)), replace=False)
    X_sample = X_imp[idx]

    explainer   = shap.TreeExplainer(xgb)
    shap_expl   = explainer(X_sample)
    shap_arr    = np.abs(shap_expl.values)    # [n_samples, n_features, n_classes]
    feat_scores = shap_arr.mean(axis=(0, 2))  # [n_features]

    shap_series = pd.Series(feat_scores, index=feature_cols).sort_values(ascending=False)
    top = shap_series.head(TOP_N)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.Oranges_r(np.linspace(0.25, 0.80, TOP_N))
    top[::-1].plot(kind="barh", ax=ax, color=colors[::-1])

    ax.set_title("XGB — SHAP Global Importance", **_TITLE_KW)
    ax.set_xlabel("Média de |SHAP value|", **_XLABEL_KW)
    ax.tick_params(axis="y", labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()

    out = FIG_XGB / "xgb_shap_fault_prediction.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Salvo: {out}")

def main():
    FIG_RF.mkdir(parents=True, exist_ok=True)
    FIG_XGB.mkdir(parents=True, exist_ok=True)

    X_raw, y, groups, feature_cols = load_data()
    gkf = GroupKFold(n_splits=N_SPLITS_CV)

    # Random Forest
    print("\nCarregando modelo RF...")
    rf      = joblib.load(MODELS_DIR / "rf_fault_prediction.joblib")
    imputer = joblib.load(MODELS_DIR / "imputer_rf_fault_prediction.joblib")

    if hasattr(rf, "named_steps"):
        rf_clf = rf.named_steps["clf"]
        X_imp  = rf.named_steps["imputer"].transform(X_raw)
    else:
        rf_clf = rf
        X_imp  = imputer.transform(X_raw)

    plot_rf_mdi(rf_clf, feature_cols)
    plot_rf_perm_boxplot(rf_clf, X_imp, y, feature_cols)
    plot_rf_shap(rf_clf, X_imp, feature_cols)

    print("\n[4/7] RF — Matriz de Confusão (predicoes OOF)...")
    if hasattr(rf, "named_steps"):
        y_pred_rf = cross_val_predict(
            rf, X_raw, y, cv=gkf, groups=groups, n_jobs=N_JOBS, verbose=1
        )
    else:
        y_pred_rf = cross_val_predict(
            rf_clf, X_imp, y, cv=gkf, groups=groups, n_jobs=N_JOBS, verbose=1
        )
    plot_confusion_matrix_custom(
        y, y_pred_rf,
        title="RF — Matriz de Confusão",
        out_path=FIG_RF / "rf_confusion_matrix_fault_prediction.png",
    )

    # XGBoost
    print("\nCarregando modelo XGBoost...")
    xgb         = joblib.load(MODELS_DIR / "xgboost_fault_prediction.joblib")
    imputer_xgb = joblib.load(MODELS_DIR / "imputer_xgb_fault_prediction.joblib")
    le          = joblib.load(MODELS_DIR / "label_encoder_xgb_fault_prediction.joblib")

    if hasattr(xgb, "named_steps"):
        xgb_clf   = xgb.named_steps["clf"]
        X_imp_xgb = xgb.named_steps["imputer"].transform(X_raw)
    else:
        xgb_clf   = xgb
        X_imp_xgb = imputer_xgb.transform(X_raw)

    plot_xgb_combined(xgb_clf, feature_cols)
    plot_xgb_shap(xgb_clf, X_imp_xgb, feature_cols)

    print("\n[7/7] XGB — Matriz de Confusão (predicoes OOF)...")
    y_enc = le.transform(y)
    if hasattr(xgb, "named_steps"):
        y_pred_enc = cross_val_predict(
            xgb, X_raw, y_enc, cv=gkf, groups=groups, n_jobs=N_JOBS, verbose=1
        )
    else:
        y_pred_enc = cross_val_predict(
            xgb_clf, X_imp_xgb, y_enc, cv=gkf, groups=groups, n_jobs=N_JOBS, verbose=1
        )
    y_pred_xgb = le.inverse_transform(y_pred_enc)
    plot_confusion_matrix_custom(
        y, y_pred_xgb,
        title="XGB — Matriz de Confusão",
        out_path=FIG_XGB / "xgb_confusion_matrix_fault_prediction.png",
    )

    print("\nConcluído!")


if __name__ == "__main__":
    main()
