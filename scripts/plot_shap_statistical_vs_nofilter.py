"""
Gera gráficos SHAP comparando XGBoost Filtro Estatístico vs XGBoost Sem Filtro.

Para cada classe, produz um bee swarm side-by-side mostrando as top-15 features
de cada modelo, facilitando a comparação do efeito do filtro estatístico.

Saídas:
    results/figures/shap/statistical_vs_nofilter/shap_classe{C}.png  (17 figuras)
    results/figures/shap/statistical_vs_nofilter/shap_resumo_comparativo.png
"""

import os
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["PYTHONIOENCODING"] = "utf-8"

from config import MODELS_DIR, PROCESSED_DIR, FIGURES_DIR, WINDOW_CLASSES

OUT_DIR = FIGURES_DIR / "shap" / "statistical_vs_nofilter"
OUT_DIR.mkdir(parents=True, exist_ok=True)

META_COLS = ["instance_id", "fault_class", "window_label", "source_type", "window_start"]
N_SAMPLE  = 3000
RANDOM_STATE = 42
TOP_N = 15

CLASSES_PRESENT = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 101, 102, 105, 106, 107, 108, 109]


def load_model_and_data(features_path, model_path, imputer_path):
    print(f"  Carregando features: {features_path.name}")
    df = pd.read_parquet(features_path)
    feature_cols = [c for c in df.columns if c not in META_COLS]
    X_raw = df[feature_cols].values
    y     = df["window_label"].values

    imputer = joblib.load(imputer_path)
    X = imputer.transform(X_raw)

    model = joblib.load(model_path)
    le    = model.get_booster().feature_names   # não usado diretamente
    return model, X, y, feature_cols


def stratified_sample(X, y, n=N_SAMPLE):
    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=n, random_state=RANDOM_STATE)
        _, idx = next(sss.split(X, y))
    except ValueError:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    return X[idx], y[idx]


def compute_shap(model, X_sample):
    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer(X_sample)       # (n, n_feat, n_classes)
    return shap_vals


def make_class_index_map(y):
    """Mapeia label original → índice interno do XGBoost (LabelEncoder order)."""
    return {c: i for i, c in enumerate(sorted(np.unique(y)))}

# mapas global construídos após carregar os dados
cls_map_stat: dict = {}
cls_map_nof:  dict = {}

def class_index(cls_map, class_label):
    return cls_map[class_label]


def plot_comparison(shap_stat, shap_nof, X_stat, X_nof,
                    feat_stat, feat_nof, cls, save_path):
    idx_stat = class_index(cls_map_stat, cls)
    idx_nof  = class_index(cls_map_nof,  cls)

    vals_stat = shap_stat.values[:, :, idx_stat]   # (n, n_feat)
    vals_nof  = shap_nof.values[:, :, idx_nof]

    top_stat = np.argsort(np.abs(vals_stat).mean(axis=0))[::-1][:TOP_N]
    top_nof  = np.argsort(np.abs(vals_nof).mean(axis=0))[::-1][:TOP_N]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    cls_name  = WINDOW_CLASSES.get(cls, str(cls))
    fig.suptitle(f"SHAP — Classe {cls}: {cls_name}\nFiltro Estatístico (σ=0.5)  vs  Sem Filtro",
                 fontsize=13, fontweight="bold")

    for ax, vals, top_idx, feat_names, title in [
        (axes[0], vals_stat, top_stat, feat_stat, "Filtro Estatístico"),
        (axes[1], vals_nof,  top_nof,  feat_nof,  "Sem Filtro"),
    ]:
        feat_labels = [feat_names[i] for i in top_idx]
        mean_abs    = np.abs(vals[:, top_idx]).mean(axis=0)
        colors = ["#e05252" if v > 0 else "#5278e0"
                  for v in vals[:, top_idx].mean(axis=0)]

        bars = ax.barh(range(TOP_N), mean_abs[::-1], color=colors[::-1], height=0.7)
        ax.set_yticks(range(TOP_N))
        ax.set_yticklabels(feat_labels[::-1], fontsize=8)
        ax.set_xlabel("mean |SHAP value|", fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_summary_delta(f1_stat, f1_nof):
    """Barplot de delta F1 (estatístico − sem-filtro) por classe."""
    classes = sorted(f1_stat.keys())
    deltas  = [f1_stat[c] - f1_nof[c] for c in classes]
    labels  = [f"{c}\n{WINDOW_CLASSES.get(c,'?')[:12]}" for c in classes]
    colors  = ["#2ca02c" if d >= 0 else "#d62728" for d in deltas]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(classes)), deltas, color=colors, width=0.6)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("ΔF1 (Estatístico − Sem Filtro)", fontsize=10)
    ax.set_title("Impacto do Filtro Estatístico por Classe\n(verde = melhora, vermelho = piora)",
                 fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "shap_delta_f1_statistical_vs_nofilter.png",
                dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: shap_delta_f1_statistical_vs_nofilter.png")


# ── Main ──────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  SHAP: Filtro Estatístico vs Sem Filtro")
print("=" * 60)

print("\n[1/4] Carregando modelos e dados...")
model_stat = joblib.load(MODELS_DIR / "xgboost_statistical_window_class.joblib")
model_nof  = joblib.load(MODELS_DIR / "xgboost_nofilter_window_class.joblib")

df_stat = pd.read_parquet(PROCESSED_DIR / "features_statistical_window_class.parquet")
df_nof  = pd.read_parquet(PROCESSED_DIR / "features_nofilter_window_class.parquet")

feat_stat = [c for c in df_stat.columns if c not in META_COLS]
feat_nof  = [c for c in df_nof.columns  if c not in META_COLS]

imp_stat = joblib.load(MODELS_DIR / "imputer_statistical_window_class.joblib")
imp_nof  = joblib.load(MODELS_DIR / "imputer_nofilter_window_class.joblib")

X_stat_full = imp_stat.transform(df_stat[feat_stat].values)
X_nof_full  = imp_nof.transform(df_nof[feat_nof].values)
y_stat = df_stat["window_label"].values
y_nof  = df_nof["window_label"].values

cls_map_stat = make_class_index_map(df_stat["window_label"].values)
cls_map_nof  = make_class_index_map(df_nof["window_label"].values)
print(f"  Stat: {len(df_stat):,} janelas | NoFilter: {len(df_nof):,} janelas")

print(f"\n[2/4] Amostrando {N_SAMPLE} janelas estratificadas...")
X_stat_s, y_stat_s = stratified_sample(X_stat_full, y_stat, N_SAMPLE)
X_nof_s,  y_nof_s  = stratified_sample(X_nof_full,  y_nof,  N_SAMPLE)

print("\n[3/4] Calculando SHAP values (TreeExplainer)...")
print("  Filtro Estatístico...")
shap_stat = compute_shap(model_stat, X_stat_s)
print("  Sem Filtro...")
shap_nof  = compute_shap(model_nof,  X_nof_s)
print(f"  shap_stat: {shap_stat.values.shape} | shap_nof: {shap_nof.values.shape}")

print(f"\n[4/4] Gerando {len(CLASSES_PRESENT)} gráficos de comparação...")
for cls in CLASSES_PRESENT:
    save_path = OUT_DIR / f"shap_classe{cls}.png"
    plot_comparison(shap_stat, shap_nof, X_stat_s, X_nof_s,
                    feat_stat, feat_nof, cls, save_path)
    print(f"  Classe {cls:>3} ({WINDOW_CLASSES.get(cls,'?'):<25}) -> {save_path.name}")

# Delta F1 summary
import json
with open(r"D:\Documentos\UnB\Projeto Final de Curso\TCC\results\metrics\xgboost_statistical_metrics.json") as f:
    m_stat = json.load(f)
with open(r"D:\Documentos\UnB\Projeto Final de Curso\TCC\results\metrics\xgboost_nofilter_metrics.json") as f:
    m_nof = json.load(f)

f1_stat = {int(k): v["f1"] for k, v in m_stat["per_class"].items()}
f1_nof  = {int(k): v["f1"] for k, v in m_nof["per_class"].items()}
plot_summary_delta(f1_stat, f1_nof)

print("\n" + "=" * 60)
print(f"  Figuras salvas em: {OUT_DIR}")
print("=" * 60)
