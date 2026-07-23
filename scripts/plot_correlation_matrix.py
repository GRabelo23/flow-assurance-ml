"""
Gera a matriz de correlação de Pearson entre as variáveis do 3W Dataset.

Tratamento de dados faltantes: correlação por par (pairwise complete
observations) — para cada par de sensores, usa apenas as amostras onde
ambos têm valor. Pares com menos de MIN_PERIODS amostras em comum
ficam como NaN e são exibidos em cinza no heatmap.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FAULT_CLASSES, FIGURES_DIR, KEY_SENSORS, RAW_DATA_DIR, SENSOR_UNITS

# ── Configurações ─────────────────────────────────────────────────────────────

MIN_PERIODS = 500    # mínimo de amostras em comum para calcular a correlação
ROWS_PER_CLASS = 500 # linhas amostradas por classe no pairplot

# Sensores contínuos de interesse (excluem ESTADO-* e class)
SENSORS = [
    "P-PDG", "P-TPT", "P-MON-CKP", "P-ANULAR",
    "P-JUS-CKP", "P-JUS-CKGL", "P-MON-CKGL", "P-MON-SDV-P",
    "T-TPT", "T-PDG", "T-JUS-CKP", "T-MON-CKP",
    "ABER-CKP", "ABER-CKGL",
    "QBS", "QGL",
]

SAVE_PATH = FIGURES_DIR / "correlation_matrix_3w.png"

# ── Carregamento ──────────────────────────────────────────────────────────────

def load_all_instances() -> pd.DataFrame:
    import pyarrow.parquet as pq

    parquets = sorted(RAW_DATA_DIR.rglob("*.parquet"))
    print(f"Carregando {len(parquets)} instâncias...")

    chunks = []
    for i, path in enumerate(parquets, 1):
        # Lê só os metadados para saber quais colunas existem neste arquivo
        schema_cols = set(pq.read_schema(path).names)
        cols = [c for c in SENSORS if c in schema_cols]
        if not cols:
            continue
        df = pd.read_parquet(path, columns=cols)
        chunks.append(df)
        if i % 200 == 0:
            print(f"  {i}/{len(parquets)} instâncias carregadas...")

    print("Concatenando...")
    return pd.concat(chunks, ignore_index=True)


# ── Carregamento para pairplot ────────────────────────────────────────────────

def load_for_pairplot() -> pd.DataFrame:
    """Carrega KEY_SENSORS + fault_class (da pasta) e amostra ROWS_PER_CLASS por classe."""
    import pyarrow.parquet as pq

    parquets = sorted(RAW_DATA_DIR.rglob("*.parquet"))
    print(f"Carregando {len(parquets)} instâncias para pairplot...")

    chunks = []
    for path in parquets:
        try:
            fault_class = int(path.parent.name)
        except ValueError:
            continue

        schema_cols = set(pq.read_schema(path).names)
        cols = [c for c in KEY_SENSORS if c in schema_cols]
        if not cols:
            continue

        df = pd.read_parquet(path, columns=cols)
        df["fault_class"] = fault_class
        chunks.append(df)

    combined = pd.concat(chunks, ignore_index=True)

    avail = [c for c in KEY_SENSORS if c in combined.columns]
    # Remove apenas linhas onde todos os sensores são NaN; NaN parciais ficam
    # (seaborn descarta por par em cada subplot)
    combined = combined.dropna(subset=avail, how="all")

    # Filtro de limites físicos: leituras fora desses limites são dados corrompidos
    for col in avail:
        if col.startswith("T-"):
            combined.loc[combined[col] < -200, col] = np.nan
        elif col.startswith("P-") or col.startswith("Q"):
            combined.loc[combined[col] < 0, col] = np.nan

    # Clipping robusto por IQR: limita a [Q1 - 3·IQR, Q3 + 3·IQR]
    for col in avail:
        q1, q3 = combined[col].quantile(0.25), combined[col].quantile(0.75)
        iqr = q3 - q1
        combined[col] = combined[col].clip(q1 - 3 * iqr, q3 + 3 * iqr)

    sampled = (
        combined.groupby("fault_class", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), ROWS_PER_CLASS), random_state=42))
        .reset_index(drop=True)
    )
    return sampled


# ── Script principal ──────────────────────────────────────────────────────────

def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Carrega e concatena todas as instâncias
    data = load_all_instances()
    print(f"Total de amostras: {len(data):,}")
    print(f"Colunas disponíveis: {list(data.columns)}")

    # Reordena para a lista de SENSORS (apenas os que existem no dataset)
    cols_ordered = [c for c in SENSORS if c in data.columns]
    data = data[cols_ordered]

    # Correlação pairwise — ignora NaN por par, exige MIN_PERIODS em comum
    corr = data.corr(method="pearson", min_periods=MIN_PERIODS)

    # Máscara para o triângulo superior (evita repetição)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    # ── Figura principal: heatmap de correlação ───────────────────────────────
    n = len(corr)
    fig, ax = plt.subplots(figsize=(n * 0.7 + 1.5, n * 0.7 + 1))

    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmin=-1, vmax=1, center=0,
        annot=True, fmt=".2f",
        annot_kws={"size": 7},
        linewidths=0.4, linecolor="white",
        square=True,
        cbar_kws={"shrink": 0.7, "label": "Correlação de Pearson"},
        ax=ax,
    )

    ax.set_title(
        "Matriz de Correlação de Pearson — 3W Dataset 2.0.0\n"
        f"(correlação por par; pares com menos de {MIN_PERIODS:,} amostras em cinza)",
        fontsize=11, pad=12,
    )
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0,  labelsize=9)

    plt.tight_layout()
    fig.savefig(SAVE_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSalvo: {SAVE_PATH}")

    # ── Figura auxiliar: cobertura de pares (% do dataset completo) ─────────
    total_samples = len(data)
    notna = data.notna().astype(int)
    counts = notna.T @ notna  # shape (n_sensors, n_sensors), dtype int
    coverage_pct = (counts / total_samples * 100).where(~mask)

    fig2, ax2 = plt.subplots(figsize=(n * 0.7 + 1.5, n * 0.7 + 1))

    sns.heatmap(
        coverage_pct,
        mask=mask,
        cmap="YlGn",
        vmin=0, vmax=100,
        annot=True, fmt=".1f",
        annot_kws={"size": 7},
        linewidths=0.4, linecolor="white",
        square=True,
        cbar_kws={"shrink": 0.7, "label": "Amostras em comum (% do dataset)"},
        ax=ax2,
    )

    ax2.set_title(
        f"Cobertura por par de sensores — 3W Dataset 2.0.0\n"
        f"(% de amostras com leitura simultânea; total = {total_samples:,})",
        fontsize=11, pad=12,
    )
    ax2.tick_params(axis="x", rotation=45, labelsize=9)
    ax2.tick_params(axis="y", rotation=0,  labelsize=9)

    plt.tight_layout()
    coverage_path = FIGURES_DIR / "correlation_coverage_3w.png"
    fig2.savefig(coverage_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Salvo: {coverage_path}")


SENSOR_LABELS = {
    s: f"{s} ({u})" for s, u in SENSOR_UNITS.items()
}


def plot_pairplot():
    data = load_for_pairplot()
    data["Classe"] = data["fault_class"].map(FAULT_CLASSES)

    plot_cols = [c for c in KEY_SENSORS if c in data.columns]
    print(f"Pairplot: {len(data):,} amostras × {len(plot_cols)} sensores")

    # Renomeia colunas para incluir unidades nos eixos
    rename_map = {c: SENSOR_LABELS[c] for c in plot_cols if c in SENSOR_LABELS}
    plot_data = data[plot_cols + ["Classe"]].rename(columns=rename_map)
    labeled_cols = [rename_map.get(c, c) for c in plot_cols]

    # Ordem fixa das classes para garantir mapeamento correto de cores
    class_order = [FAULT_CLASSES[k] for k in sorted(FAULT_CLASSES)]
    palette = dict(zip(class_order, sns.color_palette("tab10", n_colors=10)))

    sns.set_context("paper", font_scale=1.1)

    g = sns.pairplot(
        plot_data[labeled_cols + ["Classe"]],
        hue="Classe",
        hue_order=class_order,
        palette=palette,
        height=1.8,
        plot_kws={"alpha": 0.4, "s": 7, "linewidth": 0},
        diag_kind="kde",
        corner=True,
    )

    sns.set_context("notebook")

    g.fig.suptitle(
        "Pairplot dos Sensores Principais — 3W Dataset 2.0.0\n"
        f"({ROWS_PER_CLASS} amostras por classe, triângulo inferior)",
        y=1.01, fontsize=12,
    )

    # Ajusta fonte dos ticks e labels dos eixos em todos os subplots
    for ax_row in g.axes:
        for ax in ax_row:
            if ax is None:
                continue
            ax.tick_params(axis="both", labelsize=7)
            xl = ax.get_xlabel()
            if xl:
                ax.set_xlabel(xl, fontsize=8)
            yl = ax.get_ylabel()
            if yl:
                ax.set_ylabel(yl, fontsize=8)

    # O primeiro elemento da diagonal tem o eixo x suprimido — habilita manualmente
    ax_first = g.axes[0][0]
    ax_first.tick_params(axis="x", labelbottom=True, labelsize=7)
    ax_first.set_xlabel(labeled_cols[0], fontsize=8)

    # Move a legenda para o espaço vazio no canto superior direito (triângulo vazio
    # do corner=True), usando 2 colunas para ficar mais compacto
    leg = g.legend
    if leg:
        handles = leg.legend_handles
        labels  = [t.get_text() for t in leg.get_texts()]
        leg.remove()

        new_leg = g.fig.legend(
            handles, labels,
            title="Classe",
            title_fontproperties={"size": 11, "weight": "bold"},
            fontsize=9,
            ncol=2,
            loc="upper right",
            bbox_to_anchor=(0.875, 0.98),
            framealpha=0.95,
            edgecolor="#aaaaaa",
            markerscale=1.8,
        )
        for handle in new_leg.legend_handles:
            handle.set_alpha(1.0)

    latex_figs = Path(__file__).parent.parent / "docs" / "latex" / "unbtex-example" / "figuras"
    pairplot_path = FIGURES_DIR / "pairplot_key_sensors_3w.png"
    g.fig.savefig(pairplot_path, dpi=150, bbox_inches="tight")
    g.fig.savefig(latex_figs / "pairplot_key_sensors_3w.png", dpi=150, bbox_inches="tight")
    plt.close(g.fig)
    print(f"Salvo: {pairplot_path}")


if __name__ == "__main__":
    main()
    plot_pairplot()
