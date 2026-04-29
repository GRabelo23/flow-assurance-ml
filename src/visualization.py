"""
Gráficos padronizados para o TCC — Garantia de Escoamento.

Centraliza o estilo visual para que todos os notebooks
gerem figuras com aparência consistente.
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FAULT_CLASSES, FIGURES_DIR, KEY_SENSORS

# Paleta de 10 cores para as classes
CLASS_PALETTE = sns.color_palette("tab10", n_colors=10)
CLASS_COLORS = {cls: CLASS_PALETTE[i] for i, cls in enumerate(FAULT_CLASSES)}

# Cores de fundo por estado operacional (cor, alpha)
# Convenção do 3W: class=0 → Normal, class=1-9 → Falha, class=101-109 → Transiente, NaN → Normal
STATE_STYLES = {
    "normal":    {"color": "#a5d6a7", "alpha": 0.35, "label": "Normal"},
    "transient": {"color": "#fff59d", "alpha": 0.50, "label": "Transiente"},
    "fault":     {"color": "#ef9a9a", "alpha": 0.50, "label": "Falha"},
}

# Estilo global
plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
})


def _classify_state(class_val) -> str:
    """Converte o valor da coluna 'class' do 3W para 'normal', 'transient' ou 'fault'.

    Convenção do 3W Dataset v2.0:
    - 0 ou NaN : operação normal (período pré-evento ou sem ocorrência)
    - 1–9      : falha ativa (mesmo número da pasta/classe)
    - 101–109  : transiente (período de transição antes da falha, = classe + 100)
    """
    if pd.isna(class_val):
        return "normal"
    val = int(class_val)
    if val == 0:
        return "normal"
    if val > 100:
        return "transient"
    return "fault"


def _draw_state_background(axes, class_series: pd.Series) -> None:
    """Pinta o fundo de todos os eixos com a cor do estado operacional de cada amostra.

    Agrupa períodos contíguos do mesmo estado para minimizar o número de axvspan,
    o que é muito mais rápido do que chamar axvspan para cada amostra individualmente.
    """
    states = class_series.map(_classify_state).reset_index(drop=True)
    n = len(states)

    # Encontrar blocos contíguos do mesmo estado
    blocks = []
    start = 0
    current = states.iloc[0]
    for i in range(1, n):
        if states.iloc[i] != current:
            blocks.append((start, i - 1, current))
            start = i
            current = states.iloc[i]
    blocks.append((start, n - 1, current))

    for ax in axes:
        for x_start, x_end, state in blocks:
            style = STATE_STYLES[state]
            ax.axvspan(x_start, x_end,
                       facecolor=style["color"],
                       alpha=style["alpha"],
                       zorder=0, linewidth=0)


def _state_legend_handles() -> list:
    """Retorna os patches para a legenda de estados."""
    return [
        mpatches.Patch(facecolor=s["color"], alpha=s["alpha"] + 0.2, label=s["label"])
        for s in STATE_STYLES.values()
    ]


def _build_title(df_instance: pd.DataFrame, extra: str = "") -> str:
    """Constrói o título padrão com classe de falha e instância."""
    fault_class = df_instance["fault_class"].iloc[0] if "fault_class" in df_instance.columns else "?"
    fault_label = df_instance["fault_label"].iloc[0] if "fault_label" in df_instance.columns else FAULT_CLASSES.get(fault_class, "")
    instance_id = df_instance["instance_id"].iloc[0] if "instance_id" in df_instance.columns else "?"
    title = f"Classe {fault_class}: {fault_label}  |  Instância: {instance_id}"
    if extra:
        title = f"{extra}\n{title}"
    return title


def plot_time_series(df_instance: pd.DataFrame,
                     sensors: list[str] | None = None,
                     title: str = "",
                     save_name: str | None = None,
                     state_col: str = "class") -> plt.Figure:
    """Plota as séries temporais de múltiplos sensores de uma instância.

    O fundo de cada gráfico é colorido automaticamente pelo estado operacional:
    - Verde  : operação normal
    - Amarelo: período transiente (antes da falha)
    - Vermelho: falha ativa

    O título é gerado automaticamente com a classe de falha e a instância.

    Parâmetros
    ----------
    df_instance : pd.DataFrame
        Uma instância com colunas de sensores, metadados (instance_id, fault_class)
        e, opcionalmente, a coluna de estado (padrão: 'class').
    sensors : list[str] | None
        Sensores a plotar. Se None, usa KEY_SENSORS disponíveis.
    title : str
        Prefixo opcional para o título (o título padrão é sempre adicionado).
    save_name : str | None
        Se fornecido, salva a figura em results/figures/.
    state_col : str
        Nome da coluna de estado operacional no DataFrame (padrão: 'class').
    """
    if sensors is None:
        sensors = [s for s in KEY_SENSORS if s in df_instance.columns]

    n = len(sensors)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    # Fundo colorido por estado (se a coluna existir)
    has_state = state_col in df_instance.columns
    if has_state:
        _draw_state_background(axes, df_instance[state_col])

    for ax, sensor in zip(axes, sensors):
        ax.plot(df_instance[sensor].values, linewidth=0.8, color="steelblue", zorder=2)
        ax.set_ylabel(sensor, fontsize=9)
        ax.grid(True, alpha=0.3, zorder=1)

    # Legenda de estados no primeiro subplot
    if has_state:
        axes[0].legend(
            handles=_state_legend_handles(),
            loc="upper right",
            framealpha=0.85,
            fontsize=8,
        )

    axes[-1].set_xlabel("Tempo (amostras)")
    fig.suptitle(_build_title(df_instance, extra=title), y=1.01, fontsize=12)
    plt.tight_layout()

    if save_name:
        _save_figure(fig, save_name)
    return fig


def plot_class_distribution(df: pd.DataFrame,
                             save_name: str = "distribuicao_classes") -> plt.Figure:
    """Plota a distribuição de instâncias por classe e por tipo de fonte."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    counts = df.groupby("fault_class")["instance_id"].nunique()
    labels = [f"{k}\n{FAULT_CLASSES[k]}" for k in counts.index]
    colors = [CLASS_COLORS[k] for k in counts.index]
    axes[0].bar(labels, counts.values, color=colors)
    axes[0].set_title("Instâncias por Classe")
    axes[0].set_ylabel("Nº de instâncias")
    axes[0].tick_params(axis="x", rotation=30)

    pivot = df.groupby(["fault_class", "source_type"])["instance_id"].nunique().unstack(fill_value=0)
    pivot.plot(kind="bar", ax=axes[1], colormap="Set2")
    axes[1].set_title("Instâncias por Classe e Tipo de Fonte")
    axes[1].set_ylabel("Nº de instâncias")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].legend(title="Fonte")

    plt.tight_layout()
    _save_figure(fig, save_name)
    return fig


def plot_missing_data(df: pd.DataFrame,
                      sensors: list[str] | None = None,
                      save_name: str = "dados_faltantes") -> plt.Figure:
    """Heatmap da proporção de dados faltantes por sensor e por classe."""
    if sensors is None:
        sensors = [s for s in KEY_SENSORS if s in df.columns]

    missing = (
        df.groupby("fault_class")[sensors]
        .apply(lambda g: g.isna().mean())
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(missing.T, annot=True, fmt=".1%", cmap="YlOrRd",
                linewidths=0.5, ax=ax, vmin=0, vmax=1)
    ax.set_title("Proporção de Dados Faltantes por Sensor e Classe")
    ax.set_xlabel("Classe de Falha")
    ax.set_ylabel("Sensor")
    ax.set_xticklabels([FAULT_CLASSES[int(t.get_text())] for t in ax.get_xticklabels()],
                       rotation=30, ha="right")
    plt.tight_layout()
    _save_figure(fig, save_name)
    return fig


def plot_correlation_heatmap(df: pd.DataFrame,
                             sensors: list[str] | None = None,
                             save_name: str = "correlacao_sensores") -> plt.Figure:
    """Heatmap de correlação de Pearson entre sensores."""
    if sensors is None:
        sensors = [s for s in KEY_SENSORS if s in df.columns]

    corr = df[sensors].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Correlação entre Sensores")
    plt.tight_layout()
    _save_figure(fig, save_name)
    return fig


def _save_figure(fig: plt.Figure, name: str) -> None:
    """Salva figura em results/figures/ com nome padronizado."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Figura salva: {path}")
