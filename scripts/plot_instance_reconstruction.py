"""
Reconstrói o rótulo de estado operacional ao longo de uma instância completa,
comparando rótulos verdadeiros (coluna 'class') com as predições do XGBoost.

Para cada classe de falha (1–9) seleciona a instância mais ilustrativa:
aquela que contém Normal + Transiente + Ativo com o transiente mais longo.
Fallback: instância com maior diversidade de estados.

Uso:
    python scripts/plot_instance_reconstruction.py
    python scripts/plot_instance_reconstruction.py --filter statistical
    python scripts/plot_instance_reconstruction.py --classes 2 7 9
"""

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    FAULT_CLASSES,
    FFILL_LIMIT,
    FIGURES_DIR,
    KEY_SENSORS,
    METRICS_DIR,
    MODELS_DIR,
    RAW_DATA_DIR,
    STEP_SIZE,
    WINDOW_CLASSES,
    WINDOW_SIZE,
)
from src.data_loader import load_class
from src.feature_engineering import extract_features_from_instance


# ── Cores e rótulos de estado ─────────────────────────────────────────────────
_COLOR_NORMAL     = "#2ca02c"   # verde
_COLOR_TRANSIENT  = "#ff7f0e"   # laranja
_COLOR_ACTIVE     = "#d62728"   # vermelho
_COLOR_UNKNOWN    = "#cccccc"   # cinza

def _state_color(state: int) -> str:
    if state == 0:
        return _COLOR_NORMAL
    if 101 <= state <= 109:
        return _COLOR_TRANSIENT
    if 1 <= state <= 9:
        return _COLOR_ACTIVE
    return _COLOR_UNKNOWN


def _coerce_state(val) -> int:
    """Converte NaN → 0 (Normal) e garante int."""
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return 0
        return int(val)
    except (TypeError, ValueError):
        return 0


# ── Seleção da instância mais ilustrativa ────────────────────────────────────
def select_best_instance(fault_class: int,
                          preferred_source: str = "WELL") -> tuple[str, pd.DataFrame]:
    """
    Escolhe a instância mais ilustrativa de uma classe de falha.
    Critérios (por ordem de prioridade):
      1. Tem os três estados: Normal + Transiente + Ativo
      2. Entre essas, prefere a fonte indicada por preferred_source
      3. Maximiza a duração do período transiente
    Fallback: instância com maior número de estados distintos (da fonte preferida,
              se disponível, senão qualquer fonte).
    """
    df_class = load_class(fault_class)
    transient_code = 100 + fault_class

    best_id     = None
    best_score  = None
    fallback_id = None
    fallback_n  = -1

    for inst_id, df_inst in df_class.groupby("instance_id"):
        if "class" not in df_inst.columns:
            continue

        source = df_inst["source_type"].iloc[0]
        valid  = df_inst["class"].dropna()
        unique = set(valid.unique())
        n_distinct = len(unique) + int(df_inst["class"].isna().any())

        # Fallback: maior diversidade de estados, com preferência pela fonte
        fallback_prio = (0 if source == preferred_source else 1, -n_distinct)
        if fallback_id is None or fallback_prio < (0 if df_class[df_class["instance_id"] == fallback_id]["source_type"].iloc[0] == preferred_source else 1, -fallback_n):
            fallback_n  = n_distinct
            fallback_id = inst_id

        has_transient = transient_code in unique
        has_active    = fault_class in unique

        if has_transient and has_active:
            trans_len   = int((valid == transient_code).sum())
            source_prio = 0 if source == preferred_source else 1
            score       = (source_prio, -trans_len)
            if best_score is None or score < best_score:
                best_score = score
                best_id    = inst_id

    chosen_id = best_id if best_id is not None else fallback_id
    df_chosen = df_class[df_class["instance_id"] == chosen_id].copy()
    return chosen_id, df_chosen


# ── Mapeamento de predições para o eixo de tempo ──────────────────────────────
def predictions_to_timeseries(
    df_windows: pd.DataFrame,
    y_pred: np.ndarray,
    n_timesteps: int,
) -> np.ndarray:
    """
    Para cada janela, preenche os timesteps cobertos com o rótulo predito.
    Regiões com sobreposição (50%): a janela mais recente prevalece.
    Regiões sem cobertura (início/fim): rótulo -1.
    """
    series = np.full(n_timesteps, -1, dtype=int)
    for i, start in enumerate(df_windows["window_start"].values):
        end = int(start) + WINDOW_SIZE
        series[int(start):end] = y_pred[i]
    return series


# ── Utilitário de segmentação contígua ───────────────────────────────────────
def _iter_segments(states: np.ndarray):
    """Gera (start, end, state) para cada segmento contíguo de rótulo."""
    if len(states) == 0:
        return
    seg_start = 0
    cur = _coerce_state(states[0])
    for i in range(1, len(states)):
        s = _coerce_state(states[i])
        if s != cur:
            yield seg_start, i, cur
            seg_start = i
            cur = s
    yield seg_start, len(states), cur


# ── Plot de uma instância ─────────────────────────────────────────────────────
def plot_instance(
    fault_class: int,
    instance_id: str,
    df_ffill: pd.DataFrame,
    y_pred_series: np.ndarray,
    sensors_to_plot: list[str],
    save_dir: Path,
) -> None:
    true_states = df_ffill["class"].values
    t = np.arange(len(df_ffill))
    n_s = len(sensors_to_plot)

    fig, axes = plt.subplots(
        n_s + 1, 1,
        figsize=(16, 2.5 * n_s + 2.8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2] * n_s + [1.4]},
    )

    source = df_ffill["source_type"].iloc[0]
    fig.suptitle(
        f"Classe {fault_class} — {FAULT_CLASSES[fault_class]}   "
        f"({source})   Instância: {instance_id}\n"
        f"Duração: {len(df_ffill):,} s  |  "
        f"Janelas: {int((len(df_ffill) - WINDOW_SIZE) / STEP_SIZE) + 1}  |  "
        f"Filtro: Gaussiano",
        fontsize=10, y=1.01,
    )

    # ── Painéis de sensores ───────────────────────────────────────────────────
    for ax, sensor in zip(axes[:-1], sensors_to_plot):
        if sensor not in df_ffill.columns:
            ax.set_ylabel(sensor, fontsize=8)
            ax.text(0.5, 0.5, "Sensor indisponível", transform=ax.transAxes,
                    ha="center", color="gray")
            continue

        vals = df_ffill[sensor].values.astype(float)
        ax.plot(t, vals, color="#1f77b4", linewidth=0.7, alpha=0.9, zorder=2)
        ax.set_ylabel(sensor, fontsize=8)
        ax.tick_params(labelsize=7)

        # Sombreamento por estado verdadeiro (fundo)
        for seg_s, seg_e, state in _iter_segments(true_states):
            ax.axvspan(seg_s, seg_e, alpha=0.15,
                       color=_state_color(state), linewidth=0, zorder=1)

    # ── Painel de comparação de rótulos ───────────────────────────────────────
    ax_lbl = axes[-1]
    _draw_label_bands(ax_lbl, true_states, y_pred_series)
    ax_lbl.set_xlabel("Tempo (s)", fontsize=9)
    ax_lbl.set_xlim(0, len(df_ffill))
    ax_lbl.tick_params(labelsize=7)

    # Legenda global
    legend_items = [
        mpatches.Patch(color=_COLOR_NORMAL,    label="Normal"),
        mpatches.Patch(color=_COLOR_TRANSIENT, label="Transiente"),
        mpatches.Patch(color=_COLOR_ACTIVE,    label="Ativo"),
        mpatches.Patch(color=_COLOR_UNKNOWN,   label="Sem cobertura"),
    ]
    fig.legend(handles=legend_items, loc="upper right",
               fontsize=8, ncol=4, bbox_to_anchor=(1.0, 1.03))

    plt.tight_layout()
    fname = f"reconstruction_class{fault_class}.png"
    plt.savefig(save_dir / fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Salvo → {fname}")


def _draw_label_bands(ax, true_states: np.ndarray, pred_series: np.ndarray):
    """Painel inferior: duas faixas coloridas (verdadeiro em cima, predito embaixo)."""
    y_top = 0.52      # faixa verdadeiro: [0.52, 0.97]
    y_bot = 0.02      # faixa predito:    [0.02, 0.47]
    height = 0.45

    for y_pos, states, txt in [(y_top, true_states, "Verdadeiro"),
                               (y_bot, pred_series, "XGBoost")]:
        for seg_s, seg_e, state in _iter_segments(states):
            color = _COLOR_UNKNOWN if state == -1 else _state_color(state)
            ax.barh(y_pos, seg_e - seg_s, left=seg_s,
                    height=height, color=color, align="edge", linewidth=0)
        ax.text(-len(true_states) * 0.005, y_pos + height / 2, txt,
                va="center", ha="right", fontsize=8, fontweight="bold")

    # Linha divisória
    ax.axhline(0.5, color="white", linewidth=1.2, zorder=5)
    ax.set_ylim(0, 1)
    ax.set_yticks([])


# ── Pipeline principal ────────────────────────────────────────────────────────
_META_COLS = {"instance_id", "fault_class", "fault_label",
              "source_type", "class", "state", "window_label", "window_start"}


def main():
    parser = argparse.ArgumentParser(
        description="Reconstrói rótulos preditos ao longo de uma instância completa."
    )
    parser.add_argument("--filter", choices=["gaussian", "statistical", "none"],
                        default="gaussian", dest="filter_type")
    parser.add_argument("--classes", nargs="+", type=int, default=list(range(1, 10)),
                        help="Classes a plotar (padrão: 1–9)")
    parser.add_argument("--source", choices=["WELL", "SIMULATED", "DRAWN"],
                        default="WELL", help="Fonte preferida (padrão: WELL)")
    args = parser.parse_args()
    filter_type     = args.filter_type
    preferred_source = args.source
    suffix = f"_{filter_type}" if filter_type != "gaussian" else ""

    # ── Carregar modelo ───────────────────────────────────────────────────────
    model_path   = MODELS_DIR / f"xgboost_window_class{suffix}.joblib"
    imputer_path = MODELS_DIR / f"imputer_xgb_window_class{suffix}.joblib"
    encoder_path = MODELS_DIR / f"label_encoder_window_class{suffix}.joblib"

    for p in (model_path, imputer_path, encoder_path):
        if not p.exists():
            print(f"Arquivo não encontrado: {p}")
            print("Execute train_xgboost_window_class.py antes deste script.")
            sys.exit(1)

    print("Carregando modelo XGBoost e pré-processadores...")
    model   = joblib.load(model_path)
    imputer = joblib.load(imputer_path)
    encoder = joblib.load(encoder_path)

    save_dir = FIGURES_DIR / "reconstruction" / preferred_source.lower()
    save_dir.mkdir(parents=True, exist_ok=True)

    PLOT_SENSORS = [s for s in ["P-TPT", "T-TPT", "P-MON-CKP", "T-JUS-CKP"]
                    if s in KEY_SENSORS]

    for fault_class in args.classes:
        label = FAULT_CLASSES[fault_class]
        print(f"\n{'-'*55}")
        print(f"[Classe {fault_class}] {label}")

        # 1. Selecionar instância
        instance_id, df_raw = select_best_instance(fault_class, preferred_source)
        source = df_raw["source_type"].iloc[0]
        print(f"  Instância: {instance_id}  fonte={source}  {len(df_raw):,} s")

        # 2. Forward-fill causal (mesmo limite usado no treino)
        df_ffill = df_raw.copy()
        sensor_cols = [c for c in df_ffill.columns if c not in _META_COLS]
        df_ffill[sensor_cols] = df_ffill[sensor_cols].ffill(limit=FFILL_LIMIT)

        # 3. Extrair features de TODAS as janelas (label_strategy="instance"
        #    garante que janelas 100% NaN em 'class' não sejam descartadas)
        df_feat = extract_features_from_instance(
            df_ffill,
            label_strategy="instance",
            smooth_filter=filter_type,
        )

        if df_feat is None or len(df_feat) == 0:
            print("  Sem janelas extraídas — instância muito curta. Pulando.")
            continue

        n_windows = len(df_feat)
        print(f"  Janelas extraídas: {n_windows}")

        # 4. Predizer
        feature_cols = [c for c in df_feat.columns if c not in _META_COLS]
        X = imputer.transform(df_feat[feature_cols])
        y_encoded = model.predict(X)
        y_pred    = encoder.inverse_transform(y_encoded).astype(int)

        # 5. Reconstruir série de rótulos no eixo de tempo
        y_pred_series = predictions_to_timeseries(df_feat, y_pred, len(df_ffill))

        # 6. Resumo de acertos
        true_window_labels = df_feat["window_label"].values if "window_label" in df_feat.columns else None
        if true_window_labels is not None:
            from sklearn.metrics import f1_score as _f1
            acc = (y_pred == true_window_labels.astype(int)).mean()
            print(f"  Acurácia nas janelas desta instância: {acc:.1%}")

        # 7. Plotar
        sensors_avail = [s for s in PLOT_SENSORS if s in df_ffill.columns]
        plot_instance(
            fault_class, instance_id, df_ffill,
            y_pred_series, sensors_avail, save_dir,
        )

    print(f"\n{'='*55}")
    print(f"Figuras salvas em: {save_dir}")



if __name__ == "__main__":
    main()
