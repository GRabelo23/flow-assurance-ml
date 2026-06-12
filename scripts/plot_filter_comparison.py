"""
Compara sinal bruto (z-score) vs filtro estatístico adaptativo.
Gera grade 3x2 com janela de +/-3000 s em torno do início do evento ativo.

Uso:
    python scripts/plot_filter_comparison.py
    python scripts/plot_filter_comparison.py --class 9 --instance SIMULATED_00080
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FAULT_CLASSES, FFILL_LIMIT, FIGURES_DIR
from src.data_loader import load_class
from src.feature_engineering import (
    _apply_statistical_filter,
    _normalize_instance_sensors,
)

META_COLS     = {"instance_id", "fault_class", "fault_label", "source_type", "class", "state"}
SENSORS       = ["P-JUS-CKGL", "P-MON-CKP", "P-TPT", "QGL", "T-MON-CKP", "P-ANULAR"]
WINDOW_BEFORE = 3000
WINDOW_AFTER  = 3000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--class",    type=int, default=8, dest="fault_class")
    parser.add_argument("--instance", type=str, default="WELL-00028_20210617053128")
    parser.add_argument("--sensors",  nargs="+", default=SENSORS)
    parser.add_argument("--before",   type=int, default=WINDOW_BEFORE)
    parser.add_argument("--after",    type=int, default=WINDOW_AFTER)
    args = parser.parse_args()

    fault_class = args.fault_class
    sensors     = args.sensors

    print(f"Carregando classe {fault_class}...")
    df_cls = load_class(fault_class)
    df_raw = df_cls[df_cls["instance_id"] == args.instance].copy()
    if df_raw.empty:
        raise RuntimeError(f"Instancia '{args.instance}' nao encontrada.")

    source = df_raw["source_type"].iloc[0]
    print(f"Instancia: {args.instance}  fonte={source}  {len(df_raw):,} s")

    # Forward-fill causal
    sensor_cols = [c for c in df_raw.columns if c not in META_COLS]
    df_raw[sensor_cols] = df_raw[sensor_cols].ffill(limit=FFILL_LIMIT)

    # Localizar início do evento ativo
    active_mask = df_raw["class"].fillna(-1).astype(int) == fault_class
    if not active_mask.any():
        raise RuntimeError("Evento ativo nao encontrado nessa instancia.")
    active_start = int(np.where(active_mask.values)[0][0])

    # Recortar janela
    win_start = max(0, active_start - args.before)
    win_end   = min(len(df_raw), active_start + args.after)
    df_win    = df_raw.iloc[win_start:win_end].copy()
    t = np.arange(len(df_win))

    print(f"Janela: {win_start} — {win_end}  ({len(df_win):,} s)")

    # Z-score sobre a instância completa (não só a janela)
    df_norm_full = _normalize_instance_sensors(df_raw.copy(), sensors)
    df_norm      = df_norm_full.iloc[win_start:win_end]

    sensors_avail = [s for s in sensors if s in df_norm.columns
                     and df_norm[s].notna().any()]

    # ── Layout 3×2 ────────────────────────────────────────────────────────────
    ncols = 2
    nrows = int(np.ceil(len(sensors_avail) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(13, 3.5 * nrows),
                             sharex=True)
    axes_flat = axes.flatten()

    for idx, sensor in enumerate(sensors_avail):
        ax = axes_flat[idx]
        raw      = df_norm[sensor].values.astype(float)
        filtered = _apply_statistical_filter(raw)

        ax.plot(t, raw,      color="#1f77b4", linewidth=0.6, alpha=0.85,
                label="Sinal Original")
        ax.plot(t, filtered, color="#d62728",  linewidth=1.4, alpha=0.95,
                label="Filtro Estatístico")

        ax.set_title(sensor, fontsize=10)
        ax.set_ylabel("z-score", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(linestyle=":", alpha=0.3)

    # Ocultar subplots excedentes
    for idx in range(len(sensors_avail), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Rótulo do eixo x só na última linha
    for ax in axes_flat[(nrows - 1) * ncols:]:
        ax.set_xlabel("Tempo relativo (s)", fontsize=9)

    # Legenda global
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        "Comparação entre sinal original e sinal filtrado pelo\n"
        "filtro estatístico adaptativo para instância do 3W Dataset",
        fontsize=11, y=1.02,
    )

    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    save_path = FIGURES_DIR / f"filter_comparison_class{fault_class}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Salvo: {save_path}")


if __name__ == "__main__":
    main()
