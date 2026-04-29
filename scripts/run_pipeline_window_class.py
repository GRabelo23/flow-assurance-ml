"""
Gera features_window_class.parquet com rotulagem por estado operacional.

Le do cleaned.parquet existente (sem re-processar dados brutos) e extrai
features com label_strategy='window': cada janela de 300 s recebe como
label a MODA da coluna 'class' dentro daquela janela.

Labels resultantes:
  0        -> operacao normal
  1-9      -> evento ativo do tipo correspondente
  101-109  -> transiente do tipo correspondente
  (janelas 100% NaN em 'class' sao descartadas)

Saida: data/processed/features_window_class.parquet

Execucao:
    python scripts/run_pipeline_window_class.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["PYTHONIOENCODING"] = "utf-8"

from config import CLEANED_DATA_PATH, FEATURES_WINDOW_PATH, VALIDATION_MODE

# Remover arquivo anterior se existir
if FEATURES_WINDOW_PATH.exists():
    FEATURES_WINDOW_PATH.unlink()
    print(f"Arquivo anterior removido: {FEATURES_WINDOW_PATH.name}")

mode = "VALIDACAO" if VALIDATION_MODE else "COMPLETO"
print(f"Modo: {mode}")
print("Iniciando extracao de features com label_strategy='window'...\n")

from src.feature_engineering import run_pipeline_from_cleaned

run_pipeline_from_cleaned(
    cleaned_path=CLEANED_DATA_PATH,
    features_path=FEATURES_WINDOW_PATH,
    label_strategy="window",
    flush_every=50,
    verbose=True,
)

# Inspecionar resultado
import pandas as pd
df = pd.read_parquet(FEATURES_WINDOW_PATH)
print(f"\nShape: {df.shape}")
print(f"Labels unicos (window_label):")
vc = df["window_label"].value_counts().sort_index()
for label, count in vc.items():
    if label == 0:
        desc = "Normal"
    elif label < 100:
        desc = f"Ativo tipo {label}"
    else:
        desc = f"Transiente tipo {label - 100}"
    print(f"  {label:3d} ({desc}): {count:>8,} janelas")

print(f"\nJanelas descartadas (class=NaN 100%): nao aparecem no arquivo")
print("Pronto!")
