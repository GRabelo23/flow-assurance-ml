"""
Regenera os gráficos de importância nativa do XGBoost (gain, weight, cover)
que ficaram em branco por falta de nomes de features no booster.
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    FAULT_CLASSES,
    FEATURES_NOFILTER_PATH,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
)

META_COLS = ["instance_id", "fault_class", "window_label", "source_type", "window_start"]

df = pd.read_parquet(FEATURES_NOFILTER_PATH)
df = df[df["window_label"] == 0]
feature_cols = [c for c in df.columns if c not in META_COLS]

best_xgb = joblib.load(MODELS_DIR / "xgboost_fault_prediction.joblib")
booster = best_xgb.get_booster()
booster.feature_names = feature_cols  # corrige os nomes f0/f1/... para os reais

for imp_type in ["weight", "gain", "cover"]:
    scores = booster.get_score(importance_type=imp_type)
    full_scores = {f: scores.get(f, 0.0) for f in feature_cols}
    xgb_imp = pd.Series(full_scores).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    xgb_imp.head(20).plot(kind="barh", ax=ax)
    ax.set_title(f"XGBoost Importance — {imp_type} (top 20 features)")
    ax.set_xlabel(f"Score ({imp_type})")
    ax.invert_yaxis()
    plt.tight_layout()
    out = FIGURES_DIR / "fault_prediction" / "xgb" / f"xgb_{imp_type}_fault_prediction.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Salvo: {out}")

    if imp_type == "gain":
        with open(METRICS_DIR / "xgb_gain_fault_prediction.json", "w") as f:
            json.dump(xgb_imp.to_dict(), f, indent=2)
        print(f"  JSON de gain salvo")

print("Concluido.")
