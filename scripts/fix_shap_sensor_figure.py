"""
Regenera shap_por_sensor_estado_operacional.png sem '(Abordagem 2)' no título.
Usa os mesmos parâmetros do notebook 06_interpretacao.ipynb (seed=42, N=50/classe).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

from config import FEATURES_WINDOW_PATH, FIGURES_DIR, MODELS_DIR, WINDOW_CLASSES

META_COLS = ['instance_id', 'fault_class', 'window_label', 'source_type', 'window_start']
DEST_LATEX = Path(__file__).parent.parent / 'docs' / 'latex' / 'unbtex-example' / 'figuras'

print('Carregando dados e modelos...')
df2 = pd.read_parquet(FEATURES_WINDOW_PATH)
feature_cols = [c for c in df2.columns if c not in META_COLS]

imp  = joblib.load(MODELS_DIR / 'imputer_window_class.joblib')
rf   = joblib.load(MODELS_DIR / 'rf_window_class.joblib')
xgb  = joblib.load(MODELS_DIR / 'xgboost_window_class.joblib')

X = imp.transform(df2[feature_cols].values)
y = df2['window_label'].values

# Amostra balanceada — idêntica ao notebook (seed=42, N=50/classe)
rng = np.random.default_rng(42)
N_SHAP_PER_CLASS = 50
n_per_class = min(int(np.sum(y == c)) for c in np.unique(y))
n_per_class = min(n_per_class, N_SHAP_PER_CLASS)
print(f'N por classe: {n_per_class} | Total: {n_per_class * len(np.unique(y)):,}')

idx_balanced = np.concatenate([
    rng.choice(np.where(y == c)[0], n_per_class, replace=False)
    for c in np.unique(y)
])
rng.shuffle(idx_balanced)
X_sample = pd.DataFrame(X[idx_balanced], columns=feature_cols)

def sensor_importance(shap_explanation, feature_names):
    sensors = ['P-PDG', 'T-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP', 'P-JUS-CKGL', 'QGL']
    mean_abs        = np.abs(shap_explanation.values).mean(axis=0)
    mean_abs_global = mean_abs.mean(axis=-1)
    feat_imp = pd.Series(mean_abs_global, index=feature_names)
    sensor_imp = {}
    for s in sensors:
        cols = [c for c in feature_names if c.startswith(s + '_')]
        sensor_imp[s] = feat_imp[cols].sum()
    return pd.Series(sensor_imp).sort_values(ascending=True)

print('Calculando SHAP — RF (~25 min)...')
explainer_rf = shap.TreeExplainer(rf)
shap_rf      = explainer_rf(X_sample)
print(f'SHAP RF calculado. Shape: {shap_rf.shape}')

print('Calculando SHAP — XGBoost (~2-3 min)...')
explainer_xgb = shap.TreeExplainer(xgb)
shap_xgb      = explainer_xgb(X_sample)
print(f'SHAP XGBoost calculado. Shape: {shap_xgb.shape}')

sensor_rf  = sensor_importance(shap_rf,  feature_cols)
sensor_xgb = sensor_importance(shap_xgb, feature_cols)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sensor_rf.plot(kind='barh',  ax=axes[0], color='steelblue')
axes[0].set_title('RF — Importância por Sensor', fontsize=11)
axes[0].set_xlabel('Soma do mean|SHAP|')

sensor_xgb.plot(kind='barh', ax=axes[1], color='darkorange')
axes[1].set_title('XGBoost — Importância por Sensor', fontsize=11)
axes[1].set_xlabel('Soma do mean|SHAP|')

plt.suptitle('Importância SHAP por Sensor — RF e XGBoost', fontsize=12, y=1.01)
plt.tight_layout()

out = FIGURES_DIR / 'shap_por_sensor_estado_operacional.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f'Salvo: {out}')

dest = DEST_LATEX / 'shap_por_sensor_estado_operacional.png'
import shutil
shutil.copy(out, dest)
print(f'Copiado para LaTeX: {dest}')
