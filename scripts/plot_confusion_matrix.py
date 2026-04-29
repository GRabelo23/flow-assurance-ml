"""
Plota a matriz de confusão do Random Forest usando predições out-of-fold.

Por que cross_val_predict e não model.predict(X)?
  O modelo foi treinado em TODOS os dados (refit=True no RandomizedSearchCV).
  Usar model.predict produziria a matriz de treino — inflada e desonesta.
  cross_val_predict garante que cada janela é avaliada por um modelo que
  nunca a viu, replicando o mesmo protocolo GroupKFold usado no treinamento.

Execução:
    python scripts/plot_confusion_matrix.py
"""

import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["PYTHONIOENCODING"] = "utf-8"

from config import (
    FEATURES_DATA_PATH,
    FIGURES_DIR,
    MODELS_DIR,
    N_JOBS,
    N_SPLITS_CV,
)
from src.evaluation import plot_confusion_matrix, print_classification_report


def main():
    print("=" * 60)
    print("  Matriz de Confusao — Random Forest")
    print("=" * 60)

    # ── Carregar features ─────────────────────────────────────────────────────
    print("\n[1/4] Carregando features...")
    df = pd.read_parquet(FEATURES_DATA_PATH)
    META_COLS = ["instance_id", "fault_class", "source_type", "window_start"]
    feature_cols = [c for c in df.columns if c not in META_COLS]

    X = df[feature_cols].values
    y = df["fault_class"].values
    groups = df["instance_id"].values

    print(f"  Janelas: {len(df):,} | Features: {len(feature_cols)}")

    # ── Imputar NaN ───────────────────────────────────────────────────────────
    print("\n[2/4] Imputando NaN...")
    imputer = joblib.load(MODELS_DIR / "imputer.joblib")
    X = imputer.transform(X)

    # ── Carregar modelo e gerar predições out-of-fold ─────────────────────────
    print(f"\n[3/4] Gerando predicoes out-of-fold (GroupKFold, {N_SPLITS_CV} folds)...")
    print("  (cada janela e avaliada por um modelo que nunca a viu)")
    model = joblib.load(MODELS_DIR / "random_forest.joblib")
    gkf = GroupKFold(n_splits=N_SPLITS_CV)

    y_pred = cross_val_predict(
        model, X, y,
        cv=gkf,
        groups=groups,
        n_jobs=N_JOBS,
        verbose=1,
    )

    # ── Plotar e salvar ───────────────────────────────────────────────────────
    print("\n[4/4] Plotando matriz de confusao...")
    print_classification_report(y, y_pred, model_name="Random Forest")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig = plot_confusion_matrix(y, y_pred, model_name="Random Forest", save=True)
    print("\nConcluido!")


if __name__ == "__main__":
    main()
