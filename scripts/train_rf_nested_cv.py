"""
Validação por Nested Cross-Validation — Random Forest (Estado Operacional, 17 classes).

CONFIGURAÇÃO PARA COMPARAÇÃO JUSTA COM O FLAT CV:
  Flat CV:   GroupKFold(5), N_ITER=20, 1 loop    → 100 fits
  Nested CV: GroupKFold(5) externo
             GroupKFold(3) interno, N_ITER=20     → 300 fits

O nested CV usa o MESMO grid de hiperparâmetros e o MESMO número de
iterações de busca que o flat CV. A única diferença estrutural é o loop
externo: a seleção de hiperparâmetros nunca enxerga os dados do fold
de teste externo.

Flat CV (viés de seleção):
  Todos os 5 folds participam da busca de hiperparâmetros.
  O mesmo conjunto de poços que ajuda a escolher os hiperparâmetros
  também aparece como "teste" em algum fold.

Nested CV (sem viés):
  Fold externo divide os poços em treino+inner vs teste externo.
  O fold de teste externo nunca é visto até a avaliação final.

Total de fits: 5 × 20 × 3 = 300 treinamentos (~8–12 h)

Execução:
    python -u scripts/train_rf_nested_cv.py 2>&1 | Tee-Object -FilePath training_nested_cv.log

Monitorar em outro terminal:
    Get-Content training_nested_cv.log -Wait

Saída:
    results/metrics/rf_nested_cv_results.json
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["PYTHONIOENCODING"] = "utf-8"

from config import (
    FEATURES_WINDOW_PATH,
    METRICS_DIR,
    N_JOBS,
    RANDOM_STATE,
    WINDOW_CLASSES,
)


def _impute_per_instance(X_raw: np.ndarray, groups: np.ndarray, feature_cols: list) -> np.ndarray:
    """Preenche NaN com a mediana da própria instância para cada feature.

    Como GroupKFold nunca divide janelas de uma mesma instância entre folds,
    isso é equivalente à imputação por fold — sem vazamento de informação.
    Fallback para 0.0 quando a feature é inteiramente NaN na instância
    (valor neutro no espaço z-score normalizado).
    """
    df = pd.DataFrame(X_raw, columns=feature_cols)
    df["_gid"] = groups
    inst_med = df.groupby("_gid")[feature_cols].transform("median")
    df[feature_cols] = df[feature_cols].fillna(inst_med).fillna(0.0)
    return df[feature_cols].values


def main():
    print("=" * 60)
    print("  Nested CV — Random Forest Estado Operacional (17 classes)")
    print("  Comparação justa com flat CV (GroupKFold 5, N_ITER=20)")
    print(f"  Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ── Configuração espelhando o flat CV ─────────────────────────────────────
    N_OUTER_SPLITS = 5   # igual ao flat CV — 5 folds externos
    N_INNER_SPLITS = 3   # folds internos para busca de hiperparâmetros
    N_ITER_INNER   = 20  # igual ao flat CV — mesmo orçamento de busca

    # ── Carregar features ─────────────────────────────────────────────────────
    print("\n[1/3] Carregando features...")
    if not FEATURES_WINDOW_PATH.exists():
        print(f"ERRO: {FEATURES_WINDOW_PATH} nao encontrado.")
        sys.exit(1)

    df = pd.read_parquet(FEATURES_WINDOW_PATH)
    META_COLS = ["instance_id", "fault_class", "window_label", "source_type", "window_start"]
    feature_cols = [c for c in df.columns if c not in META_COLS]

    X      = df[feature_cols].values
    y      = df["window_label"].values
    groups = df["instance_id"].values

    classes = np.unique(y)
    print(f"  Janelas: {len(df):,} | Features: {len(feature_cols)}")
    print(f"  Instancias: {len(np.unique(groups)):,} | Classes: {len(classes)}")
    print(f"  Config: outer={N_OUTER_SPLITS} folds | inner={N_INNER_SPLITS} folds | N_ITER={N_ITER_INNER}")
    print(f"  Total de fits: {N_OUTER_SPLITS} x {N_ITER_INNER} x {N_INNER_SPLITS} = "
          f"{N_OUTER_SPLITS * N_ITER_INNER * N_INNER_SPLITS}")

    # ── Imputar NaN ───────────────────────────────────────────────────────────
    print("\n[2/3] Imputando NaN com mediana por instancia...")
    X = _impute_per_instance(X, groups, feature_cols)

    # ── Grid IDÊNTICO ao flat CV ───────────────────────────────────────────────
    # Qualquer diferença no grid tornaria a comparação injusta.
    param_grid = {
        "n_estimators":     [100, 200, 300],
        "max_depth":        [None, 10, 20, 30],
        "min_samples_leaf": [1, 2, 4],
        "max_features":     ["sqrt", "log2"],
    }

    # ── Nested CV ─────────────────────────────────────────────────────────────
    print("\n[3/3] Executando Nested Cross-Validation...")
    gkf_outer = GroupKFold(n_splits=N_OUTER_SPLITS)
    gkf_inner = GroupKFold(n_splits=N_INNER_SPLITS)

    outer_scores         = []
    best_params_per_fold = []
    per_class_per_fold   = []

    for outer_fold, (train_idx, test_idx) in enumerate(
        gkf_outer.split(X, y, groups), 1
    ):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        g_train         = groups[train_idx]

        n_train_wells = len(np.unique(g_train))
        n_test_wells  = len(np.unique(groups[test_idx]))

        print(f"\n  Fold externo {outer_fold}/{N_OUTER_SPLITS} "
              f"(treino: {len(X_train):,} janelas/{n_train_wells} pocos | "
              f"teste: {len(X_test):,} janelas/{n_test_wells} pocos)")
        print(f"  Buscando melhores hiperparametros (inner CV, {N_ITER_INNER} iters x {N_INNER_SPLITS} folds)...")

        rf = RandomForestClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        )

        inner_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_grid,
            n_iter=N_ITER_INNER,
            cv=gkf_inner,
            scoring="f1_macro",
            n_jobs=1,
            random_state=RANDOM_STATE + outer_fold,
            verbose=1,
            refit=True,
        )

        # Busca de hiperparâmetros SEM ver X_test (dado de teste externo)
        inner_search.fit(X_train, y_train, groups=g_train)
        best_params    = inner_search.best_params_
        best_inner_f1  = inner_search.best_score_

        print(f"  Melhor inner F1-macro: {best_inner_f1:.4f}")
        print(f"  Melhores params: {best_params}")
        best_params_per_fold.append(best_params)

        # Avaliação no fold externo — poços que NUNCA foram vistos
        y_pred = inner_search.best_estimator_.predict(X_test)
        outer_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        outer_scores.append(outer_f1)

        print(f"  F1-macro externo (pocos novos): {outer_f1:.4f}")

        # F1 por classe neste fold (para diagnóstico detalhado)
        report_fold = classification_report(
            y_test, y_pred,
            labels=sorted(classes.tolist()),
            output_dict=True,
            zero_division=0,
        )
        fold_class_f1 = {}
        for c in sorted(classes.tolist()):
            key = str(c)
            if key in report_fold:
                fold_class_f1[c] = round(report_fold[key]["f1-score"], 4)
            else:
                fold_class_f1[c] = None
        per_class_per_fold.append(fold_class_f1)

        print(f"  Timestamp: {datetime.now().strftime('%H:%M:%S')}")

    # ── Resultados finais ─────────────────────────────────────────────────────
    mean_outer = float(np.mean(outer_scores))
    std_outer  = float(np.std(outer_scores))

    # F1 médio por classe ao longo dos folds externos
    mean_per_class = {}
    for c in sorted(classes.tolist()):
        vals = [fold[c] for fold in per_class_per_fold if fold.get(c) is not None]
        mean_per_class[c] = round(float(np.mean(vals)), 4) if vals else None

    print("\n" + "=" * 60)
    print("  Resultados do Nested CV:")
    for i, s in enumerate(outer_scores, 1):
        print(f"    Fold {i}: F1-macro = {s:.4f}")

    print(f"\n  Nested CV  — Media: {mean_outer:.4f} ± {std_outer:.4f}")
    print(f"  Flat CV    — Referencia: 0.8827")
    delta = mean_outer - 0.8827
    print(f"  Diferenca:  {delta:+.4f} "
          f"({'pessimista' if delta < 0 else 'otimista'} em relacao ao flat CV)")

    print("\n  F1 medio por classe (nested CV vs flat CV):")
    print(f"  {'Classe':<6} {'Nome':<26} {'Nested CV':>10} {'Flat CV':>10} {'Delta':>8}")
    flat_cv_per_class = {
        0: 0.9022, 1: 0.9410, 2: 0.9722, 3: 0.9641, 4: 0.9188,
        5: 0.9789, 6: 0.9904, 7: 0.0086, 8: 0.8548, 9: 0.9909,
        101: 0.9492, 102: 0.7864, 105: 0.8928, 106: 0.9772,
        107: 0.9073, 108: 0.8204, 109: 0.9616,
    }
    for c in sorted(classes.tolist()):
        nested_f1 = mean_per_class.get(c)
        flat_f1   = flat_cv_per_class.get(c)
        name      = WINDOW_CLASSES.get(c, str(c))
        if nested_f1 is not None and flat_f1 is not None:
            d = nested_f1 - flat_f1
            print(f"  {c:<6} {name:<26} {nested_f1:>10.4f} {flat_f1:>10.4f} {d:>+8.4f}")
    print("=" * 60)

    # ── Salvar resultados ─────────────────────────────────────────────────────
    results = {
        "model": "rf_nested_cv",
        "run_at": datetime.now().isoformat(),
        "config": {
            "n_outer_splits": N_OUTER_SPLITS,
            "n_inner_splits": N_INNER_SPLITS,
            "n_iter_inner":   N_ITER_INNER,
            "total_fits":     N_OUTER_SPLITS * N_ITER_INNER * N_INNER_SPLITS,
            "param_grid":     {k: v for k, v in param_grid.items()},
            "note": "Grid identico ao flat CV para comparacao justa",
        },
        "outer_fold_scores": [round(s, 4) for s in outer_scores],
        "mean_f1_macro":  round(mean_outer, 4),
        "std_f1_macro":   round(std_outer,  4),
        "best_params_per_fold": best_params_per_fold,
        "mean_per_class_f1": {str(c): v for c, v in mean_per_class.items()},
        "flat_cv_reference": {
            "model":        "rf_window_class",
            "f1_macro_cv":  0.8827,
            "strategy":     "GroupKFold(n_splits=5)",
            "n_iter_search": 20,
            "note": "RandomizedSearchCV N_ITER=20, GroupKFold(5) — mesmo grid que o nested CV",
        },
        "delta_vs_flat_cv": round(mean_outer - 0.8827, 4),
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = METRICS_DIR / "rf_nested_cv_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  Resultados salvos: {out_path}")
    print(f"  Fim: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
