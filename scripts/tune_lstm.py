"""
Busca de hiperparâmetros da CNN-LSTM via Optuna.

Estratégia:
  - Fold 0 do GroupKFold(5) é usado como fold fixo de busca.
  - MedianPruner interrompe trials ruins após as primeiras épocas.
  - TPE sampler (padrão Optuna) guia a busca com base nos trials anteriores.
  - 30 trials → estimativa: 10-15 h com pruning ativo.

Espaço de busca:
  lstm_units    : 32, 64, 128
  dropout_rate  : 0.1 – 0.5
  learning_rate : 1e-4 – 1e-2 (log-uniform)
  batch_size    : 256, 512

Workflow:
  1. python scripts/tune_lstm.py    → salva lstm_best_params.json
  2. python scripts/train_lstm.py   → carrega os melhores parâmetros e
                                       roda os 5 folds completos

Para rodar:
  python scripts/tune_lstm.py [--trials 30]
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Activation,
    MaxPooling1D, LSTM, Dropout, Dense,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CLEANED_DATA_PATH,
    FEATURES_WINDOW_PATH,
    METRICS_DIR,
    KEY_SENSORS,
    WINDOW_SIZE,
    STEP_SIZE,
    RANDOM_STATE,
    WINDOW_CLASSES,
)
from src.feature_engineering import apply_filter

LABEL_MAP     = {v: i for i, v in enumerate(sorted(WINDOW_CLASSES.keys()))}
INV_LABEL_MAP = {i: v for v, i in LABEL_MAP.items()}
N_CLASSES     = len(LABEL_MAP)
N_SENSORS     = len(KEY_SENSORS)


# ── Pré-processamento ─────────────────────────────────────────────────────────

def preprocess_instance(df: pd.DataFrame, filter_type: str) -> tuple[np.ndarray, np.ndarray]:
    n_rows = len(df)
    arr = np.zeros((n_rows, N_SENSORS), dtype=np.float32)
    for i, s in enumerate(KEY_SENSORS):
        if s not in df.columns:
            continue
        col = df[s].to_numpy(dtype=np.float64)
        finite_mask = np.isfinite(col)
        finite_vals = col[finite_mask]
        if len(finite_vals) < 2:
            continue
        mean_val = finite_vals.mean()
        std_val  = finite_vals.std()
        if std_val < 1e-8:
            continue
        normalized = np.where(finite_mask, (col - mean_val) / std_val, 0.0).astype(np.float32)
        arr[:, i] = apply_filter(normalized, filter_type).astype(np.float32)
    class_arr = (
        df["class"].to_numpy(dtype=np.float32)
        if "class" in df.columns
        else np.full(n_rows, np.nan, dtype=np.float32)
    )
    return arr, class_arr


# ── Extração de janelas ───────────────────────────────────────────────────────

def extract_windows(arr: np.ndarray, class_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X_list, y_list = [], []
    for start in range(0, len(arr) - WINDOW_SIZE + 1, STEP_SIZE):
        end   = start + WINDOW_SIZE
        valid = class_arr[start:end]
        valid = valid[~np.isnan(valid)]
        if len(valid) == 0:
            continue
        vals, counts = np.unique(valid.astype(np.int32), return_counts=True)
        lbl = int(vals[np.argmax(counts)])
        if lbl not in LABEL_MAP:
            continue
        X_list.append(arr[start:end].copy())
        y_list.append(LABEL_MAP[lbl])
    if not X_list:
        return (
            np.empty((0, WINDOW_SIZE, N_SENSORS), dtype=np.float32),
            np.empty(0, dtype=np.int32),
        )
    return np.stack(X_list), np.array(y_list, dtype=np.int32)


def collect_windows(
    iid_list, instance_arrays, instance_class
) -> tuple[np.ndarray, np.ndarray]:
    X_parts, y_parts = [], []
    for iid in iid_list:
        if iid not in instance_arrays:
            continue
        X_i, y_i = extract_windows(instance_arrays[iid], instance_class[iid])
        if len(X_i):
            X_parts.append(X_i)
            y_parts.append(y_i)
    if not X_parts:
        return (
            np.empty((0, WINDOW_SIZE, N_SENSORS), dtype=np.float32),
            np.empty(0, dtype=np.int32),
        )
    return np.concatenate(X_parts), np.concatenate(y_parts)


# ── Dataset de treino ─────────────────────────────────────────────────────────

def make_train_dataset(fit_iids, instance_arrays, instance_class, cw_dict, batch_size):
    cw_tensor = tf.constant(
        [cw_dict.get(i, 1.0) for i in range(N_CLASSES)], dtype=tf.float32
    )
    fit_iids_copy = list(fit_iids)

    def generator():
        iids = list(fit_iids_copy)
        np.random.shuffle(iids)
        for iid in iids:
            if iid not in instance_arrays:
                continue
            X_i, y_i = extract_windows(instance_arrays[iid], instance_class[iid])
            for j in range(len(X_i)):
                yield X_i[j], y_i[j]

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(WINDOW_SIZE, N_SENSORS), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )
    ds = ds.shuffle(20_000, reshuffle_each_iteration=True).repeat()
    ds = ds.map(
        lambda x, y: (x, y, tf.gather(cw_tensor, tf.cast(y, tf.int32))),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ── Callback: F1-macro + pruning Optuna ──────────────────────────────────────

class MacroF1PruningCallback(tf.keras.callbacks.Callback):
    """
    Calcula val_f1_macro a cada época, injeta nos logs do Keras
    e reporta ao Optuna para pruning. Raise TrialPruned se o trial
    deve ser interrompido.
    """
    _CHUNK = 8_192

    def __init__(self, X_val: np.ndarray, y_val: np.ndarray, trial: optuna.Trial) -> None:
        super().__init__()
        self._X     = X_val
        self._y     = y_val
        self._trial = trial

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        preds = []
        for start in range(0, len(self._X), self._CHUNK):
            chunk = self._X[start : start + self._CHUNK]
            p = self.model.predict(chunk, batch_size=512, verbose=0)
            preds.append(np.argmax(p, axis=1))
        y_pred = np.concatenate(preds)
        f1 = f1_score(self._y, y_pred, average="macro", zero_division=0)
        if logs is not None:
            logs["val_f1_macro"] = f1

        self._trial.report(f1, epoch)
        if self._trial.should_prune():
            raise optuna.TrialPruned()


# ── Arquitetura parametrizada ─────────────────────────────────────────────────

def build_model(lstm_units: int, dropout_rate: float) -> Model:
    inp = Input(shape=(WINDOW_SIZE, N_SENSORS))
    x = Conv1D(64, kernel_size=5, padding="same", use_bias=False)(inp)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(pool_size=2)(x)             # → (150, 64)
    x = Conv1D(128, kernel_size=3, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(pool_size=2)(x)             # → (75, 128)
    x = LSTM(lstm_units)(x)
    x = Dropout(dropout_rate)(x)
    out = Dense(N_CLASSES, activation="softmax")(x)
    return Model(inp, out, name="CNN_LSTM_trial")


# ── Objetivo Optuna ───────────────────────────────────────────────────────────

def make_objective(fold_data: dict):
    """
    Retorna a função objetivo com os dados do fold fixo já carregados.
    fold_data evita recarregar os arrays em cada trial.
    """
    X_val          = fold_data["X_val"]
    y_val          = fold_data["y_val"]
    fit_iids       = fold_data["fit_iids"]
    instance_arrays = fold_data["instance_arrays"]
    instance_class  = fold_data["instance_class"]
    cw_dict        = fold_data["cw_dict"]
    n_train_windows = fold_data["n_train_windows"]

    def objective(trial: optuna.Trial) -> float:
        lstm_units    = trial.suggest_categorical("lstm_units",   [32, 64, 128])
        dropout_rate  = trial.suggest_float("dropout_rate",       0.1, 0.5)
        learning_rate = trial.suggest_float("learning_rate",      1e-4, 1e-2, log=True)
        batch_size    = trial.suggest_categorical("batch_size",   [256, 512])

        model     = None
        ds_train  = None
        try:
            model    = build_model(lstm_units, dropout_rate)
            model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

            ds_train        = make_train_dataset(fit_iids, instance_arrays, instance_class, cw_dict, batch_size)
            steps_per_epoch = max(1, n_train_windows // batch_size)

            model.fit(
                ds_train,
                epochs=100,
                steps_per_epoch=steps_per_epoch,
                validation_data=(X_val, y_val),
                callbacks=[
                    MacroF1PruningCallback(X_val, y_val, trial),   # antes do EarlyStopping
                    EarlyStopping(
                        monitor="val_f1_macro", mode="max",
                        patience=15, restore_best_weights=True,
                    ),
                    ReduceLROnPlateau(
                        monitor="val_f1_macro", mode="max",
                        patience=7, factor=0.5, min_lr=1e-5,
                    ),
                ],
                verbose=0,
            )

            preds = np.argmax(model.predict(X_val, batch_size=512, verbose=0), axis=1)
            return float(f1_score(y_val, preds, average="macro", zero_division=0))

        finally:
            del model, ds_train
            gc.collect()
            tf.keras.backend.clear_session()

    return objective


# ── Pipeline principal ────────────────────────────────────────────────────────

def main(n_trials: int = 30, filter_type: str = "gaussian") -> None:
    suffix     = f"_{filter_type}" if filter_type != "gaussian" else ""
    params_out = METRICS_DIR / f"lstm{suffix}_best_params.json"
    study_db   = METRICS_DIR / f"lstm{suffix}_optuna.db"
    study_name = f"cnn_lstm_tune{suffix}"

    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    optuna.logging.set_verbosity(optuna.logging.INFO)

    # ── Carga de dados (igual ao train_lstm.py) ───────────────────────────────
    print(f"Filtro: {filter_type}")
    print("Carregando metadados...")
    df_meta = pd.read_parquet(
        FEATURES_WINDOW_PATH, columns=["instance_id", "window_label"]
    )
    print(f"  {len(df_meta):,} janelas | {df_meta['instance_id'].nunique()} instâncias")

    needed_cols = list(KEY_SENSORS) + ["instance_id", "class"]
    print("\nFase 1 — carregando cleaned.parquet em lotes...")
    pf = pq.ParquetFile(CLEANED_DATA_PATH)
    instance_raw: dict[str, list[pd.DataFrame]] = {}
    for batch in pf.iter_batches(batch_size=200_000, columns=needed_cols):
        chunk = batch.to_pandas()
        for iid, grp in chunk.groupby("instance_id", sort=False):
            instance_raw.setdefault(iid, []).append(grp)
        del chunk
    gc.collect()

    print("\nFase 2 — pré-processando para numpy...")
    instance_arrays: dict[str, np.ndarray] = {}
    instance_class:  dict[str, np.ndarray] = {}
    for iid in list(instance_raw.keys()):
        parts   = instance_raw.pop(iid)
        df_inst = pd.concat(parts, ignore_index=True)
        del parts
        arr, cls = preprocess_instance(df_inst, filter_type)
        del df_inst
        instance_arrays[iid] = arr
        instance_class[iid]  = cls
    del instance_raw
    gc.collect()
    print("  Pré-processamento concluído.")

    # ── Extrair fold 0 (fold fixo de busca) ──────────────────────────────────
    print("\nPreparando fold 0 para busca...")
    gkf         = GroupKFold(n_splits=5)
    iid_per_row = df_meta["instance_id"].values
    train_idx, _ = next(iter(gkf.split(np.arange(len(df_meta)), groups=iid_per_row)))

    train_iids = df_meta.iloc[train_idx]["instance_id"].unique()
    rng        = np.random.default_rng(RANDOM_STATE)
    n_val      = max(1, int(len(train_iids) * 0.15))
    val_iids   = rng.choice(train_iids, size=n_val, replace=False)
    fit_iids   = np.setdiff1d(train_iids, val_iids)

    y_weights = (
        df_meta[df_meta["instance_id"].isin(fit_iids)]["window_label"]
        .map(LABEL_MAP).dropna().astype(int).values
    )
    present = np.unique(y_weights)
    weights = compute_class_weight("balanced", classes=present, y=y_weights)
    cw_dict = {int(c): float(w) for c, w in zip(present, weights)}

    print("  Materializando validação...")
    X_val, y_val = collect_windows(val_iids, instance_arrays, instance_class)
    n_train_windows = int(
        df_meta[df_meta["instance_id"].isin(fit_iids)]["window_label"].notna().sum()
    )
    print(f"  X_val: {X_val.shape} | n_train_windows: {n_train_windows:,}")

    fold_data = dict(
        X_val=X_val, y_val=y_val,
        fit_iids=fit_iids,
        instance_arrays=instance_arrays,
        instance_class=instance_class,
        cw_dict=cw_dict,
        n_train_windows=n_train_windows,
    )

    # ── Estudo Optuna ─────────────────────────────────────────────────────────
    # storage em SQLite permite retomar a busca se o script for interrompido:
    #   python scripts/tune_lstm.py --trials 10   (retoma de onde parou)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=TPESampler(seed=RANDOM_STATE),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1),
        storage=f"sqlite:///{study_db}",
        load_if_exists=True,   # retoma trials anteriores se o DB existir
    )

    print(f"\nIniciando busca: {n_trials} trials | storage: {study_db}")
    study.optimize(make_objective(fold_data), n_trials=n_trials, show_progress_bar=True)

    # ── Resultados ────────────────────────────────────────────────────────────
    best = study.best_trial
    print(f"\n{'='*60}")
    print(f"Melhor trial #{best.number}  —  val_f1_macro = {best.value:.4f}")
    print(f"Parâmetros:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")
    print(f"{'='*60}")

    best_params = {
        "lstm_units":    best.params["lstm_units"],
        "dropout_rate":  best.params["dropout_rate"],
        "learning_rate": best.params["learning_rate"],
        "batch_size":    best.params["batch_size"],
        "best_val_f1_macro": best.value,
        "best_trial": best.number,
        "n_trials_completed": len([t for t in study.trials if t.state.is_finished()]),
    }
    with open(params_out, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)
    print(f"\nMelhores parâmetros salvos em: {params_out}")
    print(f"Próximo passo: python scripts/train_lstm.py --filter {filter_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30,
                        help="Número de trials Optuna (padrão: 30)")
    parser.add_argument(
        "--filter", choices=["gaussian", "statistical", "none"],
        default="gaussian", dest="filter_type",
        help="Tipo de filtro aplicado ao sinal (padrão: gaussian)",
    )
    args = parser.parse_args()
    main(n_trials=args.trials, filter_type=args.filter_type)
