"""
Treina uma CNN-LSTM para classificação de estado operacional de poços.

Arquitetura: dois blocos Conv1D+BN+ReLU+MaxPool reduzem a sequência de
300 para 75 timesteps; uma LSTM(64) processa essa representação compacta.
Essa combinação é ~4× mais rápida que LSTM pura sobre 300 passos e
captura tanto padrões locais (CNN) quanto dependências temporais de
longo alcance (LSTM).

Comparação de arquiteturas no TCC:
    RF / XGBoost  — 88 features artesanais        → referência
    CNN-1D FCN    — série bruta, sem dependência temporal explícita
    CNN-LSTM      — série bruta, com memória temporal (este script)

Fluxo recomendado:
    1. python scripts/tune_lstm.py      → busca hiperparâmetros (fold 0, ~10h)
    2. python scripts/train_lstm.py     → treina 5 folds com os melhores parâmetros

Se lstm_best_params.json existir em results/metrics/, os hiperparâmetros
são carregados automaticamente. Caso contrário, usam-se os defaults abaixo.

Para rodar:
    python scripts/train_lstm.py
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
from sklearn.metrics import f1_score, classification_report

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Activation,
    MaxPooling1D, LSTM, Dropout, Dense,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

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
from src.evaluation import plot_confusion_matrix
from src.feature_engineering import apply_filter

# Defaults usados quando lstm_best_params.json não existe
_DEFAULT_LSTM_UNITS   = 64
_DEFAULT_DROPOUT_RATE = 0.3
_DEFAULT_LR           = 1e-3
_DEFAULT_BATCH_SIZE   = 512   # maior que CNN-1D (256) — LSTM beneficia de batches maiores

LSTM_STEP    = STEP_SIZE

LABEL_MAP    = {v: i for i, v in enumerate(sorted(WINDOW_CLASSES.keys()))}
INV_LABEL_MAP = {i: v for v, i in LABEL_MAP.items()}
N_CLASSES    = len(LABEL_MAP)
N_SENSORS    = len(KEY_SENSORS)


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
    if "class" in df.columns:
        class_arr = df["class"].to_numpy(dtype=np.float32)
    else:
        class_arr = np.full(n_rows, np.nan, dtype=np.float32)
    return arr, class_arr


# ── Extração de janelas ───────────────────────────────────────────────────────

def extract_windows(arr: np.ndarray, class_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_rows = len(arr)
    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    for start in range(0, n_rows - WINDOW_SIZE + 1, LSTM_STEP):
        end   = start + WINDOW_SIZE
        valid = class_arr[start:end]
        valid = valid[~np.isnan(valid)]
        if len(valid) == 0:
            continue
        vals, counts = np.unique(valid.astype(np.int32), return_counts=True)
        window_label = int(vals[np.argmax(counts)])
        if window_label not in LABEL_MAP:
            continue
        X_list.append(arr[start:end].copy())
        y_list.append(LABEL_MAP[window_label])
    if not X_list:
        return (
            np.empty((0, WINDOW_SIZE, N_SENSORS), dtype=np.float32),
            np.empty(0, dtype=np.int32),
        )
    return np.stack(X_list), np.array(y_list, dtype=np.int32)


def collect_windows(
    iid_list: np.ndarray,
    instance_arrays: dict[str, np.ndarray],
    instance_class: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    X_parts, y_parts = [], []
    for iid in iid_list:
        if iid not in instance_arrays:
            continue
        X_inst, y_inst = extract_windows(instance_arrays[iid], instance_class[iid])
        if len(X_inst):
            X_parts.append(X_inst)
            y_parts.append(y_inst)
    if not X_parts:
        return (
            np.empty((0, WINDOW_SIZE, N_SENSORS), dtype=np.float32),
            np.empty(0, dtype=np.int32),
        )
    return np.concatenate(X_parts), np.concatenate(y_parts)


# ── Dataset de treino ─────────────────────────────────────────────────────────

def make_train_dataset(
    fit_iids: np.ndarray,
    instance_arrays: dict[str, np.ndarray],
    instance_class: dict[str, np.ndarray],
    cw_dict: dict[int, float],
    batch_size: int,
) -> tf.data.Dataset:
    cw_tensor    = tf.constant([cw_dict.get(i, 1.0) for i in range(N_CLASSES)], dtype=tf.float32)
    fit_iids_copy = list(fit_iids)

    def generator():
        iids = list(fit_iids_copy)
        np.random.shuffle(iids)
        for iid in iids:
            if iid not in instance_arrays:
                continue
            X_inst, y_inst = extract_windows(instance_arrays[iid], instance_class[iid])
            for i in range(len(X_inst)):
                yield X_inst[i], y_inst[i]

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(WINDOW_SIZE, N_SENSORS), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )
    ds = ds.shuffle(buffer_size=20_000, reshuffle_each_iteration=True)
    ds = ds.repeat()
    ds = ds.map(
        lambda x, y: (x, y, tf.gather(cw_tensor, tf.cast(y, tf.int32))),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ── Callback: macro F1 na validação ──────────────────────────────────────────

class MacroF1Callback(tf.keras.callbacks.Callback):
    _CHUNK = 8_192

    def __init__(self, X_val: np.ndarray, y_val: np.ndarray, batch_size: int) -> None:
        super().__init__()
        self._X = X_val
        self._y = y_val
        self._batch_size = batch_size

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        preds = []
        for start in range(0, len(self._X), self._CHUNK):
            chunk = self._X[start : start + self._CHUNK]
            p = self.model.predict(chunk, batch_size=self._batch_size, verbose=0)
            preds.append(np.argmax(p, axis=1))
        y_pred = np.concatenate(preds)
        f1 = f1_score(self._y, y_pred, average="macro", zero_division=0)
        if logs is not None:
            logs["val_f1_macro"] = f1


# ── Arquitetura CNN-LSTM ──────────────────────────────────────────────────────

def build_cnn_lstm(lstm_units: int = 64, dropout_rate: float = 0.3) -> Model:
    """
    CNN reduz 300 → 75 timesteps; LSTM captura dependências temporais
    sobre essa representação compacta.

    (300, 8)
      Conv1D(64, k=5) + BN + ReLU + MaxPool(2)  → (150, 64)
      Conv1D(128, k=3) + BN + ReLU + MaxPool(2) → (75, 128)
      LSTM(lstm_units)                           → (lstm_units,)
      Dropout(dropout_rate)
      Dense(N_CLASSES, softmax)
    """
    inp = Input(shape=(WINDOW_SIZE, N_SENSORS))

    x = Conv1D(64, kernel_size=5, padding="same", use_bias=False)(inp)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(pool_size=2)(x)   # → (150, 64)

    x = Conv1D(128, kernel_size=3, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(pool_size=2)(x)   # → (75, 128)

    x = LSTM(lstm_units)(x)
    x = Dropout(dropout_rate)(x)
    out = Dense(N_CLASSES, activation="softmax")(x)

    return Model(inp, out, name="CNN_LSTM")


# ── Pipeline principal ────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filter", choices=["gaussian", "statistical", "none"],
        default="gaussian", dest="filter_type",
        help="Tipo de filtro aplicado ao sinal (padrão: gaussian)",
    )
    args = parser.parse_args()
    filter_type = args.filter_type
    suffix = f"_{filter_type}" if filter_type != "gaussian" else ""

    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    # Carrega hiperparâmetros do tune_lstm.py para o filtro correspondente
    params_path = METRICS_DIR / f"lstm{suffix}_best_params.json"
    if params_path.exists():
        with open(params_path, encoding="utf-8") as f:
            best = json.load(f)
        lstm_units   = int(best["lstm_units"])
        dropout_rate = float(best["dropout_rate"])
        lr           = float(best["learning_rate"])
        batch_size   = int(best["batch_size"])
        print(f"Hiperparâmetros carregados de {params_path.name}:")
        print(f"  lstm_units={lstm_units}, dropout={dropout_rate:.2f}, "
              f"lr={lr:.2e}, batch={batch_size}")
    else:
        lstm_units, dropout_rate, lr, batch_size = (
            _DEFAULT_LSTM_UNITS, _DEFAULT_DROPOUT_RATE, _DEFAULT_LR, _DEFAULT_BATCH_SIZE
        )
        print(f"lstm{suffix}_best_params.json não encontrado — usando hiperparâmetros default.")
        print(f"  lstm_units={lstm_units}, dropout={dropout_rate:.2f}, "
              f"lr={lr:.2e}, batch={batch_size}")

    print(f"\nFiltro: {filter_type}")
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
            if iid not in instance_raw:
                instance_raw[iid] = []
            instance_raw[iid].append(grp)
        del chunk
    gc.collect()
    print(f"  {len(instance_raw)} instâncias carregadas.")

    print("\nFase 2 — pré-processando para numpy (normalização + filtro)...")
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

    gkf = GroupKFold(n_splits=5)
    iid_per_row = df_meta["instance_id"].values

    oof_true: list[np.ndarray] = []
    oof_pred: list[np.ndarray] = []
    per_fold_f1: list[float]   = []

    for fold, (train_idx, test_idx) in enumerate(
        gkf.split(np.arange(len(df_meta)), groups=iid_per_row)
    ):
        print(f"\n{'='*60}")
        print(f"FOLD {fold + 1} / 5")
        print(f"{'='*60}")

        train_iids = df_meta.iloc[train_idx]["instance_id"].unique()
        test_iids  = df_meta.iloc[test_idx]["instance_id"].unique()

        rng     = np.random.default_rng(RANDOM_STATE + fold)
        n_val   = max(1, int(len(train_iids) * 0.15))
        val_iids = rng.choice(train_iids, size=n_val, replace=False)
        fit_iids = np.setdiff1d(train_iids, val_iids)
        print(f"  Treino: {len(fit_iids)} | Val: {len(val_iids)} | Teste: {len(test_iids)} instâncias")

        y_weights = (
            df_meta[df_meta["instance_id"].isin(fit_iids)]["window_label"]
            .map(LABEL_MAP).dropna().astype(int).values
        )
        present = np.unique(y_weights)
        weights = compute_class_weight("balanced", classes=present, y=y_weights)
        cw_dict = {int(c): float(w) for c, w in zip(present, weights)}

        print("  Materializando val e teste...")
        X_val,  y_val  = collect_windows(val_iids,  instance_arrays, instance_class)
        X_test, y_test = collect_windows(test_iids, instance_arrays, instance_class)
        print(f"  X_val: {X_val.shape} | X_test: {X_test.shape}")
        print(f"  Classes em val: {sorted(np.unique(y_val).tolist())}")

        n_train_windows = int(
            df_meta[df_meta["instance_id"].isin(fit_iids)]["window_label"].notna().sum()
        )
        steps_per_epoch = max(1, n_train_windows // batch_size)

        ds_train = make_train_dataset(fit_iids, instance_arrays, instance_class, cw_dict, batch_size)

        model = build_cnn_lstm(lstm_units=lstm_units, dropout_rate=dropout_rate)
        if fold == 0:
            model.summary()

        model.compile(
            optimizer=Adam(learning_rate=lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        model.fit(
            ds_train,
            epochs=100,
            steps_per_epoch=steps_per_epoch,
            validation_data=(X_val, y_val),
            callbacks=[
                MacroF1Callback(X_val, y_val, batch_size),
                EarlyStopping(
                    monitor="val_f1_macro",
                    mode="max",
                    patience=15,
                    restore_best_weights=True,
                    verbose=1,
                ),
                ReduceLROnPlateau(
                    monitor="val_f1_macro",
                    mode="max",
                    factor=0.5,
                    patience=7,
                    min_lr=1e-5,
                    verbose=1,
                ),
            ],
            verbose=1,
        )

        y_pred_fold = np.argmax(model.predict(X_test, batch_size=batch_size), axis=1)
        f1 = f1_score(y_test, y_pred_fold, average="macro", zero_division=0)
        per_fold_f1.append(float(f1))
        print(f"\n  F1-macro fold {fold + 1}: {f1:.4f}")

        oof_true.append(y_test)
        oof_pred.append(y_pred_fold)

        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        np.save(METRICS_DIR / f"lstm{suffix}_oof_true.npy", np.concatenate(oof_true))
        np.save(METRICS_DIR / f"lstm{suffix}_oof_pred.npy", np.concatenate(oof_pred))

        del X_val, y_val, X_test, y_test, model, ds_train
        gc.collect()
        tf.keras.backend.clear_session()

    y_true_all = np.concatenate(oof_true)
    y_pred_all = np.concatenate(oof_pred)

    f1_macro    = float(f1_score(y_true_all, y_pred_all, average="macro",    zero_division=0))
    f1_weighted = float(f1_score(y_true_all, y_pred_all, average="weighted", zero_division=0))
    accuracy    = float(np.mean(y_true_all == y_pred_all))

    present_labels   = sorted(np.unique(np.concatenate([y_true_all, y_pred_all])))
    class_names_all  = [WINDOW_CLASSES[INV_LABEL_MAP[i]] for i in range(N_CLASSES)]
    class_names_pres = [WINDOW_CLASSES[INV_LABEL_MAP[i]] for i in present_labels]

    print(f"\n{'='*60}")
    print("RESULTADOS FINAIS (OOF)")
    print(f"  F1-macro    : {f1_macro:.4f}")
    print(f"  F1-weighted : {f1_weighted:.4f}")
    print(f"  Acurácia    : {accuracy:.4f}")
    print(f"  F1 por fold : {[f'{v:.4f}' for v in per_fold_f1]}")
    print(f"{'='*60}\n")
    print(classification_report(
        y_true_all, y_pred_all,
        labels=present_labels,
        target_names=class_names_pres,
        zero_division=0,
    ))

    metrics = {
        "model":        f"CNN_LSTM{suffix}",
        "filter":       filter_type,
        "hyperparams": {
            "lstm_units":    lstm_units,
            "dropout_rate":  dropout_rate,
            "learning_rate": lr,
            "batch_size":    batch_size,
        },
        "f1_macro":     f1_macro,
        "f1_weighted":  f1_weighted,
        "accuracy":     accuracy,
        "f1_per_fold":  per_fold_f1,
        "per_class_f1": {
            class_names_all[i]: float(f1_score(
                (y_true_all == i).astype(int),
                (y_pred_all == i).astype(int),
                average="binary", zero_division=0,
            ))
            for i in range(N_CLASSES)
        },
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = METRICS_DIR / f"lstm{suffix}_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Métricas salvas em: {out_path}")

    y_true_orig = np.array([INV_LABEL_MAP[i] for i in y_true_all])
    y_pred_orig = np.array([INV_LABEL_MAP[i] for i in y_pred_all])
    plot_confusion_matrix(
        y_true_orig, y_pred_orig,
        model_name=f"lstm{suffix}_estado_operacional",
        label_map=WINDOW_CLASSES,
        save=True,
    )


if __name__ == "__main__":
    main()
