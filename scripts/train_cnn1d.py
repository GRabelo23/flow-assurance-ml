"""
Treina uma CNN-1D (FCN) para classificação de estado operacional de poços.

Arquitetura: Fully Convolutional Network — Wang et al. (2017)
  "Time Series Classification from Scratch with Deep Neural Networks: A Strong Baseline"

Input : séries temporais brutas (300 timesteps × 8 sensores), Z-score + filtro Gaussiano
Output: 17 estados operacionais (classes 0, 1-9, 101-109 exceto 103 e 104)

Para rodar:
    python scripts/train_cnn1d.py

Estimativa de tempo:
    CPU: 5-10 horas (5 folds × ~1h-2h cada)
    GPU: 1-2 horas

Estratégia de memória:
    O treino usa tf.data.Dataset com gerador — as janelas são extraídas
    on-the-fly e enviadas ao modelo em lotes, sem materializar X_train
    (~3,5 GB) em RAM. Apenas instance_groups (~1,5 GB) e X_test (~0,9 GB)
    ficam em memória simultaneamente.
"""

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
    GlobalAveragePooling1D, Dense,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CLEANED_DATA_PATH,
    FEATURES_WINDOW_PATH,
    MODELS_DIR,
    METRICS_DIR,
    KEY_SENSORS,
    WINDOW_SIZE,
    STEP_SIZE,
    RANDOM_STATE,
    WINDOW_CLASSES,
)
from src.evaluation import plot_confusion_matrix
from src.feature_engineering import (
    _normalize_instance_sensors,
    _apply_gaussian_filter,
)

CNN_STEP_SIZE = STEP_SIZE  # 150 — mesma sobreposição do RF/XGBoost

# window_label usa {0, 1-9, 101-109}; Keras exige índices contínuos 0..N-1.
LABEL_MAP = {v: i for i, v in enumerate(sorted(WINDOW_CLASSES.keys()))}
INV_LABEL_MAP = {i: v for v, i in LABEL_MAP.items()}
N_CLASSES = len(LABEL_MAP)  # 17


# ── Extração de janelas brutas ────────────────────────────────────────────────

def extract_raw_windows(df_instance: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Para uma instância (série temporal de um poço):
      1. Normaliza Z-score por sensor
      2. Aplica filtro Gaussiano (sigma = GAUSSIAN_SIGMA, mesmo que RF/XGBoost)
      3. Extrai janelas deslizantes de WINDOW_SIZE amostras com passo CNN_STEP_SIZE

    Janelas 100% pré-evento (class == NaN) e classes fora do LABEL_MAP são descartadas.

    Retorna
    -------
    X : (N_janelas, WINDOW_SIZE, n_sensores)  float32
    y : (N_janelas,)                          int32 — índices contínuos 0-16
    """
    sensors = [s for s in KEY_SENSORS if s in df_instance.columns]
    df = _normalize_instance_sensors(df_instance.copy(), sensors)
    for s in sensors:
        df[s] = _apply_gaussian_filter(df[s].values.astype(float))

    has_class = "class" in df.columns
    n_rows = len(df)
    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    for start in range(0, n_rows - WINDOW_SIZE + 1, CNN_STEP_SIZE):
        end = start + WINDOW_SIZE
        window_df = df.iloc[start:end]

        if not has_class:
            continue
        valid_states = window_df["class"].dropna()
        if valid_states.empty:
            continue
        window_label = int(valid_states.mode().iloc[0])
        if window_label not in LABEL_MAP:
            continue

        X_w = np.zeros((WINDOW_SIZE, len(sensors)), dtype=np.float32)
        for i, s in enumerate(sensors):
            vals = window_df[s].values.astype(float)
            X_w[:, i] = np.where(np.isfinite(vals), vals, 0.0)

        X_list.append(X_w)
        y_list.append(LABEL_MAP[window_label])

    if not X_list:
        return (
            np.empty((0, WINDOW_SIZE, len(sensors)), dtype=np.float32),
            np.empty(0, dtype=np.int32),
        )
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)


def collect_windows(
    iid_list: np.ndarray,
    instance_groups: dict[str, pd.DataFrame],
) -> tuple[np.ndarray, np.ndarray]:
    """Extrai e concatena janelas brutas para uma lista de instâncias (usado no teste)."""
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for iid in iid_list:
        if iid not in instance_groups:
            continue
        X_inst, y_inst = extract_raw_windows(instance_groups[iid])
        if len(X_inst):
            X_parts.append(X_inst)
            y_parts.append(y_inst)
    return np.concatenate(X_parts), np.concatenate(y_parts)


def make_train_dataset(
    iid_list: np.ndarray,
    instance_groups: dict[str, pd.DataFrame],
    cw_dict: dict[int, float],
    shuffle_buffer: int = 20_000,
) -> tf.data.Dataset:
    """
    Cria um tf.data.Dataset que extrai janelas on-the-fly sem materializar X_train.

    O gerador processa uma instância por vez e emite janelas individuais.
    Os pesos de classe são incorporados como sample_weight via tf.gather,
    pois class_weight não é compatível com tf.data no Keras.
    """
    n_sensors = len(KEY_SENSORS)
    cw_tensor = tf.constant(
        [cw_dict.get(i, 1.0) for i in range(N_CLASSES)], dtype=tf.float32
    )

    iid_list_copy = list(iid_list)

    def generator():
        iids = list(iid_list_copy)
        np.random.shuffle(iids)  # ordem das instâncias varia a cada epoch
        for iid in iids:
            if iid not in instance_groups:
                continue
            X_inst, y_inst = extract_raw_windows(instance_groups[iid])
            for xi, yi in zip(X_inst, y_inst):
                yield xi, np.int32(yi)

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(WINDOW_SIZE, n_sensors), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )
    ds = ds.shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=True)
    ds = ds.map(
        lambda x, y: (x, y, tf.gather(cw_tensor, tf.cast(y, tf.int32))),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(256).prefetch(tf.data.AUTOTUNE)
    return ds


# ── Arquitetura FCN ───────────────────────────────────────────────────────────

def build_fcn(n_sensors: int = len(KEY_SENSORS)) -> Model:
    """
    Fully Convolutional Network para classificação de séries temporais.

    3 blocos Conv1D com kernels decrescentes (8 → 5 → 3) capturam padrões
    em múltiplas escalas temporais. GlobalAveragePooling elimina hiperparâmetro
    de tamanho e reduz overfitting.
    """
    inp = Input(shape=(WINDOW_SIZE, n_sensors))
    x = inp
    for n_filters, kernel_size in [(128, 8), (256, 5), (128, 3)]:
        x = Conv1D(n_filters, kernel_size, padding="same", use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
    x = GlobalAveragePooling1D()(x)
    out = Dense(N_CLASSES, activation="softmax")(x)
    return Model(inp, out, name="FCN_1D")


# ── Pipeline principal ────────────────────────────────────────────────────────

def main() -> None:
    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    print("Carregando metadados de features...")
    df_meta = pd.read_parquet(
        FEATURES_WINDOW_PATH, columns=["instance_id", "window_label"]
    )
    n_instances = df_meta["instance_id"].nunique()
    print(f"  {len(df_meta):,} janelas | {n_instances} instâncias")

    # Lê em lotes e distribui direto por instância — nunca cria um DataFrame completo
    needed_cols = list(KEY_SENSORS) + ["instance_id", "class"]
    print("\nCarregando cleaned.parquet em lotes (pode demorar alguns minutos)...")
    pf = pq.ParquetFile(CLEANED_DATA_PATH)
    instance_data: dict[str, list[pd.DataFrame]] = {}
    for batch in pf.iter_batches(batch_size=200_000, columns=needed_cols):
        chunk = batch.to_pandas()
        for s in KEY_SENSORS:
            if s in chunk.columns:
                vals = chunk[s].to_numpy(dtype=np.float64, na_value=np.nan).copy()
                vals[~np.isfinite(vals)] = np.nan  # substitui inf antes do cast
                chunk[s] = vals.astype(np.float32)
        for iid, grp in chunk.groupby("instance_id", sort=False):
            if iid not in instance_data:
                instance_data[iid] = []
            instance_data[iid].append(grp)
        del chunk
    gc.collect()
    # Concatena dentro de cada instância (cada uma é pequena)
    instance_groups = {
        iid: pd.concat(parts, ignore_index=True)
        for iid, parts in instance_data.items()
    }
    del instance_data
    gc.collect()
    print(f"  {len(instance_groups)} instâncias em memória.")

    gkf = GroupKFold(n_splits=5)
    iid_per_row = df_meta["instance_id"].values

    oof_true: list[np.ndarray] = []
    oof_pred: list[np.ndarray] = []
    per_fold_f1: list[float] = []

    for fold, (train_idx, test_idx) in enumerate(
        gkf.split(np.arange(len(df_meta)), groups=iid_per_row)
    ):
        print(f"\n{'='*60}")
        print(f"FOLD {fold + 1} / 5")
        print(f"{'='*60}")

        train_iids = df_meta.iloc[train_idx]["instance_id"].unique()
        test_iids  = df_meta.iloc[test_idx]["instance_id"].unique()

        # 10% das instâncias de treino para validação (separação por instância)
        rng = np.random.default_rng(RANDOM_STATE + fold)
        n_val = max(1, int(len(train_iids) * 0.1))
        val_iids = rng.choice(train_iids, size=n_val, replace=False)
        fit_iids = np.setdiff1d(train_iids, val_iids)
        print(f"  Treino: {len(fit_iids)} inst | Val: {len(val_iids)} inst | Teste: {len(test_iids)} inst")

        # Pesos de classe calculados a partir dos metadados (leve, sem extrair janelas)
        train_meta = df_meta[df_meta["instance_id"].isin(fit_iids)]
        y_for_weights = (
            train_meta["window_label"]
            .map(LABEL_MAP)
            .dropna()
            .astype(int)
            .values
        )
        present = np.unique(y_for_weights)
        weights = compute_class_weight("balanced", classes=present, y=y_for_weights)
        cw_dict = {int(c): float(w) for c, w in zip(present, weights)}

        # Datasets tf.data — sem materializar X_train em RAM
        print("  Criando datasets tf.data...")
        ds_train = make_train_dataset(fit_iids, instance_groups, cw_dict, shuffle_buffer=20_000)
        ds_val   = make_train_dataset(val_iids,  instance_groups, cw_dict, shuffle_buffer=5_000)

        # steps_per_epoch estimado a partir dos metadados (sem extrair janelas)
        n_train_windows = int(df_meta[df_meta["instance_id"].isin(fit_iids)]["window_label"].notna().sum())
        steps_per_epoch = max(1, n_train_windows // 256)

        model = build_fcn(n_sensors=len(KEY_SENSORS))
        if fold == 0:
            model.summary()

        model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(
            ds_train,
            epochs=100,
            steps_per_epoch=steps_per_epoch,
            validation_data=ds_val,
            callbacks=[
                EarlyStopping(
                    monitor="val_loss",
                    patience=10,
                    restore_best_weights=True,
                    verbose=1,
                ),
                ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=5,
                    min_lr=1e-5,
                    verbose=1,
                ),
            ],
            verbose=1,
        )

        # Teste: materializa X_test (~20% dos dados, ~0,9 GB — cabe em RAM)
        print("  Extraindo janelas de teste...")
        X_test, y_test = collect_windows(test_iids, instance_groups)
        print(f"  X_test: {X_test.shape}")

        y_pred_fold = np.argmax(model.predict(X_test, batch_size=512), axis=1)
        f1 = f1_score(y_test, y_pred_fold, average="macro", zero_division=0)
        per_fold_f1.append(float(f1))
        print(f"\n  F1-macro fold {fold + 1}: {f1:.4f}")

        oof_true.append(y_test)
        oof_pred.append(y_pred_fold)

        del X_test, y_test, model, ds_train, ds_val
        gc.collect()
        tf.keras.backend.clear_session()

    # ── Métricas OOF globais ──────────────────────────────────────────────────
    y_true_all = np.concatenate(oof_true)
    y_pred_all = np.concatenate(oof_pred)

    f1_macro    = float(f1_score(y_true_all, y_pred_all, average="macro",    zero_division=0))
    f1_weighted = float(f1_score(y_true_all, y_pred_all, average="weighted", zero_division=0))
    accuracy    = float(np.mean(y_true_all == y_pred_all))

    class_names = [WINDOW_CLASSES[INV_LABEL_MAP[i]] for i in range(N_CLASSES)]

    print(f"\n{'='*60}")
    print("RESULTADOS FINAIS (OOF)")
    print(f"  F1-macro    : {f1_macro:.4f}")
    print(f"  F1-weighted : {f1_weighted:.4f}")
    print(f"  Acurácia    : {accuracy:.4f}")
    print(f"  F1 por fold : {[f'{v:.4f}' for v in per_fold_f1]}")
    print(f"{'='*60}\n")
    print(classification_report(
        y_true_all, y_pred_all,
        target_names=class_names,
        zero_division=0,
    ))

    metrics = {
        "model":        "FCN_1D",
        "f1_macro":     f1_macro,
        "f1_weighted":  f1_weighted,
        "accuracy":     accuracy,
        "f1_per_fold":  per_fold_f1,
        "per_class_f1": {
            class_names[i]: float(f1_score(
                (y_true_all == i).astype(int),
                (y_pred_all == i).astype(int),
                average="binary",
                zero_division=0,
            ))
            for i in range(N_CLASSES)
        },
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = METRICS_DIR / "cnn1d_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Métricas salvas em: {out_path}")

    y_true_orig = np.array([INV_LABEL_MAP[i] for i in y_true_all])
    y_pred_orig = np.array([INV_LABEL_MAP[i] for i in y_pred_all])
    plot_confusion_matrix(
        y_true_orig, y_pred_orig,
        model_name="cnn1d_estado_operacional",
        label_map=WINDOW_CLASSES,
        save=True,
    )


if __name__ == "__main__":
    main()
