"""
Engenharia de features para séries temporais de sensores de poços.

Transforma cada instância (série temporal) em janelas de tamanho fixo e
extrai características estatísticas e dinâmicas de cada janela.
"""

import gc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.ndimage import gaussian_filter1d
from scipy.special import erf
from scipy.stats import kurtosis, skew

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CLEANED_DATA_PATH,
    CRITICAL_SENSOR,
    FAULT_CLASSES,
    FEATURES_DATA_PATH,
    FFILL_LIMIT,
    GAUSSIAN_SIGMA,
    KEY_SENSORS,
    MAX_MISSING_RATIO,
    N_INSTANCES_VALIDATION,
    RAW_DATA_DIR,
    STATISTICAL_SIGMA,
    STEP_SIZE,
    VALIDATION_MODE,
    WINDOW_SIZE,
)

# Limiar abaixo do qual um sensor é considerado constante (sem variação)
_CONSTANT_THRESHOLD = 1e-10


def _apply_gaussian_filter(series: np.ndarray, sigma: float = GAUSSIAN_SIGMA) -> np.ndarray:
    """Suaviza uma série com filtro Gaussiano, preservando NaN como NaN."""
    mask = np.isnan(series)
    if mask.all():
        return series
    smoothed = series.copy()
    smoothed[~mask] = gaussian_filter1d(series[~mask], sigma=sigma)
    return smoothed


def _apply_statistical_filter(series: np.ndarray, sigma: float = STATISTICAL_SIGMA) -> np.ndarray:
    """Filtro estatístico adaptativo (forward + backward), preservando NaN como NaN.

    Diferente do filtro Gaussiano fixo, este filtro adapta o grau de suavização
    ao conteúdo do sinal a cada passo:
    - Mudanças pequenas (< sigma) → alpha ≈ 0 → suaviza (trata como ruído)
    - Mudanças grandes (>> sigma) → alpha → 1 → segue o sinal (trata como evento real)

    alpha = erf(|x_anterior - u| / (sqrt(2) * 2 * sigma))
    saída = (1 - alpha) * x_anterior  +  alpha * u

    Aplica o filtro duas vezes (frente→trás e trás→frente) para eliminar o
    atraso de fase introduzido pelo processamento causal — equivalente ao
    `filtfilt` do scipy para filtros lineares.

    Parâmetros
    ----------
    series : np.ndarray
        Série temporal de um único sensor (pode conter NaN).
    sigma : float
        Erro típico de medição em unidades z-score. Controla o limiar entre
        ruído e evento: mudanças menores que ~2*sigma são suavizadas.
        Default: STATISTICAL_SIGMA (0.5 — conservador para dados z-scored).
    """
    mask = np.isnan(series)
    if mask.all():
        return series

    valid = series[~mask].copy()
    denom = np.sqrt(2.0) * 2.0 * sigma

    def _pass(x: np.ndarray) -> np.ndarray:
        out = np.empty_like(x)
        out[0] = x[0]
        for i in range(1, len(x)):
            alpha = min(float(erf(abs(out[i - 1] - x[i]) / denom)), 1.0)
            out[i] = (1.0 - alpha) * out[i - 1] + alpha * x[i]
        return out

    # forward pass → backward pass (cancela o atraso de fase)
    forward  = _pass(valid)
    backward = _pass(forward[::-1])[::-1]

    smoothed = series.copy()
    smoothed[~mask] = backward
    return smoothed


def apply_filter(series: np.ndarray, filter_type: str) -> np.ndarray:
    """Aplica o filtro especificado a uma série 1D já normalizada (z-score)."""
    if filter_type == "gaussian":
        return _apply_gaussian_filter(series)
    elif filter_type == "statistical":
        return _apply_statistical_filter(series)
    else:  # "none"
        return series


def _normalize_instance_sensors(df_instance: pd.DataFrame,
                                  sensors: list[str]) -> pd.DataFrame:
    """Z-score por instância: remove diferenças de linha de base entre poços.

    Por que normalizar por instância e não por fold de treino?
    - Poços diferentes operam em condições absolutas distintas (ex: Poço A a 50 bar,
      Poço B a 200 bar). Se o modelo aprender que "alta pressão = falha", ele não
      generalizará para poços em outras faixas de operação.
    - Normalizando por instância, o modelo aprende PADRÕES DE MUDANÇA (a pressão subiu
      20% acima do normal daquele poço), não valores absolutos.
    - Sensores com desvio padrão ≈ 0 (constantes, desligados) são ignorados para
      evitar divisão por zero.

    Parâmetros
    ----------
    df_instance : pd.DataFrame
        Uma única instância de poço.
    sensors : list[str]
        Colunas de sensores a normalizar.

    Retorna
    -------
    pd.DataFrame
        Cópia do DataFrame com sensores normalizados.
    """
    df = df_instance.copy()
    for sensor in sensors:
        if sensor not in df.columns:
            continue
        col = df[sensor].values.astype(float)
        valid = col[~np.isnan(col)]
        if len(valid) < 2:
            continue
        std_val  = np.std(valid)
        mean_val = np.mean(valid)
        if std_val < _CONSTANT_THRESHOLD:
            # Sensor constante em qualquer nível absoluto (0, 38e6, etc.):
            # centraliza em 0 para evitar que o nível absoluto vaze como feature.
            # Sensores inativos (valor=0) e sensores travados (valor=38M) ficam
            # indistinguíveis em termos de features temporais — ambos geram std=0,
            # derivadas=0 e mean=0, o que é correto.
            df[sensor] = 0.0
            continue
        df[sensor] = (col - mean_val) / std_val
    return df


def _extract_window_features(window: np.ndarray, sensor_name: str) -> dict:
    """Extrai 11 features de uma janela de um único sensor.

    Nota sobre sinais constantes (std ≈ 0):
    Um sensor travado em zero ou em valor fixo tem skewness=0 e kurtosis=0 por
    definição matemática — não NaN. Tratamos esse caso explicitamente para evitar
    que scipy retorne NaN por 'catastrophic cancellation'.

    Preserva picos e transientes como feature (max_zscore) em vez de removê-los.
    """
    valid = window[~np.isnan(window)]
    if len(valid) < 10:
        # Janela quase vazia: sem dados suficientes para calcular estatísticas
        feature_names = [
            "mean", "std", "min", "max", "skewness", "kurtosis",
            "median", "iqr", "diff1_std", "diff2_std", "max_zscore",
        ]
        return {f"{sensor_name}_{f}": np.nan for f in feature_names}

    mean_v = valid.mean()
    std_v  = valid.std()
    diff1  = np.diff(valid)
    diff2  = np.diff(diff1)
    q75, q25 = np.percentile(valid, [75, 25])

    if std_v < _CONSTANT_THRESHOLD:
        # Sinal constante: assimetria e curtose são 0 por definição
        skew_v     = 0.0
        kurt_v     = 0.0
        max_zscore = 0.0
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            skew_v = float(skew(valid))
            kurt_v = float(kurtosis(valid))
        max_zscore = float((np.abs(valid - mean_v) / std_v).max())

    return {
        f"{sensor_name}_mean":       mean_v,
        f"{sensor_name}_std":        std_v,
        f"{sensor_name}_min":        valid.min(),
        f"{sensor_name}_max":        valid.max(),
        f"{sensor_name}_skewness":   skew_v,
        f"{sensor_name}_kurtosis":   kurt_v,
        f"{sensor_name}_median":     float(np.median(valid)),
        f"{sensor_name}_iqr":        float(q75 - q25),
        f"{sensor_name}_diff1_std":  float(diff1.std()) if len(diff1) > 1 else np.nan,
        f"{sensor_name}_diff2_std":  float(diff2.std()) if len(diff2) > 1 else np.nan,
        f"{sensor_name}_max_zscore": max_zscore,
    }


def extract_features_from_instance(df_instance: pd.DataFrame,
                                    sensors: list[str] | None = None,
                                    window_size: int = WINDOW_SIZE,
                                    step_size: int = STEP_SIZE,
                                    normalize: bool = True,
                                    label_strategy: str = "instance",
                                    smooth_filter: str = "gaussian") -> pd.DataFrame:
    """Aplica normalização por instância, janela deslizante e extrai features.

    Parâmetros
    ----------
    df_instance : pd.DataFrame
        Uma instância (sequência temporal de um poço). Deve conter
        'instance_id', 'fault_class', 'source_type'.
    sensors : list[str] | None
        Lista de colunas de sensores a usar. Se None, usa KEY_SENSORS disponíveis.
    window_size : int
        Tamanho da janela em amostras (default: 300 s a 1 Hz).
    step_size : int
        Passo entre janelas (default: 150 → 50% de sobreposição).
    normalize : bool
        Se True (default), aplica z-score por instância antes das features.
    label_strategy : str
        'instance' (default) → todas as janelas herdam fault_class da instância.
        'window'             → cada janela recebe a moda da coluna 'class'
                               (0=normal, 1-9=ativo, 101-109=transiente).
                               Janelas 100% NaN em 'class' são descartadas.
    smooth_filter : str
        'gaussian'     (default) → filtro Gaussiano fixo (scipy gaussian_filter1d).
        'statistical'            → filtro adaptativo baseado na função erf;
                                   preserva picos grandes, suaviza ruído pequeno.

    Retorna
    -------
    pd.DataFrame
        Uma linha por janela, com features + metadados de rastreabilidade.
        No modo 'window', inclui coluna 'window_label' além de 'fault_class'.
    """
    if sensors is None:
        sensors = [s for s in KEY_SENSORS if s in df_instance.columns]

    instance_id = df_instance["instance_id"].iloc[0]
    fault_class  = df_instance["fault_class"].iloc[0]
    source_type  = df_instance["source_type"].iloc[0]
    has_class_col = "class" in df_instance.columns

    # Normalização por instância: remove diferenças de linha de base entre poços
    if normalize:
        df_instance = _normalize_instance_sensors(df_instance, sensors)

    n_rows = len(df_instance)
    rows = []

    for start in range(0, n_rows - window_size + 1, step_size):
        end = start + window_size
        window_df = df_instance.iloc[start:end]

        # Rotulagem da janela
        if label_strategy == "window" and has_class_col:
            valid_states = window_df["class"].dropna()
            if valid_states.empty:
                # Janela 100% pré-evento (NaN) — descartada
                continue
            window_label = int(valid_states.mode().iloc[0])
        else:
            window_label = None

        features = {
            "instance_id":  instance_id,
            "fault_class":  fault_class,
            "source_type":  source_type,
            "window_start": start,
        }
        if window_label is not None:
            features["window_label"] = window_label

        for sensor in sensors:
            if sensor not in window_df.columns:
                continue
            raw = window_df[sensor].values.astype(float)
            if smooth_filter == "statistical":
                smoothed = _apply_statistical_filter(raw)
            elif smooth_filter == "none":
                smoothed = raw
            else:
                smoothed = _apply_gaussian_filter(raw)
            features.update(_extract_window_features(smoothed, sensor))

        rows.append(features)

    return pd.DataFrame(rows)


def extract_features_from_dataset(df: pd.DataFrame,
                                   sensors: list[str] | None = None,
                                   window_size: int = WINDOW_SIZE,
                                   step_size: int = STEP_SIZE,
                                   normalize: bool = True,
                                   label_strategy: str = "instance",
                                   smooth_filter: str = "gaussian",
                                   verbose: bool = True) -> pd.DataFrame:
    """Extrai features de todas as instâncias de um DataFrame limpo.

    Parâmetros
    ----------
    df : pd.DataFrame
        Dataset limpo com coluna 'instance_id'.
    sensors, window_size, step_size, normalize
        Repassados para extract_features_from_instance.
    label_strategy : str
        'instance' ou 'window' — repassado para extract_features_from_instance.
    smooth_filter : str
        'gaussian' ou 'statistical' — repassado para extract_features_from_instance.
    verbose : bool
        Se True, imprime progresso a cada 50 instâncias.
    """
    instance_ids = df["instance_id"].unique()
    all_features = []

    for i, instance_id in enumerate(instance_ids):
        if verbose and i % 50 == 0:
            print(f"  Processando instancia {i+1}/{len(instance_ids)}...")
        df_inst = df[df["instance_id"] == instance_id].copy()
        feat_df = extract_features_from_instance(
            df_inst, sensors, window_size, step_size,
            normalize=normalize, label_strategy=label_strategy,
            smooth_filter=smooth_filter,
        )
        all_features.append(feat_df)

    result = pd.concat(all_features, ignore_index=True)
    if verbose:
        n_meta = 5 if label_strategy == "window" else 4
        n_feat = result.shape[1] - n_meta
        print(f"\nTotal de janelas: {len(result):,} | Features por janela: {n_feat}")
    return result


# ── Pipeline em partes (memória eficiente) ────────────────────────────────────

def _clean_class_df(df_class: pd.DataFrame,
                    sensor_cols: list[str]) -> pd.DataFrame:
    """Aplica limpeza (forward-fill + filtro de qualidade) em uma única classe."""
    # Forward-fill limitado por instância (não vaza entre poços)
    df_class = df_class.copy()
    df_class[sensor_cols] = (
        df_class.groupby("instance_id", sort=False)[sensor_cols]
        .transform(lambda col: col.ffill(limit=FFILL_LIMIT))
    )

    # Descartar instâncias onde o sensor crítico tem >50% de dados ausentes
    if CRITICAL_SENSOR in df_class.columns:
        missing_ratio = (
            df_class.groupby("instance_id")[CRITICAL_SENSOR]
            .apply(lambda s: s.isna().mean())
        )
        valid = missing_ratio[missing_ratio <= MAX_MISSING_RATIO].index
        df_class = df_class[df_class["instance_id"].isin(valid)]

    return df_class


def run_pipeline_chunked(max_instances_per_class: int | None = None,
                          batch_size: int = 50,
                          sensors: list[str] | None = None,
                          window_size: int = WINDOW_SIZE,
                          step_size: int = STEP_SIZE,
                          normalize: bool = True,
                          label_strategy: str = "instance",
                          smooth_filter: str = "gaussian",
                          cleaned_path: Path = CLEANED_DATA_PATH,
                          features_path: Path = FEATURES_DATA_PATH,
                          verbose: bool = True) -> None:
    """Executa limpeza + normalização + extração de features, em sub-lotes.

    Processa `batch_size` instâncias por vez (nunca uma classe inteira de uma vez),
    evitando picos de RAM em classes com centenas de poços. Salva incrementalmente
    via PyArrow.

    Parâmetros
    ----------
    max_instances_per_class : int | None
        Limite de instâncias por classe. Se None e VALIDATION_MODE=True, usa
        N_INSTANCES_VALIDATION. Se None em modo completo, processa todas.
    batch_size : int
        Quantas instâncias carregar por vez dentro de cada classe (default: 50).
        Reduzir se a RAM ainda estiver insuficiente.
    sensors : list[str] | None
        Sensores a usar. None = KEY_SENSORS disponíveis.
    window_size, step_size
        Parâmetros da janela deslizante.
    normalize : bool
        Se True (default), aplica z-score por instância antes das features.
    label_strategy : str
        'instance' (default) → label = fault_class da instância.
        'window'             → label = moda da coluna 'class' dentro da janela.
    cleaned_path, features_path
        Caminhos de saída.
    verbose : bool
        Progresso detalhado.
    """
    from src.data_loader import _parse_source_type

    if max_instances_per_class is None and VALIDATION_MODE:
        max_instances_per_class = N_INSTANCES_VALIDATION

    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    features_path.parent.mkdir(parents=True, exist_ok=True)

    cleaned_writer: pq.ParquetWriter | None = None
    features_writer: pq.ParquetWriter | None = None
    total_windows = 0

    meta_cols = {"instance_id", "fault_class", "fault_label", "source_type",
                 "class", "state", "timestamp"}

    try:
      for fault_class, label in FAULT_CLASSES.items():
        class_dir = RAW_DATA_DIR / str(fault_class)
        if not class_dir.exists():
            continue

        all_files = sorted(class_dir.glob("*.parquet"))
        if max_instances_per_class is not None:
            all_files = all_files[:max_instances_per_class]

        n_files = len(all_files)
        n_batches = max(1, (n_files + batch_size - 1) // batch_size)

        if verbose:
            print(f"  Classe {fault_class}: {label} — {n_files} instâncias "
                  f"({n_batches} lote(s) de ate {batch_size})")

        for b_idx in range(n_batches):
            batch_files = all_files[b_idx * batch_size:(b_idx + 1) * batch_size]

            if verbose and n_batches > 1:
                print(f"    Lote {b_idx + 1}/{n_batches} "
                      f"({len(batch_files)} inst)...", end=" ", flush=True)

            # ── Carregar lote ────────────────────────────────────────────────
            frames = []
            for f in batch_files:
                df_f = pd.read_parquet(f)
                df_f["instance_id"]  = f.stem
                df_f["fault_class"]  = fault_class
                df_f["fault_label"]  = label
                df_f["source_type"]  = _parse_source_type(f.name)
                frames.append(df_f)
            df_batch = pd.concat(frames, ignore_index=True)
            del frames
            gc.collect()

            if verbose and n_batches > 1:
                print(f"{len(df_batch):,} linhas")

            sensor_cols = [c for c in df_batch.columns if c not in meta_cols]

            # ── Limpeza ──────────────────────────────────────────────────────
            df_clean = _clean_class_df(df_batch, sensor_cols)
            del df_batch
            gc.collect()

            n_inst = df_clean["instance_id"].nunique()
            if n_inst == 0:
                del df_clean
                gc.collect()
                continue

            # ── Salvar dados limpos ───────────────────────────────────────────
            table_clean = pa.Table.from_pandas(df_clean, preserve_index=False)
            if cleaned_writer is None:
                cleaned_writer = pq.ParquetWriter(cleaned_path, table_clean.schema,
                                                   compression="snappy")
            cleaned_writer.write_table(table_clean)
            del table_clean

            # ── Extração de features ──────────────────────────────────────────
            df_feat = extract_features_from_dataset(
                df_clean, sensors=sensors,
                window_size=window_size, step_size=step_size,
                label_strategy=label_strategy,
                smooth_filter=smooth_filter,
                normalize=normalize, verbose=False,
            )
            del df_clean
            gc.collect()

            total_windows += len(df_feat)
            if verbose:
                print(f"    -> {len(df_feat):,} janelas")

            # ── Salvar features ───────────────────────────────────────────────
            table_feat = pa.Table.from_pandas(df_feat, preserve_index=False)
            if features_writer is None:
                features_writer = pq.ParquetWriter(features_path, table_feat.schema,
                                                    compression="snappy")
            features_writer.write_table(table_feat)
            del df_feat, table_feat
            gc.collect()

    finally:
        if cleaned_writer:
            cleaned_writer.close()
        if features_writer:
            features_writer.close()

    print(f"\nPipeline concluido!")
    print(f"  Dados limpos -> {cleaned_path}")
    print(f"  Features     -> {features_path} ({total_windows:,} janelas)")


def run_pipeline_from_cleaned(
    cleaned_path: Path = CLEANED_DATA_PATH,
    features_path: Path = FEATURES_DATA_PATH,
    label_strategy: str = "window",
    sensors: list[str] | None = None,
    window_size: int = WINDOW_SIZE,
    step_size: int = STEP_SIZE,
    normalize: bool = True,
    smooth_filter: str = "gaussian",
    flush_every: int = 50,
    verbose: bool = True,
) -> None:
    """Extrai features do cleaned.parquet existente sem re-processar dados brutos.

    Alternativa eficiente ao run_pipeline_chunked quando cleaned.parquet ja foi
    gerado. Util para testar novas estrategias de rotulagem (label_strategy='window')
    sem risco de OOM nas instancias grandes (ex: classe 7 com 285k linhas/instancia).

    Parâmetros
    ----------
    cleaned_path : Path
        Caminho do cleaned.parquet ja gerado.
    features_path : Path
        Caminho de saida para o parquet de features.
    label_strategy : str
        'instance' ou 'window'.
    flush_every : int
        Salvar no disco a cada N instancias (controla uso de RAM).
    """
    if not cleaned_path.exists():
        raise FileNotFoundError(f"cleaned.parquet nao encontrado: {cleaned_path}")

    features_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Lendo {cleaned_path.name}...")
    df_all = pd.read_parquet(cleaned_path)
    print(f"  {len(df_all):,} linhas | {df_all['instance_id'].nunique():,} instancias")

    features_writer: pq.ParquetWriter | None = None
    total_windows = 0

    try:
        for fault_class in sorted(df_all["fault_class"].unique()):
            df_class = df_all[df_all["fault_class"] == fault_class]
            instances = sorted(df_class["instance_id"].unique())
            label = FAULT_CLASSES.get(int(fault_class), str(fault_class))

            if verbose:
                print(f"  Classe {fault_class}: {label} — {len(instances)} instancias")

            feat_rows: list[pd.DataFrame] = []
            for i, instance_id in enumerate(instances, 1):
                df_inst = df_class[df_class["instance_id"] == instance_id].copy()
                df_feat = extract_features_from_instance(
                    df_inst,
                    sensors=sensors,
                    window_size=window_size,
                    step_size=step_size,
                    normalize=normalize,
                    label_strategy=label_strategy,
                    smooth_filter=smooth_filter,
                )
                if df_feat is not None and len(df_feat) > 0:
                    feat_rows.append(df_feat)

                if i % flush_every == 0 or i == len(instances):
                    if feat_rows:
                        df_batch = pd.concat(feat_rows, ignore_index=True)
                        table = pa.Table.from_pandas(df_batch, preserve_index=False)
                        if features_writer is None:
                            features_writer = pq.ParquetWriter(
                                features_path, table.schema, compression="snappy"
                            )
                        features_writer.write_table(table)
                        total_windows += len(df_batch)
                        if verbose:
                            print(f"    -> {len(df_batch):,} janelas salvas "
                                  f"(total: {total_windows:,})")
                        feat_rows = []
                        del df_batch, table
                        gc.collect()
    finally:
        if features_writer:
            features_writer.close()

    print(f"\nPipeline concluido!")
    print(f"  Features -> {features_path} ({total_windows:,} janelas)")
