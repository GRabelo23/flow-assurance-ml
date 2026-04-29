"""
Carregamento dos dados brutos do 3W Dataset.

Lê os arquivos Parquet organizados por classe (pastas 0–9) e
adiciona metadados úteis para rastrear cada instância ao longo do pipeline.
"""

from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FAULT_CLASSES, N_INSTANCES_VALIDATION, RAW_DATA_DIR, VALIDATION_MODE


def _parse_source_type(filename: str) -> str:
    """Extrai o tipo de fonte a partir do nome do arquivo Parquet.

    O 3W usa três origens de dados:
    - WELL-XXXXX: dados reais de campo
    - SIMULATED: gerado por simulador
    - DRAWN: criado manualmente (sintético)
    """
    name = Path(filename).stem.upper()
    if "SIMULATED" in name:
        return "SIMULATED"
    if "DRAWN" in name:
        return "DRAWN"
    return "WELL"


def load_class(fault_class: int,
               data_dir: Path = RAW_DATA_DIR,
               max_instances: int | None = None) -> pd.DataFrame:
    """Carrega as instâncias de uma classe específica.

    Parâmetros
    ----------
    fault_class : int
        Número da classe (0 a 9).
    data_dir : Path
        Caminho raiz do dataset 3W.
    max_instances : int | None
        Limite de instâncias a carregar. None = carregar todas.
        Use VALIDATION_MODE ou N_INSTANCES_VALIDATION do config para controlar isso.

    Retorna
    -------
    pd.DataFrame
        DataFrame com todas as instâncias selecionadas, mais colunas:
        instance_id, fault_class, fault_label, source_type.
    """
    class_dir = data_dir / str(fault_class)
    if not class_dir.exists():
        raise FileNotFoundError(f"Pasta da classe {fault_class} não encontrada: {class_dir}")

    parquet_files = sorted(class_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"Nenhum arquivo .parquet em {class_dir}")

    if max_instances is not None:
        parquet_files = parquet_files[:max_instances]

    frames = []
    for filepath in parquet_files:
        df = pd.read_parquet(filepath)
        df["instance_id"] = filepath.stem
        df["fault_class"] = fault_class
        df["fault_label"] = FAULT_CLASSES[fault_class]
        df["source_type"] = _parse_source_type(filepath.name)
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def iter_classes(data_dir: Path = RAW_DATA_DIR,
                 max_instances_per_class: int | None = None,
                 verbose: bool = True):
    """Gerador que entrega uma classe por vez, evitando carregar tudo na memória.

    Uso recomendado para o pipeline de limpeza + features:

        for fault_class, df_class in iter_classes(max_instances_per_class=5):
            # processar df_class e salvar resultado
            del df_class  # liberar memória antes da próxima classe

    Parâmetros
    ----------
    data_dir : Path
        Caminho raiz do dataset 3W.
    max_instances_per_class : int | None
        Limite por classe. None = todas as instâncias.
    verbose : bool
        Se True, imprime o progresso.

    Yields
    ------
    (int, pd.DataFrame)
        Tupla (fault_class, df_class).
    """
    for fault_class in FAULT_CLASSES:
        if verbose:
            label = FAULT_CLASSES[fault_class]
            limit_info = f" (max {max_instances_per_class})" if max_instances_per_class else ""
            print(f"  Carregando classe {fault_class}: {label}{limit_info}...", end=" ", flush=True)

        df = load_class(fault_class, data_dir, max_instances=max_instances_per_class)

        if verbose:
            print(f"{df['instance_id'].nunique()} instâncias, {len(df):,} linhas")

        yield fault_class, df


def load_all_classes(data_dir: Path = RAW_DATA_DIR,
                     max_instances_per_class: int | None = None,
                     verbose: bool = True) -> pd.DataFrame:
    """Carrega todas as 10 classes em um único DataFrame.

    Atenção: carrega tudo na memória de uma vez. Para datasets grandes,
    prefira `iter_classes()` para processar uma classe por vez.

    Parâmetros
    ----------
    data_dir : Path
        Caminho raiz do dataset 3W.
    max_instances_per_class : int | None
        Limite por classe. Se None e VALIDATION_MODE=True, usa N_INSTANCES_VALIDATION.
    verbose : bool
        Se True, imprime progresso.
    """
    if max_instances_per_class is None and VALIDATION_MODE:
        max_instances_per_class = N_INSTANCES_VALIDATION

    frames = [df for _, df in iter_classes(data_dir, max_instances_per_class, verbose)]
    combined = pd.concat(frames, ignore_index=True)

    if verbose:
        print(f"\nTotal: {len(combined):,} linhas | {combined['instance_id'].nunique()} instâncias")
    return combined


def load_sample(n_instances_per_class: int = 3,
                data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Carrega uma amostra pequena para exploração rápida (EDA).

    Parâmetros
    ----------
    n_instances_per_class : int
        Quantas instâncias carregar por classe.
    """
    frames = [df for _, df in iter_classes(data_dir, max_instances_per_class=n_instances_per_class,
                                            verbose=False)]
    return pd.concat(frames, ignore_index=True)


def count_instances(data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Conta instâncias por classe sem carregar os dados."""
    rows = []
    for fault_class in FAULT_CLASSES:
        class_dir = data_dir / str(fault_class)
        files = list(class_dir.glob("*.parquet")) if class_dir.exists() else []
        source_counts = {"WELL": 0, "SIMULATED": 0, "DRAWN": 0}
        for f in files:
            source_counts[_parse_source_type(f.name)] += 1
        rows.append({
            "fault_class": fault_class,
            "fault_label": FAULT_CLASSES[fault_class],
            "total": len(files),
            **source_counts,
        })
    return pd.DataFrame(rows).set_index("fault_class")
