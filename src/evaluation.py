"""
Funções de avaliação de modelos — métricas e visualizações.

Padroniza a avaliação para que todos os notebooks usem
as mesmas métricas e o mesmo formato de relatório.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FAULT_CLASSES, FIGURES_DIR, METRICS_DIR, WINDOW_CLASSES


def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    y_proba: np.ndarray | None = None,
                    model_name: str = "modelo") -> dict:
    """Calcula as principais métricas de classificação.

    Parâmetros
    ----------
    y_true : array
        Classes reais.
    y_pred : array
        Classes preditas pelo modelo.
    y_proba : array | None
        Probabilidades por classe (necessário para ROC-AUC).
    model_name : str
        Nome usado nos prints e nos arquivos salvos.

    Retorna
    -------
    dict
        Dicionário com accuracy, f1_macro, f1_weighted e roc_auc (se disponível).
    """
    from sklearn.metrics import accuracy_score
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    if y_proba is not None:
        try:
            metrics["roc_auc_macro"] = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro"
            )
        except ValueError:
            metrics["roc_auc_macro"] = np.nan

    return metrics


def print_classification_report(y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 model_name: str = "modelo") -> None:
    """Imprime relatório detalhado por classe."""
    labels = sorted(set(y_true) | set(y_pred))
    target_names = [FAULT_CLASSES.get(l, str(l)) for l in labels]
    print(f"\n{'='*60}")
    print(f"Relatório de Classificação — {model_name}")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred,
                                 labels=labels,
                                 target_names=target_names,
                                 zero_division=0))


def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          model_name: str = "modelo",
                          label_map: dict | None = None,
                          save: bool = True) -> plt.Figure:
    """Plota a matriz de confusão normalizada por linha (recall).

    Cada célula mostra a fração das amostras reais da linha que foram
    classificadas como a coluna correspondente.

    Parâmetros
    ----------
    label_map : dict | None
        Mapeamento {int_label: str_nome}. Se None, usa FAULT_CLASSES (10 classes).
        Use WINDOW_CLASSES para a abordagem de 19 estados operacionais.
    """
    if label_map is None:
        label_map = FAULT_CLASSES

    labels = sorted(label_map.keys())
    # Filtra para só mostrar classes que realmente aparecem nos dados
    labels = [l for l in labels if l in set(y_true) | set(y_pred)]
    names  = [label_map[l] for l in labels]

    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

    # Ajusta tamanho da figura conforme número de classes
    n = len(labels)
    figsize = (max(12, n * 0.9), max(9, n * 0.75))
    fig, ax = plt.subplots(figsize=figsize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=names)
    disp.plot(ax=ax, colorbar=True, cmap="Blues", values_format=".2f")
    ax.set_title(f"Matriz de Confusao — {model_name}\n(normalizada por classe real)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURES_DIR / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Figura salva em: {path}")

    return fig


def compare_models(metrics_list: list[dict], save: bool = True) -> pd.DataFrame:
    """Cria tabela comparativa dos modelos e plota gráfico de barras.

    Parâmetros
    ----------
    metrics_list : list[dict]
        Lista de dicionários retornados por compute_metrics().

    Retorna
    -------
    pd.DataFrame
        Tabela formatada para inserção no TCC.
    """
    df = pd.DataFrame(metrics_list).set_index("model")
    df = df.sort_values("f1_macro", ascending=False)

    print("\nComparação de Modelos:")
    print(df.to_string(float_format="{:.4f}".format))

    # Gráfico de barras agrupadas
    fig, ax = plt.subplots(figsize=(10, 5))
    metric_cols = [c for c in ["accuracy", "f1_macro", "f1_weighted", "roc_auc_macro"] if c in df.columns]
    df[metric_cols].plot(kind="bar", ax=ax, rot=0)
    ax.set_ylim(0, 1.05)
    ax.set_title("Comparação de Métricas por Modelo")
    ax.set_ylabel("Valor da Métrica")
    ax.legend(loc="lower right")
    plt.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURES_DIR / "comparacao_modelos.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Figura salva em: {path}")

        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = METRICS_DIR / "comparacao_modelos.csv"
        df.to_csv(csv_path)
        print(f"Tabela salva em: {csv_path}")

    return df


def evaluate_by_source(y_true: np.ndarray,
                       y_pred: np.ndarray,
                       source_types: np.ndarray,
                       model_name: str = "modelo") -> pd.DataFrame:
    """Avalia o desempenho separado por tipo de fonte (REAL, SIMULATED, DRAWN).

    Importante para o TCC: um modelo que funciona em dados simulados mas
    falha em dados reais tem pouco valor prático.
    """
    results = []
    for source in sorted(set(source_types)):
        mask = source_types == source
        if mask.sum() == 0:
            continue
        f1 = f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0)
        results.append({
            "source_type": source,
            "n_samples": int(mask.sum()),
            "f1_macro": f1,
        })

    df = pd.DataFrame(results).set_index("source_type")
    print(f"\nDesempenho por tipo de fonte — {model_name}:")
    print(df.to_string(float_format="{:.4f}".format))
    return df
