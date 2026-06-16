"""
Configurações centralizadas do projeto TCC — Garantia de Escoamento.
Todos os caminhos e constantes do pipeline ficam aqui para facilitar
ajustes sem precisar editar múltiplos notebooks.
"""

from pathlib import Path

# ── Modo de execução ──────────────────────────────────────────────────────────
#
# VALIDATION_MODE = True  → pipeline rápido com poucas instâncias por classe.
#                           Use para testar se o código funciona sem erros.
# VALIDATION_MODE = False → roda o dataset completo (pode levar horas e exige
#                           mais memória RAM; recomendado só após validar o pipeline).
#
VALIDATION_MODE = False

# Número de instâncias por classe no modo de validação
N_INSTANCES_VALIDATION = 5

# ── Caminhos base ─────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent

# Dataset bruto da Petrobras (apenas leitura)
RAW_DATA_DIR = Path(r"D:\Documentos\UnB\Projeto Final de Curso\3W\dataset")

# Dados intermediários gerados pelo pipeline
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CLEANED_DATA_PATH = PROCESSED_DIR / "cleaned.parquet"
FEATURES_DATA_PATH = PROCESSED_DIR / "features.parquet"
FEATURES_WINDOW_PATH      = PROCESSED_DIR / "features_window_class.parquet"
FEATURES_STATISTICAL_PATH = PROCESSED_DIR / "features_statistical_window_class.parquet"
FEATURES_NOFILTER_PATH    = PROCESSED_DIR / "features_nofilter_window_class.parquet"

FEATURES_BY_FILTER = {
    "gaussian":    FEATURES_WINDOW_PATH,
    "statistical": FEATURES_STATISTICAL_PATH,
    "none":        FEATURES_NOFILTER_PATH,
}

# Resultados finais
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"

# ── Classes do 3W Dataset ─────────────────────────────────────────────────────

FAULT_CLASSES = {
    0: "Normal",
    1: "Aumento Abrupto de BSW",
    2: "Fechamento Espúrio da DHSV",
    3: "Golfadas Severas",
    4: "Instabilidade de Fluxo",
    5: "Perda Rápida de Produtividade",
    6: "Restrição Rápida no PCK",
    7: "Incrustação no PCK",
    8: "Hidrato na Linha de Produção",
    9: "Hidrato na Linha de Serviço",
}

# Labels para a abordagem de rotulagem por estado operacional (17 classes)
# 0=normal, 1-9=evento ativo, 101-109=transiente (exceto 103 e 104: classes 3 e 4 não têm transiente)
_FAULT_SHORT = {
    1: "BSW", 2: "DHSV", 3: "Golfadas", 4: "Inst. Fluxo",
    5: "Prod. Rapida", 6: "PCK Restrict.", 7: "PCK Incrust.",
    8: "Hidrato Prod.", 9: "Hidrato Serv.",
}
_FAULT_WITH_TRANSIENT = {k: v for k, v in _FAULT_SHORT.items() if k not in {3, 4}}
WINDOW_CLASSES = {
    0: "Normal",
    **{k: f"{v} (Ativo)" for k, v in _FAULT_SHORT.items()},
    **{100 + k: f"{v} (Trans.)" for k, v in _FAULT_WITH_TRANSIENT.items()},
}

SOURCE_TYPES = {
    "WELL": "Real",
    "SIMULATED": "Simulado",
    "DRAWN": "Sintético",
}

# ── Sensores de interesse ─────────────────────────────────────────────────────

KEY_SENSORS = [
    "P-PDG",      # Pressão no sensor de fundo (downhole)
    "T-PDG",      # Temperatura no sensor de fundo
    "P-TPT",      # Pressão no topo da árvore de natal molhada (crítico)
    "T-TPT",      # Temperatura no topo da árvore
    "P-MON-CKP",  # Pressão a montante do choke de produção
    "T-JUS-CKP",  # Temperatura a jusante do choke de produção
    "P-JUS-CKGL", # Pressão a jusante do choke de gás lift
    "QGL",        # Vazão de gás lift
]

# ── Parâmetros da Etapa 2 — Limpeza ──────────────────────────────────────────

FFILL_LIMIT = 60          # Forward-fill: máximo de 60 amostras (= 60 s a 1 Hz)
CRITICAL_SENSOR = "P-TPT" # Sensor cujo excesso de NaN descarta a instância
MAX_MISSING_RATIO = 0.50  # Instâncias com >50% NaN no sensor crítico são descartadas

# ── Parâmetros da Etapa 3 — Engenharia de Features ───────────────────────────

WINDOW_SIZE = 300    # 300 segundos por janela
STEP_SIZE = 150      # 50% de sobreposição entre janelas
GAUSSIAN_SIGMA = 2.0     # Sigma do filtro Gaussiano (suavização de ruído)
STATISTICAL_SIGMA = 0.5  # Erro típico de medição do filtro estatístico adaptativo
                          # (em unidades z-score; mudanças menores que ~1×sigma são suavizadas)

# ── Parâmetros da Etapa 4 — Modelagem ────────────────────────────────────────
# Os valores se adaptam automaticamente ao modo de execução.

RANDOM_STATE = 42  # Semente para reprodutibilidade (não mude entre execuções)

if VALIDATION_MODE:
    # Configuração leve: rápida, baixo consumo de memória
    N_SPLITS_CV   = 3   # 3 folds (mínimo aceitável para validação)
    N_ITER_SEARCH = 5   # 5 combinações de hiperparâmetros
    N_JOBS        = 2   # 2 processos paralelos (conservador para a RAM disponível)
else:
    # Configuração completa: mais precisa, mais demorada
    N_SPLITS_CV   = 5   # 5 folds
    N_ITER_SEARCH = 20  # 20 combinações de hiperparâmetros
    N_JOBS        = 6   # 6 processos (metade dos 12 núcleos, preserva RAM)
