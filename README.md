# Análise e Modelagem Integrada de Dados de Garantia de Escoamento

> **Trabalho de Conclusão de Curso** — Engenharia Mecatrônica, Universidade de Brasília (2026)
>
> Detecção e diagnóstico de falhas em poços de petróleo a partir de séries temporais de sensores, usando o [3W Dataset](https://github.com/ricardovvargas/3w_dataset) da Petrobras.

---

## Sumário

- [Contexto](#contexto)
- [Problema](#problema)
- [Metodologia](#metodologia)
- [Resultados](#resultados)
- [Estrutura do Repositório](#estrutura-do-repositório)
- [Instalação](#instalação)
- [Como Usar](#como-usar)
- [Notebooks](#notebooks)
- [Scripts de Treinamento](#scripts-de-treinamento)
- [Configuração](#configuração)
- [Stack Tecnológica](#stack-tecnológica)
- [Autor](#autor)

---

## Contexto

Poços de petróleo offshore são monitorados continuamente por sensores de pressão e temperatura. Falhas como formação de hidratos, incrustação no choke ou golfadas severas podem interromper a produção ou causar danos ao equipamento. A detecção precoce dessas anomalias em tempo real é um problema crítico para a indústria.

Este projeto desenvolve um pipeline completo de Machine Learning para duas tarefas complementares de classificação, usando o **3W Dataset** — base de dados pública da Petrobras com séries temporais reais, simuladas e sintéticas de 10 tipos de eventos em poços instrumentados.

---

## Problema

### Abordagem 1 — Diagnóstico por Instância (10 classes)

Dada uma janela de 300 s de leituras de sensores, identificar **qual tipo de falha** está associado àquele poço.

- 10 classes: Normal + 9 tipos de falha
- Uso: diagnóstico retroativo ("esse poço teve qual evento?")

### Abordagem 2 — Detecção de Estado Operacional (19 classes)

Classificar o estado instantâneo da janela: **normal**, **transiente** (falha se aproximando) ou **evento ativo**, por tipo de falha.

- 19 classes: 0 (normal), 1–9 (evento ativo), 101–109 (transiente por tipo)
- Uso: monitoramento em tempo real com detecção de progressão da falha

| Classe | Evento |
|--------|--------|
| 0 | Operação normal |
| 1 / 101 | Aumento abrupto de BSW |
| 2 / 102 | Fechamento espúrio da DHSV |
| 3 / 103 | Golfadas severas |
| 4 / 104 | Instabilidade de fluxo |
| 5 / 105 | Perda rápida de produtividade |
| 6 / 106 | Restrição rápida no PCK |
| 7 / 107 | Incrustação no PCK |
| 8 / 108 | Hidrato na linha de produção |
| 9 / 109 | Hidrato na linha de serviço |

---

## Metodologia

O pipeline segue seis etapas sequenciais:

```
Dados Brutos → Limpeza → Engenharia de Features → Modelagem → Avaliação → Interpretabilidade
```

### 1. Dados Brutos
- 1.409 instâncias (poços), ~471k timestamps
- 8 sensores de pressão e temperatura: P-PDG, T-PDG, P-TPT, T-TPT, P-MON-CKP, T-JUS-CKP, P-JUS-CKGL, QGL
- Três origens: dados reais de campo (WELL), simulados e sintéticos (DRAWN)

### 2. Limpeza
- Forward-fill causal ≤ 60 s (sem vazamento de informação futura)
- Descarte de instâncias com > 50% de NaN no sensor crítico (P-TPT)
- Z-score por instância: normaliza poços com faixas de operação distintas

### 3. Engenharia de Features
- Janela deslizante: 300 s com 50% de sobreposição (step = 150 s)
- 11 estatísticas por sensor: `mean`, `std`, `min`, `max`, `iqr`, `skewness`, `kurtosis`,`median`, `diff1_std`, `diff2_std`, `max_zscore`
- **88 features** no total (8 sensores × 11 estatísticas)
- Outliers preservados via `max_zscore` (picos são assinatura da falha)

### 4. Rotulagem
- **Abordagem 1:** janela herda a `fault_class` do poço inteiro
- **Abordagem 2:** janela recebe a moda da coluna `class` dentro da janela; janelas 100% NaN são descartadas

### 5. Modelagem
- **Separação:** `GroupKFold(n_splits=5)` por `instance_id` — janelas do mesmo poço nunca aparecem em treino e teste simultaneamente
- `class_weight='balanced'` para compensar desbalanceamento severo
- **Modelos:** Random Forest e XGBoost com busca de hiperparâmetros (`RandomizedSearchCV`)
- **Validação do overfitting:** Nested CV 5×3 (300 fits) — delta F1 vs. flat CV = −0.0095

### 6. Interpretabilidade (XAI)
- **MDI (Mean Decrease in Impurity):** importância embutida nos modelos de árvore
- **SHAP TreeExplainer:** valores exatos para RF e XGBoost; amostragem balanceada de 50 janelas por classe (850 total)
- Análise por sensor, por modelo e por classe de falha

---

## Resultados

### Abordagem 1 — Random Forest (10 classes)

| Métrica | Valor |
|---------|-------|
| F1-macro (CV) | **0.9558** |
| Número de janelas | 471.477 |
| Features | 88 |

### Abordagem 2 — Estado Operacional (19 classes)

| Métrica | Random Forest | XGBoost |
|---------|--------------|---------|
| F1-macro | 0.8716 | **0.9082** |
| F1-weighted | 0.9308 | **0.9340** |
| Accuracy | 0.9322 | **0.9364** |
| F1-macro (CV) | 0.8827 | **0.8866** |

**Nested CV (RF — validação de generalização)**

| Estatística | Valor |
|-------------|-------|
| F1-macro médio | 0.8732 |
| Desvio padrão | ±0.0202 |
| Delta vs. flat CV | −0.0095 |

> Delta próximo de zero indica **ausência de overfitting** — o modelo generaliza para poços nunca vistos.

### Destaques por Classe

| Classe | RF F1 | XGBoost F1 | Observação |
|--------|-------|-----------|------------|
| 7 — PCK Incrustação | 0.009 | **0.571** | Classe rara (0,2% do dataset); XGBoost detecta via boosting sequencial |
| 102 — DHSV Transiente | 0.793 | 0.850 | Dinâmica transiente capturada por `diff1_std` e `skewness` |
| 8 — Hidrato Produção | 0.860 | 0.890 | T-JUS-CKP_min e P-MON-CKP_max validados fisicamente |
| 9 — Hidrato Serviço | 0.991 | 0.993 | Classe mais fácil de detectar |

### Top Features (SHAP Global — Abordagem 2)

| Rank | Feature | Importância |
|------|---------|-------------|
| 1 | T-TPT\_std | Maior variabilidade global |
| 2 | P-PDG\_mean | Pressão de fundo média |
| 3 | P-TPT\_max | Pico de pressão na árvore de natal |

---

## Estrutura do Repositório

```
TCC/
├── README.md                        ← este arquivo
├── CLAUDE.md                        ← instruções para o assistente de código
├── config.py                        ← caminhos e constantes centralizados
├── requirements.txt                 ← dependências Python
│
├── src/                             ← módulos reutilizáveis
│   ├── data_loader.py               ← leitura dos parquets brutos do 3W
│   ├── feature_engineering.py      ← janela deslizante, 88 features, rotulagem
│   ├── evaluation.py               ← métricas, plots de avaliação
│   └── visualization.py            ← geração de gráficos 
│
├── scripts/                         ← execução standalone de tarefas longas
│   ├── run_pipeline_window_class.py ← gera features_window_class.parquet
│   ├── train_random_forest.py       ← treina RF Abordagem 1 (10 classes)
│   ├── train_rf_window_class.py     ← treina RF Abordagem 2 (19 classes)
│   ├── train_xgboost.py             ← treina XGBoost Abordagem 1
│   ├── train_xgboost_window_class.py← treina XGBoost Abordagem 2
│   ├── train_rf_nested_cv.py        ← validação nested CV (anti-overfitting)
│   └── plot_confusion_matrix.py     ← gera matrizes de confusão
│
├── notebooks/
│   ├── 00_pipeline_completo.ipynb   ← notebook mestre: pipeline completo e resultados
│   ├── 01_analise_exploratoria.ipynb
│   ├── 02_limpeza_preparacao.ipynb
│   ├── 03_engenharia_features.ipynb
│   ├── 04_modelagem.ipynb
│   ├── 05_avaliacao.ipynb
│   └── 06_interpretacao.ipynb       ← SHAP e MDI (XAI)
│
├── data/
│   └── processed/
│       ├── cleaned.parquet              ← séries temporais limpas
│       ├── features.parquet             ← features Abordagem 1
│       └── features_window_class.parquet← features Abordagem 2 (449k janelas)
│
├── results/
│   ├── models/                      ← modelos treinados (.joblib)
│   │   ├── random_forest.joblib
│   │   ├── rf_window_class.joblib
│   │   ├── xgboost_window_class.joblib
│   │   ├── imputer_window_class.joblib
│   │   └── label_encoder_window_class.joblib
│   ├── metrics/                     ← métricas em JSON
│   │   ├── random_forest_metrics.json
│   │   ├── rf_window_class_metrics.json
│   │   ├── xgboost_window_class_metrics.json
│   │   └── rf_nested_cv_results.json
│   └── figures/                     ← gráficos 
│       ├── eda/
│       ├── confusion_matrix/
│       └── shap/
│
└── docs/
    └── metodologia.md               ← documentação técnica detalhada do pipeline
```

---

## Instalação

### Pré-requisitos

- Python 3.10 ou superior
- 3W Dataset baixado localmente (ver [repositório oficial](https://github.com/ricardovvargas/3w_dataset))
- ~4 GB de RAM disponível para o pipeline completo

### Passos

```bash
# 1. Clone o repositório
git clone <url-do-repositório>
cd TCC

# 2. Crie um ambiente virtual
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Configure o caminho do dataset em config.py
#    Edite a variável RAW_DATA_DIR para apontar para a pasta do 3W Dataset
```

### Verificação

```bash
python -c "import config; print('RAW_DATA_DIR:', config.RAW_DATA_DIR)"
```

---

## Como Usar

### Opção 1 — Notebook Mestre (recomendado para visualizar resultados)

Abra `notebooks/00_pipeline_completo.ipynb`. Ele carrega os modelos e resultados já computados e executa em menos de 1 minuto, mostrando todas as etapas e figuras do pipeline.

```bash
jupyter notebook notebooks/00_pipeline_completo.ipynb
```

### Opção 2 — Reproduzir o Pipeline Completo do Zero

Execute os passos abaixo em ordem. Cada etapa pode levar de minutos a horas dependendo do hardware.

#### Etapa 1 — Gerar features (Abordagem 2)

```bash
python scripts/run_pipeline_window_class.py
```

Gera `data/processed/features_window_class.parquet` (~449k janelas, 88 features).

#### Etapa 2 — Treinar os modelos

```bash
# Random Forest — Abordagem 2 (19 classes)
python scripts/train_rf_window_class.py

# XGBoost — Abordagem 2 (19 classes)
python scripts/train_xgboost_window_class.py

# Random Forest — validação com Nested CV
python scripts/train_rf_nested_cv.py
```

Modelos salvos em `results/models/`. Métricas salvas em `results/metrics/`.

#### Etapa 3 — Interpretabilidade (SHAP)

Execute o notebook `notebooks/06_interpretacao.ipynb`. O cálculo dos SHAP values usa 50 janelas por classe (850 total) para viabilizar o tempo de execução com árvores profundas.

### Modo de Validação Rápida

Para testar o pipeline sem processar o dataset completo, ative o modo de validação em `config.py`:

```python
VALIDATION_MODE = True   # processa apenas 5 instâncias por classe
```

---

## Notebooks

| Notebook | Conteúdo | Execução |
|----------|----------|----------|
| `00_pipeline_completo.ipynb` | Pipeline completo com todos os resultados | < 1 min |
| `01_analise_exploratoria.ipynb` | EDA: distribuição de classes, sensores, missing data | ~5 min |
| `02_limpeza_preparacao.ipynb` | Forward-fill, z-score, descarte de instâncias | ~10 min |
| `03_engenharia_features.ipynb` | Janela deslizante, 88 features, rotulagem | ~30 min |
| `04_modelagem.ipynb` | Treinamento RF e XGBoost com busca de hiperparâmetros | ~2 h |
| `05_avaliacao.ipynb` | Métricas, matrizes de confusão, curvas ROC | ~5 min |
| `06_interpretacao.ipynb` | MDI, SHAP global, por sensor e por classe | ~15 min |

---

## Scripts de Treinamento

Scripts autônomos para tarefas computacionalmente intensas, adequados para execução em background ou servidor remoto.

| Script | Tarefa | Tempo estimado |
|--------|--------|---------------|
| `run_pipeline_window_class.py` | Geração de features Abordagem 2 | ~1–2 h |
| `train_random_forest.py` | RF Abordagem 1 (10 classes) | ~30 min |
| `train_rf_window_class.py` | RF Abordagem 2 (19 classes) | ~1 h |
| `train_xgboost_window_class.py` | XGBoost Abordagem 2 | ~1 h |
| `train_rf_nested_cv.py` | Nested CV (300 fits — anti-overfitting) | ~3–4 h |
| `plot_confusion_matrix.py` | Matrizes de confusão dos modelos | ~2 min |

---

## Configuração

Todas as constantes do projeto ficam centralizadas em `config.py`. Os principais parâmetros:

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `RAW_DATA_DIR` | *(definir localmente)* | Caminho para o 3W Dataset |
| `VALIDATION_MODE` | `False` | `True` = pipeline rápido (5 instâncias/classe) |
| `WINDOW_SIZE` | `300` | Tamanho da janela em segundos |
| `STEP_SIZE` | `150` | Passo entre janelas (50% sobreposição) |
| `FFILL_LIMIT` | `60` | Máximo de forward-fill em segundos |
| `MAX_MISSING_RATIO` | `0.50` | Limiar de NaN para descarte da instância |
| `RANDOM_STATE` | `42` | Semente global de aleatoriedade |
| `N_SPLITS_CV` | `5` | Número de folds no GroupKFold |
| `N_ITER_SEARCH` | `20` | Iterações do RandomizedSearchCV |

---

## Stack Tecnológica

| Biblioteca | Versão mínima | Uso |
|-----------|--------------|-----|
| Python | 3.10 | Linguagem base |
| pandas | 2.0 | Manipulação de séries temporais e DataFrames |
| numpy | 1.24 | Operações numéricas vetorizadas |
| scikit-learn | 1.3 | Random Forest, validação cruzada, pré-processamento |
| xgboost | 2.0 | Gradient boosting para classes desbalanceadas |
| shap | 0.44 | Interpretabilidade (SHAP TreeExplainer) |
| imbalanced-learn | 0.11 | Estratégias para classes raras |
| pyarrow | 14.0 | Leitura/escrita de Parquet |
| matplotlib / seaborn | 3.7 / 0.12 | Visualizações |
| joblib | 1.3 | Serialização de modelos e paralelismo |
| jupyter | 1.0 | Ambiente de notebooks |

---

## Autor

**Gabriel Rabelo**

Engenharia Mecatrônica — Universidade de Brasília (UnB)
Contato: rabelogabriel23@gmail.com

**Dataset:** 3W Dataset — Petrobras / Ricardo Vargas et al.
Referência: [github.com/ricardovvargas/3w_dataset](https://github.com/ricardovvargas/3w_dataset)

---

## Licença

Este projeto é de uso acadêmico. Os dados do 3W Dataset estão sujeitos à licença original da Petrobras — consulte o repositório oficial antes de usar os dados para outros fins.
