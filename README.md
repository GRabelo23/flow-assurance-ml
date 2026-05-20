# Análise e Modelagem Integrada de Dados de Garantias de Escoamento

> **Trabalho de Conclusão de Curso** — Engenharia Mecatrônica, Universidade de Brasília (2026)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-FF6F00?logo=tensorflow&logoColor=white)
![License](https://img.shields.io/badge/Licença-Acadêmica-lightgrey)

Pipeline completo de Machine Learning para **classificação do estado operacional de poços de petróleo offshore** a partir de séries temporais de sensores, utilizando o [3W Dataset](https://github.com/petrobras/3w) da Petrobras.

---

## Sumário

- [Visão Geral](#visão-geral)
- [Dataset](#dataset)
- [Metodologia](#metodologia)
- [Resultados](#resultados)
- [Estrutura do Repositório](#estrutura-do-repositório)
- [Instalação](#instalação)
- [Configuração](#configuração)
- [Stack Tecnológica](#stack-tecnológica)
- [Autor](#autor)

---

## Visão Geral

Poços de petróleo offshore são monitorados por sensores de pressão e temperatura. Falhas como formação de hidratos, incrustação no choke e golfadas severas podem interromper a produção ou danificar equipamentos. A detecção precoce em tempo real é um problema crítico.

Este projeto desenvolve e compara três abordagens de ML para classificar, a partir de uma **janela de 300 s de dados de sensores**, o estado operacional atual de um poço entre **17 classes** (operação normal, 9 eventos ativos e 7 estados transientes):

| Abordagem | Modelos | Input |
|-----------|---------|-------|
| Ensemble com features artesanais | Random Forest, XGBoost | 88 features estatísticas |
| Rede neural sobre série bruta | CNN-1D (FCN) | Série temporal raw (300 × 8) |

A comparação direta entre as duas abordagens é o eixo central do trabalho: **qual o ganho da engenharia de features manual sobre representações aprendidas automaticamente?**

---

## Dataset

O **3W Dataset** é uma base pública da Petrobras com séries temporais reais, simuladas e sintéticas de 10 tipos de eventos em poços instrumentados.

| Item | Valor |
|------|-------|
| Instâncias (poços) | 1.409 |
| Janelas processadas | 449.397 |
| Sensores | 8 (pressão e temperatura) |
| Features por janela | 88 |
| Classes no dataset | 17 |
| Janela / passo | 300 s / 150 s (50% sobreposição) |

**Sensores:** P-PDG, T-PDG, P-TPT, T-TPT, P-MON-CKP, T-JUS-CKP, P-JUS-CKGL, QGL

**Classes:**

| Código | Estado | Código | Estado |
|--------|--------|--------|--------|
| 0 | Operação Normal | — | — |
| 1 | BSW Abrupto (ativo) | 101 | BSW Abrupto (transiente) |
| 2 | DHSV Espúrio (ativo) | 102 | DHSV Espúrio (transiente) |
| 3 | Golfadas Severas (ativo) | — | *(ausente no dataset)* |
| 4 | Instabilidade de Fluxo (ativo) | — | *(ausente no dataset)* |
| 5 | Perda de Produtividade (ativo) | 105 | Perda de Produtividade (trans.) |
| 6 | Restrição no PCK (ativo) | 106 | Restrição no PCK (trans.) |
| 7 | Incrustação no PCK (ativo) | 107 | Incrustação no PCK (trans.) |
| 8 | Hidrato Produção (ativo) | 108 | Hidrato Produção (trans.) |
| 9 | Hidrato Serviço (ativo) | 109 | Hidrato Serviço (trans.) |

> Classes 103 e 104 (transientes de Golfadas e Instabilidade) estão ausentes no dataset público.

---

## Metodologia

```
Dados Brutos → Limpeza → Features → Rotulagem → Modelagem → Avaliação → Interpretabilidade
```

### Pré-processamento

- Forward-fill causal ≤ 60 s (sem uso de dados futuros)
- Descarte de instâncias com > 50% de NaN no sensor P-TPT
- Z-score por instância (poços operam em faixas absolutas distintas)
- Filtragem de sinal: Gaussiano (σ=2), filtro estatístico adaptativo ou sem filtro

### Engenharia de Features

Janelamento deslizante com 11 estatísticas por sensor:

`mean`, `std`, `min`, `max`, `iqr`, `skewness`, `kurtosis`, `median`, `diff1_std`, `diff2_std`, `max_zscore`

→ **88 features** por janela (8 sensores × 11 estatísticas)

> Outliers são preservados via `max_zscore` — picos de pressão são assinatura de falha, não ruído.

### Validação

- **GroupKFold(5) por `instance_id`** — janelas do mesmo poço nunca aparecem em treino e teste simultaneamente
- `class_weight='balanced'` — compensa desbalanceamento severo (até 112:1 entre classes)
- **Nested CV 5×3** (300 fits) para quantificar viés de seleção de hiperparâmetros

### CNN-1D (FCN)

Fully Convolutional Network treinada diretamente sobre a série temporal bruta:

```
Conv1D(128, 8) → BN → Conv1D(256, 5) → BN → Conv1D(128, 3) → BN → GlobalAvgPool → Dense(17)
```

- EarlyStopping monitorando F1-macro
- Inferência em chunks de 8.192 janelas para evitar OOM

---

## Resultados

### Comparação Global — Estado Operacional (17 classes)

| Modelo | Filtro | F1-macro | F1-weighted | Accuracy |
|--------|--------|:--------:|:-----------:|:--------:|
| Random Forest | Gaussiano | 0.8716 | 0.9308 | 0.9322 |
| XGBoost | Sem filtro | 0.9065 | 0.9339 | 0.9345 |
| XGBoost | Estatístico | 0.9067 | 0.9401 | 0.9402 |
| **XGBoost** | **Gaussiano** | **0.9082** | 0.9361 | 0.9364 |
| CNN-1D (FCN) | Gaussiano | 0.6685 | 0.7168 | 0.7107 |

> **XGBoost + filtro Gaussiano** é o melhor modelo global. A CNN-1D opera sem qualquer engenharia de features — a diferença de 24 p.p. evidencia o valor das features artesanais (em particular `diff1_std` e `diff2_std`) para detecção de estados transientes.

### Generalização — Nested CV (RF)

| Estatística | Valor |
|-------------|-------|
| F1-macro médio | 0.8732 |
| Desvio padrão | ±0.0202 |
| Delta vs. flat CV | −0.0095 |

> Delta < 1 p.p. confirma que o flat CV com GroupKFold é uma estratégia de avaliação metodologicamente honesta — sem overfitting de hiperparâmetros.

### Destaques por Classe (XGBoost vs CNN-1D)

| Classe | XGBoost F1 | CNN-1D F1 | Observação |
|--------|:----------:|:---------:|------------|
| 9 — Hidrato Serviço (ativo) | 0.993 | **0.940** | Ambos excelentes |
| 3 — Golfadas (ativo) | 0.913 | **0.917** | CNN-1D próxima do XGBoost |
| 7 — PCK Incrustação (ativo) | **0.571** | 0.182 | Classe rara: boosting supera CNN |
| 102 — DHSV (transiente) | **0.850** | 0.385 | Transientes dependem de `diff1_std` |
| 107 — PCK Incrustação (trans.) | **0.620** | 0.000 | CNN não aprende derivadas implícitas |

### Top Features — SHAP Global (XGBoost)

| Rank | Feature | Interpretação |
|------|---------|---------------|
| 1 | `T-TPT_std` | Variabilidade de temperatura na árvore de natal molhada |
| 2 | `P-PDG_mean` | Pressão de fundo média |
| 3 | `P-TPT_max` | Pico de pressão na árvore de natal |

> `T-TPT` domina globalmente por conta da alta prevalência dos eventos de hidrato no 3W Dataset.

---

## Estrutura do Repositório

```
flow-assurance-ml/
├── config.py                        ← caminhos e constantes centralizados
├── requirements.txt
│
├── src/
│   ├── data_loader.py               ← leitura dos parquets brutos do 3W
│   ├── feature_engineering.py       ← janela deslizante, 88 features, rotulagem
│   ├── evaluation.py                ← métricas e plots de avaliação
│   └── visualization.py            ← gráficos padronizados
│
├── scripts/
│   ├── run_pipeline_window_class.py ← gera features_window_class.parquet
│   ├── train_random_forest.py       ← RF diagnóstico (10 classes)
│   ├── train_rf_window_class.py     ← RF estado operacional (17 classes)
│   ├── train_rf_nested_cv.py        ← validação nested CV (300 fits)
│   ├── train_xgboost.py             ← XGBoost diagnóstico (10 classes)
│   ├── train_xgboost_window_class.py       ← XGBoost + filtro Gaussiano
│   ├── train_xgboost_nofilter.py           ← XGBoost + sem filtro
│   ├── train_xgboost_statistical.py        ← XGBoost + filtro estatístico
│   ├── train_cnn1d.py               ← CNN-1D (FCN) sobre série bruta
│   ├── plot_confusion_matrix.py     ← matrizes de confusão
│   └── plot_shap_statistical_vs_nofilter.py ← comparação SHAP entre filtros
│
├── notebooks/
│   ├── 00_pipeline_completo.ipynb   ← pipeline completo com todos os resultados
│   ├── 01_analise_exploratoria.ipynb
│   ├── 02_limpeza_preparacao.ipynb
│   ├── 03_engenharia_features.ipynb
│   ├── 04_modelagem.ipynb
│   ├── 05_avaliacao.ipynb
│   └── 06_interpretacao.ipynb       ← SHAP e interpretabilidade
│
├── data/processed/                  ← gerado localmente (não versionado)
│   ├── cleaned.parquet
│   ├── features.parquet             ← 10 classes (diagnóstico)
│   └── features_window_class.parquet ← 17 classes (estado operacional)
│
├── results/
│   ├── models/                      ← modelos treinados (.joblib, não versionados)
│   ├── metrics/                     ← métricas em JSON/CSV
│   │   ├── rf_window_class_metrics.json
│   │   ├── xgboost_window_class_metrics.json
│   │   ├── xgboost_nofilter_metrics.json
│   │   ├── xgboost_statistical_metrics.json
│   │   ├── cnn1d_metrics.json
│   │   └── rf_nested_cv_results.json
│   └── figures/
│       ├── confusion_matrix/
│       ├── shap/
│       ├── eda/
│       └── time_series/
│
└── docs/
    ├── metodologia.md               ← documentação técnica detalhada do pipeline
    ├── model_results.md             ← resultados completos por classe
    └── cnn_tensorflow.md            ← implementação e otimizações da CNN-1D
```

---

## Instalação

### Pré-requisitos

- Python 3.10+
- [3W Dataset](https://github.com/ricardovvargas/3w_dataset) baixado localmente
- ~4 GB de RAM para o pipeline de features; ~2 GB para inferência da CNN

### Passos

```bash
# 1. Clone o repositório
git clone https://github.com/GRabelo23/flow-assurance-ml.git
cd flow-assurance-ml

# 2. Crie um ambiente virtual
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Aponte para o dataset em config.py
#    Edite RAW_DATA_DIR para o caminho local do 3W Dataset
```

### Verificação

```bash
python -c "import config; print('OK — RAW_DATA_DIR:', config.RAW_DATA_DIR)"
```

---

## Configuração

Todas as constantes ficam em `config.py`:

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `RAW_DATA_DIR` | *(definir localmente)* | Caminho para o 3W Dataset |
| `VALIDATION_MODE` | `False` | Pipeline rápido (5 inst./classe) |
| `WINDOW_SIZE` | `300` | Tamanho da janela em segundos |
| `STEP_SIZE` | `150` | Passo entre janelas (50% sobreposição) |
| `FFILL_LIMIT` | `60` | Limite máximo de forward-fill (s) |
| `MAX_MISSING_RATIO` | `0.50` | Limiar de NaN para descarte |
| `RANDOM_STATE` | `42` | Semente global |
| `N_SPLITS_CV` | `5` | Folds no GroupKFold |
| `N_ITER_SEARCH` | `20` | Iterações do RandomizedSearchCV |

---

## Stack Tecnológica

| Biblioteca | Uso principal |
|-----------|--------------|
| pandas / numpy | Manipulação de séries temporais e DataFrames |
| scikit-learn | Random Forest, GroupKFold, pré-processamento |
| xgboost | Gradient boosting para classes desbalanceadas |
| tensorflow / keras | CNN-1D (FCN) com pipeline tf.data |
| shap | Interpretabilidade (TreeExplainer) |
| pyarrow | Leitura/escrita de Parquet |
| matplotlib / seaborn | Visualizações |
| joblib | Serialização de modelos e paralelismo |

---

## Autor

**Gabriel Rabelo**  
Engenharia Mecatrônica — Universidade de Brasília (UnB)  
rabelogabriel23@gmail.com

---

**Dataset:** 3W Dataset — Petrobras / Vaz Vargas et al.  
Repositório oficial: [github.com/petrobras/3W](https://github.com/petrobras/3W)

> Este projeto é de uso acadêmico. Os dados do 3W Dataset estão sujeitos à licença original da Petrobras.
