# Plano Metodológico — TCC Garantia de Escoamento

## Problema

Poços de petróleo offshore produzem sinais contínuos de sensores (pressão, temperatura, vazão).
Quando ocorre uma falha — como formação de hidrato no duto ou fechamento indevido de uma válvula —
os sinais mudam de padrão antes mesmo de o operador perceber. Este projeto treina modelos de ML
capazes de identificar automaticamente o estado do poço a partir dos dados dos sensores.

**Entrada:** séries temporais de 8 sensores por poço
**Saídas:**
- Abordagem 1 — Diagnóstico: qual dos 10 tipos de evento ocorreu nesse poço?
- Abordagem 2 — Estado operacional: a janela atual é normal, transiente ou evento ativo?

---

## Dataset: 3W (Petrobras)

- **Fonte:** <https://github.com/petrobras/3W>
- **Formato:** Parquet, um arquivo por instância de poço (2.228 instâncias no total)
- **Organização:** 10 pastas (0–9), uma por classe de evento
- **Tipos de fonte:** WELL (dados reais), SIMULATED (simulados), DRAWN (sintéticos)
- **Frequência:** ~1 Hz (uma amostra por segundo)
- **Coluna `class`:** estado instantâneo — 0=normal, 1–9=evento ativo, 101–109=transiente, NaN=pré-evento

---

## Separação dos Dados: Treino, Validação e Teste

### Por que não uma divisão simples 80/20?

Os dados são séries temporais de **poços individuais**. Se dividirmos janelas aleatoriamente,
janelas do *mesmo poço* aparecem tanto no treino quanto no teste — o modelo "memoriza" aquele
poço e parece ótimo, mas falha completamente em poços novos. Isso se chama **vazamento de dados**
(*data leakage*).

### A solução: GroupKFold por instância

```
2.228 poços divididos em 5 grupos de ~446 poços cada:

  ┌─────────────────────────────────────────────────────┐
  │ Fold 1 │ Treino: grupos 2,3,4,5 │ Teste: grupo 1   │
  │ Fold 2 │ Treino: grupos 1,3,4,5 │ Teste: grupo 2   │
  │ Fold 3 │ Treino: grupos 1,2,4,5 │ Teste: grupo 3   │
  │ Fold 4 │ Treino: grupos 1,2,3,5 │ Teste: grupo 4   │
  │ Fold 5 │ Treino: grupos 1,2,3,4 │ Teste: grupo 5   │
  └─────────────────────────────────────────────────────┘

Cada poço aparece em exatamente 1 fold de teste — nunca em treino e teste ao mesmo tempo.
```

**Resultado:** as métricas (F1-macro, accuracy) refletem o desempenho real em poços nunca
vistos pelo modelo — isso é o que importa para o TCC.

### Não há um "conjunto de teste fixo"?

Para um projeto acadêmico com dados limitados, o **GroupKFold com 5 folds é o padrão**:
cada fold de teste é um conjunto de poços não vistos, e a média dos 5 folds dá uma estimativa
robusta. Usar um único teste fixo de 20% desperdiçaria 20% dos dados de treinamento e produziria
uma estimativa menos estável (dependente de quais poços caíram nesse 20%).

O modelo *final* (salvo em `.joblib`) é treinado com **todos os dados** usando os melhores
hiperparâmetros encontrados — exatamente como se faz na prática.

---

## Pipeline de 6 Etapas

### Etapa 1 — Análise Exploratória (EDA)
**Pergunta:** "O que temos antes de fazer qualquer coisa?"

- Quantas instâncias por classe e por tipo de fonte?
- Quais sensores têm mais dados faltantes?
- Como são os padrões visuais de cada tipo de falha?

**Notebook:** `01_analise_exploratoria.ipynb`

---

### Etapa 2 — Limpeza e Preparação
**Pergunta:** "Como deixar os dados prontos para processar?"

- Forward-fill limitado a 60 segundos (respeita causalidade — não usa dados futuros)
- Descarte de instâncias com >50% de dados ausentes em P-TPT (sensor crítico)
- Processamento em sub-lotes de 30 instâncias por classe (`batch_size=30`) para
  respeitar o limite de ~2.7 GB de RAM disponível

**Notebook:** `02_limpeza_preparacao.ipynb` | **Saída:** `data/processed/cleaned.parquet`

---

### Etapa 3 — Engenharia de Features
**Pergunta:** "Como transformar séries temporais em números que o modelo entende?"

**Janela deslizante:** 300 s de janela, 150 s de passo (50% de sobreposição)

Por janela e por sensor, são extraídas 11 features:
- Estatísticas clássicas: média, desvio padrão, mínimo, máximo, assimetria, curtose
- Estatísticas robustas: mediana, IQR
- Dinâmica: desvio padrão da 1ª e 2ª derivada (velocidade e aceleração do sinal)
- Transientes: `max(|z-score|)` — captura picos como informação, sem removê-los

**8 sensores × 11 features = 88 features por janela**

**Por que não remover outliers?** Em falhas de escoamento, um pico abrupto de pressão *é*
o sinal da falha. Removê-los seria descartar exatamente o que o modelo precisa aprender.

**Normalização:** Z-score por instância (não pelo fold de treino), pois poços operam em
faixas absolutas distintas. O modelo aprende padrões de *mudança*, não valores absolutos.

**Duas estratégias de rotulagem implementadas:**

| Estratégia | Label | Arquivo de saída | Classes |
|-----------|-------|-----------------|---------|
| `instance` | `fault_class` da instância inteira | `features.parquet` | 10 |
| `window` | Moda da coluna `class` na janela | `features_window_class.parquet` | 19 |

**Notebook:** `03_engenharia_features.ipynb`

---

### Etapa 4 — Modelagem
**Pergunta:** "Qual algoritmo aprende melhor a distinguir os estados do poço?"

Três modelos são treinados para cada abordagem de rotulagem:

| Modelo | Por que usar |
|--------|-------------|
| **Random Forest** | Robusto, resistente a overfitting, referência de mercado — **já treinado (F1=0.9558)** |
| **XGBoost** | Gradient boosting: aprende com os erros anteriores, geralmente mais preciso |
| **MLP (Rede Neural)** | Captura padrões não-lineares complexos; treinado sobre as mesmas features |

**Configuração:**
- Validação: GroupKFold(n_splits=5) por `instance_id`
- Busca de hiperparâmetros: RandomizedSearchCV(n_iter=20)
- Métrica de otimização: **F1-macro** (penaliza erros em classes raras)
- Scripts dedicados em `scripts/` para evitar timeout de kernels Jupyter

**Notebooks:** `04_modelagem.ipynb`

---

### Etapa 5 — Avaliação
**Pergunta:** "Qual modelo é melhor, e onde ele erra?"

- F1-macro e F1-weighted por modelo
- Matriz de confusão normalizada por classe real (recall)
- Análise por tipo de fonte: REAL vs SIMULATED vs DRAWN
- Comparação das duas abordagens de rotulagem

**Notebook:** `05_avaliacao.ipynb`

---

### Etapa 6 — Interpretação
**Pergunta:** "O que o modelo aprendeu? Faz sentido fisicamente?"

- Feature Importance (MDI): ranking de quais variáveis o modelo mais usa
- SHAP values: explicação individual de cada predição
- Discussão: as features mais importantes correspondem ao que a engenharia esperaria?

**Notebook:** `06_interpretacao.ipynb`

---

## Tratamento do Desbalanceamento de Classes

### Abordagem 1 (10 classes): desbalanceamento moderado
O dataset já é desbalanceado (classe 7 com 9.499 janelas vs classe 4 com 350 em modo
validação), mas os modelos lidam bem com isso via `class_weight='balanced'`.

### Abordagem 2 (19 classes): desbalanceamento severo
Com a rotulagem por estado, o desbalanceamento é muito maior:

```
Esperado após gerar features_window_class.parquet:
  class=0   (Normal):       ~60–70% das janelas  ← DOMINANTE
  class=1–9 (Ativos):       ~3–10% cada
  class=101–109 (Transient): ~0.1–1% cada        ← MINORITÁRIAS
```

**Estratégias adotadas:**

| Estratégia | Como funciona | Quando usar |
|-----------|--------------|------------|
| `class_weight='balanced'` | Sklearn calcula automaticamente pesos inversamente proporcionais à frequência de cada classe | **Sempre** — primeira linha de defesa para RF e XGBoost |
| Métrica F1-macro | Média simples do F1 por classe (não ponderada pelo tamanho). Um modelo que ignora transientes terá F1-macro baixo mesmo com alta accuracy | **Sempre** — métrica principal de comparação |
| Análise por classe | Reportar precision, recall e F1 por classe individualmente | **Avaliação** — identificar quais transientes o modelo não detecta |
| Possível simplificação | Se transientes de um tipo tiverem <100 janelas, avaliar fusão em classe genérica "Transiente" | **Pós-análise** — depende dos dados reais |

**Por que NÃO usar SMOTE (geração sintética de amostras)?**
SMOTE gera interpolações artificiais entre amostras de séries temporais. Para sensores físicos
com dinâmicas complexas (pressão, temperatura), interpolações no espaço de features não
necessariamente produzem padrões realistas. O `class_weight='balanced'` é mais seguro e
igualmente eficaz para árvores de decisão.

---

## Escolhas Metodológicas — Resumo

| Decisão | Alternativa Descartada | Motivo |
|---------|----------------------|--------|
| GroupKFold por instância | KFold aleatório | Evita vazamento de dados entre poços |
| Preservar outliers (max_zscore) | Remover por z-score | Transientes são assinatura da falha |
| 3 modelos comparados | Apenas 1 modelo | Rigor acadêmico exige comparação |
| MLPClassifier (sklearn) | PyTorch LSTM | Simples, comparável, sem overhead de sequência |
| Forward-fill ≤ 60s | Interpolação | Respeita causalidade — não usa dados futuros |
| class_weight='balanced' | SMOTE | Interpolações em sensores físicos são irrealistas |
| Moda da janela | Estado final / estado mais grave | Representativa, robusta, sem viés |
| Z-score por instância | Z-score por fold | Evita leakage de estatísticas entre poços |
