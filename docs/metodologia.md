# Metodologia — Pipeline de ML para Garantia de Escoamento

**Projeto:** TCC — Análise e Modelagem Integrada de Dados de Garantia de Escoamento
**Dataset:** 3W Dataset (Petrobras) — `github.com/petrobras/3W`
**Referência de código:** `src/`, `scripts/`, `config.py`

---

## 1. Dataset

### 1.1 Origem e formato

O 3W Dataset é um conjunto de dados público da Petrobras contendo séries temporais de sensores de poços de petróleo offshore. Cada instância corresponde a um poço individual e é armazenada como um arquivo Parquet separado.

| Atributo | Valor |
|----------|-------|
| Total de instâncias | 2.228 arquivos Parquet |
| Instâncias após limpeza | 1.409 poços |
| Frequência de amostragem | ~1 Hz (uma amostra por segundo) |
| Organização | 10 pastas (0–9), uma por classe de evento |
| Tamanho típico por instância | 5.000 – 285.000 linhas |

### 1.2 Tipos de fonte

Cada instância pertence a um dos três tipos de fonte, codificado no nome do arquivo:

| Tipo | Código no arquivo | Descrição |
|------|------------------|-----------|
| Real | `WELL_` | Dados coletados de poços reais |
| Simulado | `SIMULATED_` | Gerados por simulador de dinâmica de fluidos |
| Sintético | `DRAWN_` | Construídos manualmente por especialistas |

### 1.3 Classes de eventos

| Classe | Evento | Estado Ativo | Transiente |
|--------|--------|:------------:|:----------:|
| 0 | Operação Normal | — | — |
| 1 | Aumento Abrupto de BSW | class=1 | class=101 |
| 2 | Fechamento Espúrio da DHSV | class=2 | class=102 |
| 3 | Golfadas Severas | class=3 | class=103 |
| 4 | Instabilidade de Fluxo | class=4 | class=104 |
| 5 | Perda Rápida de Produtividade | class=5 | class=105 |
| 6 | Restrição Rápida no PCK | class=6 | class=106 |
| 7 | Incrustação no PCK | class=7 | class=107 |
| 8 | Hidrato na Linha de Produção | class=8 | class=108 |
| 9 | Hidrato na Linha de Serviço | class=9 | class=109 |

A coluna `class` do dataset registra o estado instantâneo de cada amostra:
- `NaN` — período pré-evento (poço ainda normal, antes de qualquer anomalia)
- `0` — operação normal
- `1–9` — evento ativo do tipo correspondente
- `101–109` — fase transiente (anomalia se desenvolvendo)

### 1.4 Sensores utilizados

De todos os sensores disponíveis no dataset, foram selecionados 8 com maior relevância física e menor taxa de dados ausentes:

| Sensor | Descrição física |
|--------|-----------------|
| P-PDG | Pressão no sensor de fundo do poço (downhole) |
| T-PDG | Temperatura no sensor de fundo |
| P-TPT | Pressão no topo da árvore de natal molhada (crítico) |
| T-TPT | Temperatura no topo da árvore |
| P-MON-CKP | Pressão a montante do choke de produção |
| T-JUS-CKP | Temperatura a jusante do choke de produção |
| P-JUS-CKGL | Pressão a jusante do choke de gás lift |
| QGL | Vazão de gás lift |

**P-TPT foi definido como sensor crítico:** instâncias com mais de 50% de dados ausentes neste sensor são descartadas na etapa de limpeza, pois é o sensor mais informativo para detecção de falhas.

---

## 2. Limpeza e Preparação dos Dados

**Implementação:** `src/feature_engineering.py` → `_clean_class_df()`
**Saída:** `data/processed/cleaned.parquet`

### 2.1 Processamento em sub-lotes (batch_size=30)

O dataset completo da classe 0 (Normal) possui 594 instâncias com séries de até 21.000 linhas cada — carregar tudo de uma vez ultrapassaria os ~2,7 GB de RAM disponíveis. A solução é processar **30 instâncias por vez** dentro de cada classe, salvando incrementalmente via `pyarrow.ParquetWriter`.

Cada lote processa exatamente as instâncias daquele lote e descarta os dados da memória antes de carregar o próximo.

### 2.2 Forward-fill limitado (FFILL_LIMIT = 60 amostras)

Sensores de poços frequentemente têm lacunas de dados por perda de comunicação ou manutenção. O tratamento usa **preenchimento por propagação anterior (forward-fill)** limitado a 60 amostras consecutivas (equivalente a 60 segundos a 1 Hz).

```
Exemplo: [..., 38.5, NaN, NaN, NaN, 38.7, ...]
Após ffill(limit=60): [..., 38.5, 38.5, 38.5, 38.5, 38.7, ...]
```

**Por que limitar a 60?**
- Forward-fill respeita causalidade: usa apenas o último valor conhecido, não dados futuros
- Lacunas maiores que 60 s permanecem como `NaN` — o modelo saberá que o sensor estava ausente
- Lacunas de horas ou dias não devem ser preenchidas: o sensor pode ter mudado seu comportamento

O forward-fill é aplicado **por instância** (por poço), nunca entre instâncias diferentes.

### 2.3 Descarte de instâncias com sensor crítico ausente

Após o forward-fill, instâncias onde **P-TPT ainda tem mais de 50% de NaN** são descartadas completamente:

```
missing_ratio = (NaN restantes em P-TPT) / (total de amostras)
Se missing_ratio > 0.50 → instância descartada
```

Isso elimina poços com dados tão fragmentados que qualquer análise seria especulativa. O limiar de 50% é conservador: mantém instâncias com falhas de sensor pontuais, descarta apenas as cronicamente ausentes.

---

## 3. Normalização por Instância (Z-score)

**Implementação:** `src/feature_engineering.py` → `_normalize_instance_sensors()`

Antes de extrair features, cada série temporal é normalizada **por instância individualmente**, usando Z-score:

```
x_norm = (x - média_instância) / desvio_padrão_instância
```

### Por que por instância e não global?

Considere dois poços:
- Poço A opera normalmente a 50 bar de pressão
- Poço B opera normalmente a 200 bar de pressão

Se normalizarmos globalmente, o modelo aprenderá que "pressão alta = poço B". Mas o que interessa é o **padrão de mudança**: a pressão do Poço A subiu 20% acima do seu próprio normal? Isso é anomalia — independente do valor absoluto.

Ao normalizar por instância, o modelo aprende padrões relativos (mudanças, tendências, variações) que generalizam para qualquer poço.

### Tratamento de sensores constantes

Sensores com desvio padrão menor que 1×10⁻¹⁰ (travados em um valor fixo) são tratados como um caso especial:

```python
if std < 1e-10:
    df[sensor] = 0.0  # centraliza em zero, mantém variação = 0
```

Isso evita divisão por zero e garante que um sensor desligado (valor=0) e um sensor travado em 38 MPa produzam o mesmo resultado de features — ambos mostram variação zero, que é a informação relevante.

---

## 4. Janelamento Deslizante

**Implementação:** `src/feature_engineering.py` → `extract_features_from_instance()`
**Parâmetros:** `WINDOW_SIZE=300`, `STEP_SIZE=150`

### 4.1 Conceito

Modelos de ML clássicos (Random Forest, XGBoost) não processam séries temporais diretamente — precisam de vetores de features de tamanho fixo. A solução é **dividir cada série em janelas de tamanho fixo** e extrair estatísticas de cada janela.

```
Série temporal de 1.000 s:
|← 300 s →|
          |← 300 s →|
                    |← 300 s →|
  passo=150 s (50% de sobreposição)

Resultado: ceil((1000 - 300) / 150) + 1 = 5 janelas
```

### 4.2 Parâmetros de janela

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| Tamanho da janela | 300 s (5 min) | Captura dinâmicas relevantes; transientes duram minutos |
| Passo entre janelas | 150 s (50% overlap) | Aumenta o número de amostras sem perder continuidade temporal |
| Suavização Gaussiana | sigma=2,0 | Remove ruído de alta frequência preservando tendências |

### 4.3 Suavização Gaussiana

Antes de extrair as features estatísticas, cada janela de sensor é suavizada com um filtro Gaussiano (sigma=2):

```python
smoothed = gaussian_filter1d(window, sigma=2.0)
```

O filtro Gaussiano atenua oscilações de alta frequência (ruído de medição) sem distorcer a forma geral do sinal. Valores `NaN` são preservados: o filtro é aplicado apenas nos pontos não-ausentes.

---

## 5. Engenharia de Features

**Implementação:** `src/feature_engineering.py` → `_extract_window_features()`

Para cada janela de 300 s, e para cada um dos 8 sensores, são extraídas **11 features**:

**Total: 8 sensores × 11 features = 88 features por janela**

### 5.1 Descrição das features

| Feature | Fórmula / Descrição | O que captura |
|---------|---------------------|---------------|
| `mean` | Média aritmética | Nível médio do sinal na janela |
| `std` | Desvio padrão | Variabilidade geral |
| `min` | Mínimo | Vale mínimo atingido |
| `max` | Máximo | Pico máximo atingido |
| `median` | Mediana | Nível central robusto a outliers |
| `iqr` | Interquartil (Q75–Q25) | Dispersão robusta a outliers |
| `skewness` | Assimetria da distribuição | Distribuição assimétrica (ex: picos positivos de pressão) |
| `kurtosis` | Curtose da distribuição | Caudas pesadas; presença de eventos extremos |
| `diff1_std` | Desvio padrão da 1ª derivada | Velocidade de variação do sinal |
| `diff2_std` | Desvio padrão da 2ª derivada | Aceleração de variação (mudanças abruptas) |
| `max_zscore` | max(\|x − μ\| / σ) | Pico máximo em unidades de desvio padrão |

### 5.2 Por que preservar outliers como max_zscore?

Em falhas de escoamento, um pico abrupto de pressão **é** o sinal da falha — não é ruído. Removê-lo seria descartar exatamente o que o modelo precisa detectar. A feature `max_zscore` captura a magnitude do pico mais extremo da janela, transformando um outlier em informação estruturada.

### 5.3 Tratamento de janelas quase vazias

Quando um sensor tem menos de 10 leituras válidas (não-NaN) dentro de uma janela de 300 s — por exemplo, um sensor offline por quase toda a janela — não há dados suficientes para calcular estatísticas confiáveis. Nesse caso, todas as 11 features daquele sensor recebem `NaN` para aquela janela.

Após a extração, a matriz X (janelas × features) pode conter NaN em posições onde sensores estavam ausentes. Esses NaN são preenchidos antes do treinamento por um `SimpleImputer(strategy='median')`, que substitui cada NaN pela **mediana daquela feature calculada sobre todas as janelas do dataset** — não pela mediana interna da janela com problema.

Exemplo: se `P-TPT_mean` for NaN em uma janela (sensor ausente naquele período), o valor imputado é a mediana de `P-TPT_mean` calculada sobre todas as ~449.000 janelas do dataset. A lógica é conservadora: assume-se o valor mais típico disponível para aquela feature.

O imputer é salvo em `.joblib` junto com o modelo para garantir que a mesma transformação seja aplicada na inferência.

---

## 6. Estratégias de Rotulagem

Foram implementadas duas estratégias de rotulagem que geram dois datasets distintos para duas tarefas de classificação complementares.

### 6.1 Abordagem 1 — Rotulagem por Instância

**Arquivo:** `data/processed/features.parquet`
**Label:** `fault_class` — a classe do evento do poço inteiro (0–9)

Cada janela herda a classe do poço ao qual pertence. Uma janela extraída de um poço de classe 3 (Golfadas) recebe label=3, independentemente do estado instantâneo naquela janela.

**Pergunta respondida:** "Dado um histórico de sensores desse poço, qual tipo de falha ele apresenta?"

**Uso:** Diagnóstico retroativo — classificar o tipo de evento após análise da série completa.

**10 classes:** Normal, BSW, DHSV, Golfadas, Instabilidade de Fluxo, Prod. Rápida, PCK Restrição, PCK Incrustação, Hidrato Produção, Hidrato Serviço.

### 6.2 Abordagem 2 — Rotulagem por Estado Operacional

**Arquivo:** `data/processed/features_window_class.parquet`
**Label:** `window_label` — a moda da coluna `class` dentro daquela janela

```python
valid_states = window_df['class'].dropna()
if valid_states.empty:
    continue  # janela 100% NaN descartada
window_label = int(valid_states.mode().iloc[0])
```

**Pergunta respondida:** "Qual é o estado operacional do poço neste momento?"

**Uso:** Monitoramento em tempo real — detectar a progressão: Normal → Transiente → Evento Ativo.

**17 classes encontradas no dataset:** 0 (Normal), 1–9 (Ativo por tipo), 101–109 (Transiente por tipo).
Classes 103 e 104 (transientes de Golfadas e Instabilidade de Fluxo) não aparecem no dataset — nessas falhas, o sinal vai de normal direto para o estado ativo sem transiente registrado.

**Janelas descartadas:** aquelas onde 100% dos timestamps têm `class=NaN` (período pré-evento puro). Essas janelas não têm estado definido e não contribuem para nenhuma das 17 classes.

### 6.3 Comparação das abordagens

| Aspecto | Abordagem 1 (Instância) | Abordagem 2 (Estado) |
|---------|------------------------|---------------------|
| Label de uma janela normal antes da falha | classe do evento (ex: 3) | 0 (Normal) |
| Label de um transiente | classe do evento (ex: 3) | 103 (Transiente tipo 3) |
| Label do estado ativo | classe do evento (ex: 3) | 3 (Ativo tipo 3) |
| Janelas pré-evento | incluídas com label errado | descartadas |
| Aplicação | Diagnóstico | Monitoramento em tempo real |

---

## 7. Separação Treino/Teste — GroupKFold

**Implementação:** `sklearn.model_selection.GroupKFold(n_splits=5)`
**Chave de agrupamento:** `instance_id` (identificador único de cada poço)

### 7.1 O problema de data leakage em séries temporais

Se dividirmos as janelas aleatoriamente (KFold padrão), janelas do **mesmo poço** aparecem tanto no treino quanto no teste. O modelo "memoriza" aquele poço e parece ter desempenho excelente — mas falha em poços novos. Isso é **data leakage** (vazamento de dados), um erro metodológico grave.

### 7.2 A solução: GroupKFold por instance_id

```
1.409 poços divididos em 5 grupos de ~282 poços:

  ┌──────────────────────────────────────────────────────────┐
  │ Fold 1 │ Treino: grupos 2,3,4,5 │ Teste: grupo 1         │
  │ Fold 2 │ Treino: grupos 1,3,4,5 │ Teste: grupo 2         │
  │ Fold 3 │ Treino: grupos 1,2,4,5 │ Teste: grupo 3         │
  │ Fold 4 │ Treino: grupos 1,2,3,5 │ Teste: grupo 4         │
  │ Fold 5 │ Treino: grupos 1,2,3,4 │ Teste: grupo 5         │
  └──────────────────────────────────────────────────────────┘

Cada poço aparece em exatamente 1 fold de teste.
```

As métricas finais (F1-macro, accuracy) refletem o desempenho em **poços completamente novos** — nunca vistos pelo modelo durante o treinamento. Isso é o que importa para aplicação real.

### 7.3 Modelo final

O modelo final salvo em `.joblib` é treinado com **todos os dados** usando os melhores hiperparâmetros encontrados. As métricas reportadas são do cross-validation (honesto), não do modelo final (que usou todos os dados para treinar e não pode ser avaliado neles mesmos).

### 7.4 Limitação: vazamento de dados na seleção de hiperparâmetros

Existe uma forma de vazamento mais sutil, que vale registrar como limitação do estudo.

**O que acontece no pipeline:**

```
Passo 1 — RandomizedSearchCV (20 configs × 5 folds = 100 fits):
  Usa TODOS os 1.409 poços para escolher os melhores hiperparâmetros.
  Cada poço aparece em 1 fold de teste durante esse processo.

Passo 2 — Avaliação final (cross_val_predict):
  Usa os MESMOS 1.409 poços e os MESMOS folds para reportar as métricas.
```

O problema: os hiperparâmetros foram escolhidos **justamente porque funcionaram bem nesses folds específicos**. Avaliar com os mesmos folds produz métricas ligeiramente otimistas — o modelo foi, indiretamente, ajustado para essa partição dos dados.

**Comparação com o vazamento grave (sem GroupKFold):**

| | Sem GroupKFold | Vazamento de hiperparâmetros |
|---|---|---|
| O que vaza | Janelas do mesmo poço em treino e teste | Escolha de hiperparâmetros influenciada pelo conjunto de teste |
| Gravidade | Alta — o modelo memoriza o poço | Baixa — hiperparâmetros robustos mudam pouco entre partições |
| Efeito típico | F1 inflado em 10–30% | F1 inflado em ~1–2% |

**A solução correta: Validação Cruzada Aninhada (Nested Cross-Validation)**

```
Loop externo — GroupKFold(5):
  Para cada fold externo:
    └─ Loop interno — RandomizedSearchCV com GroupKFold(3):
         Busca hiperparâmetros usando APENAS os poços do treino externo
    └─ Avalia no fold de teste externo com os hiperparâmetros do loop interno
         (esses poços NUNCA foram vistos na seleção de hiperparâmetros)

Resultado: 5 scores verdadeiramente independentes
```

**Resultados do Nested CV — RF Estado Operacional (Abordagem 2):**

O nested CV foi executado com configuração idêntica ao flat CV (mesmo grid, N\_ITER=20), adicionando apenas o loop externo de proteção. Total: 300 treinamentos (~6 horas).

| Fold | Outer F1-macro |
|:----:|:--------------:|
| 1 | 0.8883 |
| 2 | 0.8415 |
| 3 | 0.8743 |
| 4 | 0.8624 |
| 5 | 0.8996 |
| **Média ± DP** | **0.8732 ± 0.0202** |

| Método | F1-macro |
|--------|:--------:|
| Flat CV (GroupKFold 5) | 0.8827 |
| Nested CV | 0.8732 |
| **Viés medido** | **−0.0095 (−0,95%)** |

**Interpretação dos resultados:**

- O viés de seleção de hiperparâmetros é **real e mensurável**, mas pequeno: o flat CV superestima o desempenho em 0,95% de F1-macro
- **Nenhuma classe individual foi distorcida de forma relevante**: todas as diferenças por classe estão abaixo de ±0,03
- A Classe 7 (PCK Incrustação Ativo, F1≈0,009 no flat CV e ≈0,005 no nested CV) confirma que o problema dessa classe é estrutural — apenas 914 janelas (0,2% do dataset) — e não um artefato do método de avaliação
- O desvio padrão de ±0,0202 entre os folds externos reflete a variabilidade natural entre grupos de poços: alguns grupos contêm poços mais difíceis de generalizar (Fold 2: F1=0,8415) e outros mais representativos (Fold 5: F1=0,8996)
- **Conclusão:** A metodologia de flat CV com GroupKFold(5) é essencialmente honesta para este dataset. As comparações entre modelos são válidas, pois todos foram avaliados sob a mesma metodologia e o viés afeta todos igualmente.

---

## 8. Tratamento do Desbalanceamento de Classes

### 8.1 Abordagem 1 (10 classes) — desbalanceamento moderado

| Classe | Janelas (aprox.) | % do total |
|--------|:----------------:|:----------:|
| 5 (Prod. Rápida) | 88.079 | 18,7% |
| 0 (Normal) | 72.148 | 15,3% |
| 2 (DHSV) | 4.767 | 1,0% |

Desbalanceamento de ~18:1 entre a maior e menor classe.

### 8.2 Abordagem 2 (17 classes) — desbalanceamento severo

| Classe | Janelas | % do total |
|--------|:-------:|:----------:|
| 0 (Normal) | 102.040 | 22,7% |
| 5 (Prod. Rápida Ativo) | 70.018 | 15,6% |
| 7 (PCK Incrust. Ativo) | 914 | 0,2% |
| 102 (DHSV Trans.) | 937 | 0,2% |

Desbalanceamento de ~112:1 entre a maior e menor classe.

### 8.3 Estratégia adotada: class_weight='balanced'

O scikit-learn calcula automaticamente pesos inversamente proporcionais à frequência de cada classe:

```
peso_classe_c = n_total_amostras / (n_classes × n_amostras_classe_c)
```

Isso faz com que erros em classes raras "pesem" mais durante o treinamento. É a estratégia mais simples e eficaz para árvores de decisão.

### 8.4 Por que NÃO usar SMOTE?

SMOTE (Synthetic Minority Over-sampling Technique) cria amostras sintéticas interpolando features de amostras reais. Para dados tabulares genéricos, funciona bem. Para sensores físicos de poços de petróleo, **não é adequado**:

- Interpolar features (média, desvio padrão, derivada) entre dois transientes distintos não produz um transiente fisicamente realista
- Um transiente sintético gerado por SMOTE pode ter combinações de features impossíveis na realidade
- `class_weight='balanced'` é mais seguro e igualmente eficaz para modelos baseados em árvores

### 8.5 Métrica principal: F1-macro

Em vez de usar acurácia (que ignora classes raras), a métrica principal é o **F1-macro**:

```
F1-macro = média simples do F1 de cada classe = (F1_c1 + F1_c2 + ... + F1_cN) / N
```

Uma classe com 100 janelas contribui **igualmente** para o F1-macro que uma classe com 100.000 janelas. Um modelo que ignora as classes raras terá F1-macro baixo mesmo com acurácia alta.

---

## 9. Busca de Hiperparâmetros

**Implementação:** `sklearn.model_selection.RandomizedSearchCV`
**Configuração completa:** N_ITER=20, N_SPLITS=5, scoring='f1_macro'

### 9.1 RandomizedSearchCV vs GridSearchCV

GridSearchCV testa **todas** as combinações do grid. Com o grid abaixo e 5 folds, seriam 3×4×3×2=72 combinações × 5 folds = 360 treinamentos. RandomizedSearchCV sorteia N_ITER=20 combinações aleatórias — 100 treinamentos em vez de 360, com desempenho comparável.

### 9.2 Grid de busca — Random Forest

| Parâmetro | Valores testados | Descrição |
|-----------|:----------------:|-----------|
| n_estimators | [100, 200, 300] | Número de árvores |
| max_depth | [None, 10, 20, 30] | Profundidade máxima (None = sem limite) |
| min_samples_leaf | [1, 2, 4] | Mínimo de amostras por folha |
| max_features | ['sqrt', 'log2'] | Features consideradas em cada split |

### 9.3 Validação dentro da busca

A própria `RandomizedSearchCV` usa `GroupKFold` internamente — garantindo que a busca de hiperparâmetros também não vaza dados entre poços. O `best_score_` reportado é a média do F1-macro nos 5 folds **para cada combinação de hiperparâmetros**.

### 9.4 Paralelismo

`n_jobs=6` no `RandomForestClassifier` (paralelismo nas árvores) e `n_jobs=1` no `RandomizedSearchCV` (para evitar conflito de recursos com o paralelismo interno do RF).

---

## 10. Modelos Treinados

### 10.1 Random Forest

**Justificativa:** Método ensemble baseado em árvores de decisão. Naturalmente robusto a overfitting (por bagging), não requer normalização das features e fornece importância de features diretamente. É a referência padrão de mercado para dados tabulares.

**Treinamento:** `scripts/train_random_forest.py` (abordagem 1) e `scripts/train_rf_window_class.py` (abordagem 2)

### 10.2 XGBoost

**Justificativa:** Gradient Boosting — aprende sequencialmente minimizando os erros do modelo anterior. Geralmente supera o Random Forest em dados tabulares estruturados. Mais sensível a hiperparâmetros, mas com maior capacidade de aprendizado.

**Diferença técnica em relação ao RF:** XGBoost não suporta `class_weight` diretamente. Os pesos de classe são calculados via `compute_sample_weight('balanced', y)` e passados como `sample_weight` ao `fit()`. Além disso, os labels 101–109 precisam ser remapeados para o intervalo contíguo [0, 16] via `LabelEncoder` antes do treinamento (exigência da API `multi:softmax`).

**Treinamento:** `scripts/train_xgboost_window_class.py` (abordagem 2)

### 10.3 MLP — Multilayer Perceptron (a treinar)

**Justificativa:** Rede neural densa (totalmente conectada). Captura relações não-lineares complexas entre as 88 features. Implementado via `sklearn.neural_network.MLPClassifier` para comparabilidade direta com RF e XGBoost.

**Diferença dos modelos baseados em árvores:** requer normalização (`StandardScaler`) porque gradiente descendente é sensível à escala das features.

---

## 11. Avaliação e Métricas

### 11.1 Predições out-of-fold via cross_val_predict

Para gerar uma matriz de confusão honesta, é usado `cross_val_predict` com o mesmo `GroupKFold`:

```python
y_pred = cross_val_predict(model, X, y, cv=gkf, groups=groups)
```

Cada janela é predita por um modelo que **nunca a viu durante o treinamento**. Isso garante que a matriz de confusão reflete desempenho real em dados novos, não desempenho no treino.

### 11.2 Métricas reportadas

| Métrica | Fórmula | Uso |
|---------|---------|-----|
| F1-macro | Média simples do F1 por classe | Métrica principal — justo com classes raras |
| F1-weighted | Média ponderada pelo suporte | Complementar — reflete desempenho na massa de dados |
| Accuracy | Acertos / Total | Intuitivo, mas enganoso com desbalanceamento |
| Precision por classe | VP / (VP + FP) | Qualidade das predições positivas |
| Recall por classe | VP / (VP + FN) | Capacidade de encontrar os positivos reais |

### 11.3 Matriz de confusão normalizada

A matriz de confusão é normalizada por linha (pela classe real), mostrando a **fração de cada classe real que foi classificada como cada classe predita**. Isso é equivalente ao recall de cada classe e facilita identificar confusões sistemáticas.

---

## 12. Interpretabilidade — XAI (Explainable Artificial Intelligence)

**Implementação:** `notebooks/06_interpretacao.ipynb`
**Modelos analisados:** RF e XGBoost — Abordagem 2 (Estado Operacional, 17 classes)

Um modelo com F1-macro de 0,88 é útil, mas para um trabalho de engenharia é fundamental responder: **o que o modelo aprendeu?** As features mais importantes fazem sentido físico? O modelo aprendeu os mecanismos reais das falhas ou apenas padrões espúrios dos dados de treinamento?

### 12.1 Feature Importance MDI (Mean Decrease in Impurity)

A primeira técnica é a importância de features nativa dos modelos baseados em árvores, calculada durante o treinamento:

```python
importances = model.feature_importances_   # vetor de 88 valores, soma = 1.0
```

O MDI mede, para cada feature, o quanto ela reduziu a impureza (Gini) ao longo de todas as árvores e todos os nós onde foi usada. Features com MDI alto são usadas com frequência e produzem divisões claras.

**Limitação do MDI:** tende a superestimar a importância de features contínuas com muitos valores únicos e não captura interações entre features. É usado como baseline rápido antes do SHAP.

### 12.2 SHAP — SHapley Additive exPlanations

SHAP é a técnica padrão-ouro de XAI para modelos tabulares. Baseado na teoria dos jogos cooperativos (valores de Shapley), atribui a cada feature sua contribuição **marginal e justa** para cada predição individual.

```
SHAP value da feature j para a amostra i =
  contribuição média de j quando adicionada a cada subconjunto possível das outras features
```

**Propriedades garantidas pelo SHAP:**
- **Consistência:** se uma feature se torna mais importante no modelo, seu SHAP value aumenta
- **Aditividade:** a soma dos SHAP values de todas as features é igual à diferença entre a predição do modelo e a predição média (baseline)
- **Ausência:** features não utilizadas recebem SHAP = 0

Para classificação multiclasse, o SHAP retorna uma matriz de dimensões `(n_amostras, n_features, n_classes)` — um conjunto de valores por classe para cada amostra.

### 12.3 TreeExplainer — algoritmo exato para árvores

Em vez do `KernelExplainer` genérico (que aproxima SHAP via Monte Carlo e é lento), usamos o `TreeExplainer`:

```python
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_sample)   # exato, não aproximado
```

O `TreeExplainer` percorre as estruturas de árvore diretamente e calcula os valores de Shapley de forma **exata e determinística** em tempo polinomial.

**Limitação prática:** o RF treinado possui `max_depth=None` com profundidade média de **41,8 nós** e 3,4 milhões de nós totais. A complexidade do TreeSHAP é O(n_amostras × n_árvores × profundidade²), tornando inviável computar SHAP para o dataset completo.

### 12.4 Estratégia de amostragem balanceada

O dataset completo tem 449.397 janelas — computar SHAP em tudo levaria mais de 10 horas para o RF. A solução é uma **amostragem balanceada por classe**:

```python
N_SHAP_PER_CLASS = 50
n_per_class = min(N_SHAP_PER_CLASS, min_class_size)  # limitado pela menor classe
idx_balanced = np.concatenate([
    rng.choice(np.where(y == c)[0], n_per_class, replace=False)
    for c in np.unique(y)
])
# Total: 17 classes × 50 amostras = 850 janelas
```

A amostragem **balanceada** (igual número por classe) é preferível à estratificada proporcional por dois motivos:
1. A classe 7 tem apenas 914 janelas (0,2% do dataset) — na amostragem proporcional, teria apenas ~10 amostras de 5.000, insuficientes para estimar SHAP values estáveis para ela
2. Para comparar a importância das features *entre classes*, é preciso que cada classe tenha representação equivalente — caso contrário, classes majoritárias dominariam o cálculo global

Com 50 amostras por classe, os padrões de importância convergem de forma estável e a computação leva ~25 minutos para o RF e ~3 minutos para o XGBoost.

### 12.5 Resultados — Importância Global e por Sensor

#### Features mais importantes (RF, barplot mean|SHAP|)

O ranking das 20 features mais importantes pelo RF, medido como `mean|SHAP|` médio entre as 17 classes:

| Posição | Feature | Sensor | Estatística |
|:-------:|---------|--------|-------------|
| 1 | `T-TPT_std` | T-TPT | Desvio padrão |
| 2 | `T-TPT_iqr` | T-TPT | Amplitude interquartil |
| 3 | `P-PDG_max` | P-PDG | Máximo |
| 4 | `P-MON-CKP_max` | P-MON-CKP | Máximo |
| 5 | `T-JUS-CKP_max` | T-JUS-CKP | Máximo |
| 6 | `P-PDG_min` | P-PDG | Mínimo |
| 7 | `T-TPT_diff2_std` | T-TPT | Desvio padrão da 2ª derivada |
| 8 | `P-PDG_mean` | P-PDG | Média |
| 9 | `T-TPT_diff1_std` | T-TPT | Desvio padrão da 1ª derivada |
| 10 | `P-PDG_median` | P-PDG | Mediana |

As features de **variabilidade e taxa de variação** da temperatura no topo (`T-TPT_std`, `T-TPT_iqr`, `T-TPT_diff1_std`, `T-TPT_diff2_std`) dominam o ranking. Isso indica que os estados operacionais se distinguem principalmente pela *dinâmica* da temperatura no topo da coluna, não pelo nível absoluto.

#### Importância por sensor (RF e XGBoost)

| Posição | RF | XGBoost |
|:-------:|----|---------|
| 1 | **T-TPT** | **T-TPT** |
| 2 | **P-PDG** | **P-TPT** |
| 3 | P-TPT | P-PDG |
| 4 | T-JUS-CKP | P-MON-CKP |
| 5 | P-MON-CKP | T-JUS-CKP |
| 6 | QGL | QGL |
| 7 | P-JUS-CKGL | P-JUS-CKGL |
| 8 | T-PDG | T-PDG |

O sensor **T-TPT** (temperatura no topo da árvore de natal / TPT) é o mais informativo em ambos os modelos. O sensor **T-PDG** (temperatura no fundo do poço) é o menos relevante — a temperatura de fundo varia pouco entre estados operacionais, pois está próxima à rocha reservatório. O ranking é praticamente idêntico entre RF e XGBoost, validando a robustez dos achados.

### 12.6 Análise de Classes Desbalanceadas — Classes 7 e 102

#### Classe 7 — PCK Incrustação Ativo (RF F1=0,009 | XGBoost F1=0,571)

A classe 7 é a mais crítica do dataset do ponto de vista de desbalanceamento: **914 janelas em 449.397 (0,2%)**. A análise SHAP revela por que os dois modelos respondem de forma radicalmente diferente.

**RF — colapso completo (F1=0,009):**
Os SHAP values do RF para a classe 7 estão confinados no intervalo **−0,04 a +0,10**, praticamente zero para todas as features. O modelo simplesmente não prediz essa classe. O mecanismo é estrutural: o RF constrói todas as 200 árvores em paralelo, via *bagging* (amostras bootstrap com reposição). Em cada bootstrap, a classe 7 representa ~0,2% das amostras. Mesmo com `class_weight='balanced'`, o peso não compensa a ausência de co-ocorrências com outras features nos galhos mais profundos das árvores — a classe rara desaparece do processo de divisão.

**XGBoost — detecção parcial (F1=0,571):**
Os SHAP values do XGBoost para a classe 7 têm escala **−4 a +3**, revelando features discriminativas reais. As mais importantes são:

| Feature | Direção | Interpretação física |
|---------|---------|----------------------|
| `T-JUS-CKP_max` | Alto → classe 7 | Temperatura máxima a jusante da válvula choke elevada |
| `P-MON-CKP_mean` | Alto → classe 7 | Pressão média no manifold elevada |
| `T-TPT_min` | Baixo → classe 7 | Temperatura mínima no topo reduzida |
| `P-MON-CKP_min` | Alto → classe 7 | Pressão mínima no manifold sustentada |

**Interpretação física:** a incrustação no PCK (válvula choke de produção) é o acúmulo progressivo de depósitos minerais (escala) que restringe o orifício da válvula. A restrição altera o gradiente de pressão ao longo da válvula e modifica o efeito Joule-Thomson (resfriamento por expansão). Como consequência: (a) a pressão no manifold (P-MON-CKP) cresce porque o fluxo está represado a montante da restrição, e (b) a temperatura a jusante (T-JUS-CKP) sobe porque a expansão de gás — que normalmente resfria o fluido — é reduzida pela restrição mecânica. Esses dois padrões combinados são a "assinatura digital" da incrustação no PCK, e o XGBoost, via boosting sequencial, aprendeu a identificá-la.

**Por que o boosting supera o bagging aqui:** no boosting, cada nova árvore é treinada especificamente sobre os **erros residuais** das árvores anteriores. Quando as 914 janelas da classe 7 são sistematicamente mal classificadas nas primeiras iterações, as iterações seguintes recebem gradientes elevados para essas amostras — forçando o modelo a encontrar features discriminativas mesmo numa minoria extrema.

---

#### Classe 102 — DHSV Transiente (RF F1=0,786 | XGBoost F1=0,829)

A classe 102 representa o **período de transição** do fechamento da DHSV (Downhole Safety Valve — válvula de segurança de subsuperfície). Ao contrário da classe 7, ambos os modelos detectam essa classe, mas o desbalanceamento ainda limita o desempenho.

**Natureza do desbalanceamento:** os transientes da DHSV são por definição breves — o fechamento completo da válvula ocorre em dezenas de segundos. Isso gera muito menos janelas rotuladas como classe 102 (transiente) do que como classe 2 (DHSV ativa). O modelo precisa aprender a distinguir o transiente do estado ativo com base em características dinâmicas da série temporal.

**Features mais importantes — RF:**
- `T-JUS-CKP_max` (baixo → classe 102): quando a temperatura máxima a jusante do choke é **baixa**, o modelo sinaliza transiente — a válvula está começando a fechar e o fluxo está diminuindo
- `T-TPT_iqr` e `T-TPT_std`: variabilidade elevada da temperatura no topo — o fechamento progressivo causa oscilações enquanto o fluxo desacelera

**Features mais importantes — XGBoost:**
- `P-TPT_iqr`: alta variabilidade da pressão no topo — durante o fechamento, ondas de pressão se propagam pela coluna
- `T-TPT_diff1_std`: alto desvio padrão da taxa de variação da temperatura — a temperatura está mudando *rapidamente* e de forma *irregular*
- `T-TPT_skewness`: assimetria da distribuição de temperatura na janela — quando a janela captura tanto o período pré-fechamento quanto o pós-restrição, a distribuição de temperatura se torna assimétrica

**Interpretação física:** o transiente é um estado inerentemente dinâmico. As features que o distinguem do estado ativo completo (classe 2) não são os valores absolutos dos sensores, mas sim a *taxa de mudança* e a *forma da distribuição temporal* dentro da janela de 300 s. Por isso, features derivadas (`diff1_std`, `diff2_std`) e de forma (`skewness`, `iqr`) superam features estáticas (`mean`, `max`) para essa classe — e o XGBoost, que constrói árvores mais profundas e focadas em gradientes, captura essas nuances melhor que o RF.

### 12.7 Interpretação Física — Classe 8: Hidrato na Linha de Produção

A classe 8 é uma das mais bem classificadas por ambos os modelos (RF F1=0,855 | XGBoost F1=0,850), e a análise SHAP revela que os critérios aprendidos têm forte respaldo na termodinâmica de formação de hidratos.

#### Condições físicas de formação de hidratos

Hidratos de gás são compostos cristalinos que se formam quando **moléculas de gás (CH₄, C₂H₆, CO₂) ficam aprisionadas em gaiolas de moléculas de água** sob condições específicas de pressão e temperatura. A curva de equilíbrio hidrato separa as regiões de estabilidade:

- **Alta pressão + baixa temperatura** → região de formação de hidratos
- Em linhas de produção subseas: temperatura da água do mar no fundo pode ser 2–4°C, e a pressão hidrostática é elevada — condições propícias
- O ponto mais crítico é **a jusante da válvula choke (CKP)**: o efeito Joule-Thomson (expansão brusca do gás ao passar pelo orifício da válvula) resfria adicionalmente o fluido, podendo cruzar a curva de equilíbrio

#### Features aprendidas pelo XGBoost e respaldo físico

| Feature | SHAP | Interpretação física | Respaldo termodinâmico |
|---------|------|----------------------|------------------------|
| `T-JUS-CKP_min` | Baixo → classe 8 | Temperatura mínima a jusante do choke muito baixa | **Diretamente**: temperatura abaixo da curva de equilíbrio na zona mais fria da linha ✓ |
| `P-MON-CKP_max` | Alto → classe 8 | Pressão máxima no manifold elevada | **Alta pressão** é pré-condição para formação de hidratos ✓ |
| `P-TPT_iqr` | Alto → classe 8 | Alta variabilidade de pressão no topo | Tampão de hidrato causa bloqueio intermitente → pulsos de pressão ✓ |
| `P-PDG_mean` | Alto → classe 8 | Pressão média elevada no fundo do poço | Alta pressão de reservatório sustenta condições de formação ✓ |
| `T-TPT_mean` | Baixo → classe 8 | Temperatura média no topo abaixo do normal | Temperatura no topo dentro da janela de estabilidade de hidratos ✓ |
| `P-TPT_min` | Baixo → classe 8 | Queda de pressão mínima no topo | Reflexo da restrição a montante do tampão de hidrato ✓ |

#### Síntese da interpretação

O XGBoost aprendeu, sem supervisão física explícita, os dois vetores fundamentais que definem a formação de hidratos:

1. **Temperatura baixa a jusante do choke** (`T-JUS-CKP_min` baixo): o efeito Joule-Thomson cruza a curva de equilíbrio — a termodinâmica favorece a formação de hidratos exatamente nesse ponto da linha de produção.

2. **Alta pressão combinada com sinais de restrição de fluxo** (`P-MON-CKP_max` alto + `P-TPT_iqr` alto + `P-TPT_min` baixo): o hidrato em formação age como uma restrição progressiva — pressão acumula a montante enquanto oscilações de pressão no topo indicam o bloqueio intermitente característico do crescimento do tampão.

O acordo entre RF e XGBoost (F1=0,855 vs 0,850) e o alinhamento das features com a teoria termodinâmica constituem **validação cruzada independente**: o modelo aprendeu os mecanismos físicos reais, não padrões espúrios dos dados.

---

## 13. Artefatos Gerados

| Arquivo | Conteúdo |
|---------|---------|
| `data/processed/cleaned.parquet` | Séries temporais após limpeza e forward-fill |
| `data/processed/features.parquet` | Features com rotulagem por instância (10 classes) |
| `data/processed/features_window_class.parquet` | Features com rotulagem por estado (17 classes) |
| `results/models/random_forest.joblib` | RF treinado — abordagem 1 |
| `results/models/imputer.joblib` | Imputador de NaN — abordagem 1 |
| `results/models/rf_window_class.joblib` | RF treinado — abordagem 2 |
| `results/models/imputer_window_class.joblib` | Imputador de NaN — abordagem 2 |
| `results/metrics/random_forest_metrics.json` | Métricas detalhadas — RF abordagem 1 |
| `results/metrics/rf_window_class_metrics.json` | Métricas detalhadas — RF abordagem 2 |
| `results/figures/confusion_matrix_random_forest.png` | Matriz de confusão — RF abordagem 1 |
| `results/figures/confusion_matrix_rf_estado_operacional.png` | Matriz de confusão — RF abordagem 2 |
| `results/models/xgboost_window_class.joblib` | XGBoost treinado — abordagem 2 |
| `results/models/label_encoder_window_class.joblib` | LabelEncoder para labels 101–109 |
| `results/models/imputer_xgb_window_class.joblib` | Imputador de NaN — XGBoost abordagem 2 |
| `results/metrics/xgboost_window_class_metrics.json` | Métricas detalhadas — XGBoost abordagem 2 |
| `results/metrics/rf_nested_cv_results.json` | Resultados do Nested CV — RF abordagem 2 |
| `results/figures/confusion_matrix_xgboost_estado_operacional.png` | Matriz de confusão — XGBoost abordagem 2 |
