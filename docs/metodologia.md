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

### Tratamento de sensores constantes

Sensores com desvio padrão menor que 1×10⁻¹⁰ (travados em um valor fixo) são tratados como um caso especial:

```python
if std < 1e-10:
    df[sensor] = 0.0  # centraliza em zero, mantém variação = 0
```

Isso evita divisão por zero e garante que um sensor desligado (valor=0) e um sensor travado em 38 MPa produzam o mesmo resultado de features (ambos mostram variação zero, que é a informação relevante).

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

### 5.3 Tratamento de janelas quase vazias

Quando um sensor tem menos de 10 leituras válidas (não-NaN) dentro de uma janela de 300 s, por exemplo, um sensor offline por quase toda a janela — não há dados suficientes para calcular estatísticas confiáveis. Nesse caso, todas as 11 features daquele sensor recebem `NaN` para aquela janela.

Após a extração, a matriz X (janelas × features) pode conter NaN em posições onde sensores estavam ausentes. Esses NaN são preenchidos antes do treinamento por um `SimpleImputer(strategy='median')`, que substitui cada NaN pela mediana daquela feature calculada sobre todas as janelas.

Exemplo: se `P-TPT_mean` for NaN em uma janela (sensor ausente naquele período), o valor imputado é a mediana de `P-TPT_mean` calculada sobre todas as ~449.000 janelas do dataset. A lógica é conservadora: assume-se o valor mais típico disponível para aquela feature.

O imputer é salvo em `.joblib` junto com o modelo para garantir que a mesma transformação seja aplicada na inferência.

---

## 6. Estratégias de Rotulagem

**Arquivo:** `data/processed/features_window_class.parquet`
**Label:** `window_label` — a moda da coluna `class` dentro daquela janela

```python
valid_states = window_df['class'].dropna()
if valid_states.empty:
    continue  # janela 100% NaN descartada
window_label = int(valid_states.mode().iloc[0])
```

**17 classes encontradas no dataset:** 0 (Normal), 1–9 (Ativo por tipo), 101–109 (Transiente por tipo).
Classes 103 e 104 (transientes de Golfadas e Instabilidade de Fluxo) não aparecem no dataset — nessas falhas, o sinal vai de normal direto para o estado ativo sem transiente registrado.

**Janelas descartadas:** aquelas onde 100% dos timestamps têm `class=NaN` (período pré-evento puro). Essas janelas não têm estado definido e não contribuem para nenhuma das 17 classes.

---

## 7. Separação Treino/Teste — GroupKFold

**Implementação:** `sklearn.model_selection.GroupKFold(n_splits=5)`
**Chave de agrupamento:** `instance_id` (identificador único de cada poço)

### 7.1 O problema de data leakage em séries temporais

Se dividirmos as janelas aleatoriamente (KFold padrão), janelas do **mesmo poço** aparecem tanto no treino quanto no teste. O modelo memoriza aquele poço e parece ter bom desempenho, mas falha em poços novos (**Data Leakage**)

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

As métricas finais (F1-macro, accuracy) refletem o desempenho em **poços completamente novos** — nunca vistos pelo modelo durante o treinamento.

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

O problema: os hiperparâmetros foram escolhidos **justamente porque funcionaram bem nesses folds específicos**. Avaliar com os mesmos folds produz métricas ligeiramente otimistas — o modelo foi, indiretamente, ajustado para esse conjunto de dados.

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

**Resultados do Nested CV — Random Forest:**

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
- A Classe 7 (PCK Incrustação Ativo, F1≈0,009 no flat CV e ≈0,005 no nested CV) confirma que o problema dessa classe é estrutural — apenas 914 janelas (0,2% do dataset) 
- O desvio padrão de ±0,0202 entre os folds externos reflete a variabilidade natural entre grupos de poços: alguns grupos contêm poços mais difíceis de generalizar (Fold 2: F1=0,8415) e outros mais representativos (Fold 5: F1=0,8996)
- **Conclusão:** A metodologia de flat CV com GroupKFold(5) é essencialmente honesta para este dataset. As comparações entre modelos são válidas, pois todos foram avaliados sob a mesma metodologia e o viés afeta todos igualmente.

---

## 8. Tratamento do Desbalanceamento de Classes

### 17 classes — desbalanceamento severo

| Classe | Janelas | % do total |
|--------|:-------:|:----------:|
| 0 (Normal) | 102.040 | 22,7% |
| 5 (Prod. Rápida Ativo) | 70.018 | 15,6% |
| 7 (PCK Incrust. Ativo) | 914 | 0,2% |
| 102 (DHSV Trans.) | 937 | 0,2% |

Desbalanceamento de ~112:1 entre a maior e menor classe.

### 8.2 Estratégia adotada: class_weight='balanced'

O scikit-learn calcula automaticamente pesos inversamente proporcionais à frequência de cada classe:

```
peso_classe_c = n_total_amostras / (n_classes × n_amostras_classe_c)
```

Isso faz com que erros em classes raras pesem mais durante o treinamento. É a estratégia mais simples e eficaz para árvores de decisão.

### 8.3 Métrica principal: F1-macro

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

**Justificativa:** Método ensemble baseado em árvores de decisão. Naturalmente robusto a overfitting (por bagging), não requer normalização das features e fornece importância de features diretamente. É a referência padrão da literatura para dados tabulares.

**Treinamento:**  `scripts/train_rf_window_class.py` 

### 10.2 XGBoost

**Justificativa:** Gradient Boosting — aprende sequencialmente minimizando os erros do modelo anterior. Geralmente supera o Random Forest em dados tabulares estruturados. Mais sensível a hiperparâmetros, mas com maior capacidade de aprendizado.

**Diferença técnica em relação ao RF:** XGBoost não suporta `class_weight` diretamente. Os pesos de classe são calculados via `compute_sample_weight('balanced', y)` e passados como `sample_weight` ao `fit()`. Além disso, os labels 101–109 precisam ser remapeados para o intervalo contíguo [0, 16] via `LabelEncoder` antes do treinamento (exigência da API `multi:softmax`).

**Treinamento:** `scripts/train_xgboost_window_class.py` 

### 10.3 CNN-1D — Rede Convolucional 1D (FCN)

**Referência de arquitetura:** Wang et al. (2017) — *Time Series Classification from Scratch with Deep Neural Networks: A Strong Baseline*

**Justificativa:** Os modelos RF e XGBoost operam sobre **88 features artesanais** extraídas manualmente das janelas de sensores. A CNN-1D recebe diretamente a **série temporal bruta** como entrada e aprende suas próprias representações durante o treinamento, sem extração manual de features. Isso cria uma comparação metodológica fundamental: features manuais versus representações aprendidas automaticamente.

A arquitetura escolhida é a **FCN (Fully Convolutional Network)**, baseline neural estado da arte para classificação de séries temporais em benchmarks UCR/UEA (Fawaz et al., 2019). Vantagens em relação a LSTM e Transformer:
- Sem recorrência → treino paralelizável e mais rápido
- `GlobalAveragePooling1D` elimina hiperparâmetro de tamanho de saída
- Desempenho competitivo com menor custo computacional

**Implementação:** `scripts/train_cnn1d_v2.py`

#### Arquitetura FCN

```
Input: (batch, 300, 8)   ← janela bruta após Z-score + filtro Gaussiano (sigma=2.0)
│
├─ Conv1D(128 filtros, kernel=8, padding='same', use_bias=False)
│  BatchNormalization → ReLU
│
├─ Conv1D(256 filtros, kernel=5, padding='same', use_bias=False)
│  BatchNormalization → ReLU
│
├─ Conv1D(128 filtros, kernel=3, padding='same', use_bias=False)
│  BatchNormalization → ReLU
│
├─ GlobalAveragePooling1D()   → (batch, 128)
│
└─ Dense(17, activation='softmax')
```

Os kernels decrescentes (8→5→3) capturam padrões em múltiplas escalas temporais: o kernel largo detecta tendências de médio prazo, e os menores detectam transições locais. O `use_bias=False` nas camadas convolucionais é adequado porque o `BatchNormalization` subsequente já possui seu próprio parâmetro de deslocamento (`beta`), tornando o bias redundante.

#### Pré-processamento para a CNN

O pré-processamento é o mesmo usado no RF/XGBoost — Z-score por instância e filtro Gaussiano (sigma=2.0) — mas aplicado à **série temporal completa** antes da extração de janelas, não às janelas individuais. A normalização é feita em numpy puro e o resultado é mantido em memória como cache durante todo o treinamento (funções `preprocess_instance` e `extract_windows` no script).

#### Pipeline de Dados — `tf.data`

Com ~457.000 janelas × 300 timesteps × 8 sensores × 4 bytes ≈ 4,4 GB, materializar todas as janelas de treino em RAM ao mesmo tempo é inviável. A solução é um pipeline gerador via `tf.data`:

```
from_generator → shuffle(buffer=20.000) → repeat() → map(sample_weight) → batch(256) → prefetch(AUTOTUNE)
```

| Etapa | Função |
|-------|--------|
| `from_generator` | Produz uma janela por vez — nunca aloca o dataset completo em RAM |
| `shuffle(20.000)` | Mantém 20.000 janelas em buffer (~190 MB) para aleatorização eficiente |
| `repeat()` | Faz o gerador recomeçar antes que o buffer esvazie, mantendo-o sempre aquecido |
| `map(tf.gather)` | Adiciona o peso de classe a cada janela via indexação vetorizada |
| `batch(256)` | Agrupa 256 janelas em tensor `(256, 300, 8)` para envio à GPU |
| `prefetch(AUTOTUNE)` | CPU prepara o próximo lote enquanto GPU processa o atual |

O `.repeat()` é crítico para evitar o padrão de épocas alternadamente lentas/rápidas: sem ele, o buffer esvazia ao fim de cada época e leva ~325 s para ser repreenchido. Com `.repeat()`, o gerador recomeça antes do esvaziamento e todas as épocas levam ~40 s.

#### Balanceamento de Classes

O argumento `class_weight` do `model.fit()` do Keras não é compatível com `tf.data.Dataset`. A alternativa é incluir o peso como terceiro elemento de cada amostra (`sample_weight`), adicionado via `tf.gather` no estágio `.map()` do pipeline:

```python
cw_tensor = tf.constant([cw_dict.get(i, 1.0) for i in range(N_CLASSES)], dtype=tf.float32)
ds = ds.map(lambda x, y: (x, y, tf.gather(cw_tensor, tf.cast(y, tf.int32))))
```

Os pesos são calculados por `compute_class_weight('balanced', ...)` sobre as instâncias de treino de cada fold, da mesma forma que no RF e XGBoost.

#### Validação e Treino

A separação treino/teste usa o mesmo `GroupKFold(5)` por `instance_id`. Dentro do conjunto de treino de cada fold, **10% das instâncias** são separadas como validação (`val_iids`) e materializadas em numpy (`X_val`, `y_val`) para uso direto nos callbacks.

| Hiperparâmetro | Valor | Justificativa |
|----------------|-------|---------------|
| Optimizer | Adam (lr=1e-3) | Padrão para redes neurais — adaptativo por parâmetro |
| Loss | sparse_categorical_crossentropy | Aceita labels inteiros (sem conversão one-hot) |
| Épocas máx. | 100 | EarlyStopping interrompe antes se val_f1_macro estagnar |
| EarlyStopping | patience=10, restore_best_weights=True | Restaura os pesos da melhor época |
| ReduceLROnPlateau | patience=5, factor=0.5, min_lr=1e-5 | Reduz lr à metade antes do Early Stopping |
| Batch size | 256 | Equilíbrio entre velocidade de GPU e frequência de atualização |

O `steps_per_epoch` é calculado como `n_janelas_treino // 256`, informando ao Keras quando encerrar cada época em um dataset infinito (`.repeat()`).

**Hardware utilizado:** GPU RTX 2060 (6 GB VRAM) via TensorFlow 2.x com XLA habilitado. Tempo estimado de treinamento: ~1 hora para os 5 folds completos.

### 10.4 CNN-LSTM

**Justificativa:** A CNN-1D FCN agrega todos os timesteps da janela com peso igual no `GlobalAveragePooling1D`, perdendo a ordem temporal. A LSTM mantém um estado oculto que evolui ao longo da sequência, capturando dependências temporais de longo alcance — especialmente relevantes para estados transientes, cujo sinal discriminativo é a trajetória de mudança ao longo do tempo, não apenas a distribuição estatística.

Para manter eficiência computacional, dois blocos Conv1D+MaxPool reduzem a sequência de entrada de 300 para 75 timesteps antes da LSTM. Isso torna o custo da camada recorrente 4× menor do que sobre a sequência completa, sem perda de informação de tendência: a CNN extrai padrões locais e o MaxPool comprime a representação.

**Treinamento:** `scripts/train_lstm.py`

#### Arquitetura CNN-LSTM

```
Input: (batch, 300, 8)   ← janela bruta após Z-score + filtro Gaussiano (sigma=2.0)
│
├─ Conv1D(64 filtros, kernel=5, padding='same', use_bias=False)
│  BatchNormalization → ReLU → MaxPooling1D(pool_size=2)   → (150, 64)
│
├─ Conv1D(128 filtros, kernel=3, padding='same', use_bias=False)
│  BatchNormalization → ReLU → MaxPooling1D(pool_size=2)   → (75, 128)
│
├─ LSTM(64 unidades)   ← processa apenas 75 passos (4× mais rápido)
│  Dropout(0.3)        → (64,)
│
└─ Dense(17, activation='softmax')
```

Kernels decrescentes (5→3) capturam padrões em múltiplas escalas locais antes de comprimir a sequência. A LSTM recebe 75 vetores de 128 dimensões — cada vetor é a representação local aprendida pela CNN para aquele bloco de 4 amostras. O `Dropout(0.3)` pós-LSTM regulariza sem usar `recurrent_dropout`, compatível com a implementação CuDNN quando GPU está disponível.

#### Diferenças em relação à CNN-1D FCN

| Aspecto | CNN-1D FCN | CNN-LSTM |
|---------|-----------|----------|
| Memória temporal | Nenhuma (`GlobalAvgPool` agrega com peso igual) | Explícita (estado oculto da LSTM evolui no tempo) |
| Timesteps para camada final | 300 (após pools o GAP agrega tudo) | 75 (MaxPool reduz antes da LSTM) |
| Batch size | 256 | 512 |
| Saída principal | `results/metrics/cnn1d_metrics.json` | `results/metrics/lstm_metrics.json` |

#### Pipeline de Dados e Procedimento de Treinamento

Idênticos à CNN-1D: gerador `tf.data` com `shuffle(20.000)` + `.repeat()`, pesos de classe via `compute_class_weight('balanced')`, `GroupKFold(5)` por `instance_id`, 15% das instâncias de treino separadas para validação interna.

**Critério de parada — `MacroF1Callback`:** o `val_loss` é insuficiente para classes raras (a classe 7, com 0,2% das janelas, move a perda em apenas milésimos). Um callback customizado calcula o F1-macro sobre o conjunto de validação ao final de cada época e injeta `val_f1_macro` nos logs do Keras; `EarlyStopping` e `ReduceLROnPlateau` monitoram exclusivamente essa métrica.

| Hiperparâmetro | Valor | Justificativa |
|----------------|-------|---------------|
| Optimizer | Adam (lr=1e-3) | Padrão para redes neurais |
| Loss | sparse_categorical_crossentropy | Aceita labels inteiros |
| Épocas máx. | 100 | EarlyStopping interrompe antes |
| EarlyStopping | patience=15, mode='max', restore_best_weights=True | Mais tolerante que na FCN; classes raras precisam de mais épocas |
| ReduceLROnPlateau | patience=7, factor=0.5, min_lr=1e-5 | Reduz lr antes do Early Stopping |
| Batch size | 512 | Maior que FCN — beneficia LSTM em CPU (menos overhead por passo) |

O `steps_per_epoch` é calculado como `n_janelas_treino // 512`.

**Sobre a seleção de hiperparâmetros:** diferente do RF e do XGBoost, que passam por `RandomizedSearchCV` com 20 combinações × 5 folds, a CNN-LSTM utiliza uma arquitetura fixa escolhida por decisão de projeto. Os valores foram justificados pela literatura (Wang et al., 2017; padrões estabelecidos para CNN+LSTM em séries temporais industriais) e pelas restrições computacionais do ambiente de treinamento. Uma busca automatizada com `GroupKFold(5)` seria inviável neste contexto: cada configuração candidata requereria 5 treinamentos de ~1 h cada, tornando 20 combinações equivalentes a ~100 h de processamento.

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

SHAP é a técnica padrão de XAI para modelos tabulares. Esse método atribui a cada feature sua contribuição para cada predição individual.

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

---

## 13. Estudo do Impacto da Filtragem de Sinal

### 13.1 Motivação

O pipeline padrão aplica um filtro Gaussiano (`sigma=2.0`) sobre cada sensor antes de
calcular as 11 estatísticas por janela. Esse filtro remove variações de alta frequência
(ruído elétrico, vibrações mecânicas), mas também pode atenuar transientes rápidos que
são assinaturas de certas falhas.

Para quantificar o impacto dessa decisão de pré-processamento, foi conduzido um
estudo de comaparação: o XGBoost foi retreinado com os dados brutos (sem nenhuma filtragem)
e os resultados foram comparados diretamente com o modelo treinado com filtro Gaussiano.
Essa comparação isola o efeito do filtro, mantendo idênticos o modelo, os hiperparâmetros
(mesma busca RandomizedSearchCV), a separação GroupKFold e o balanceamento de classes.

### 13.2 Diferença Metodológica

| Configuração | Parâmetro `smooth_filter` | Pré-processamento do sinal |
|---|---|---|
| XGBoost Gaussiano | `'gaussian'` | `gaussian_filter1d(sigma=2.0)` antes de cada janela |
| XGBoost Sem Filtro | `'none'` | Sinal z-scored diretamente para extração de features |

O parâmetro foi implementado em `src/feature_engineering.py` e propagado por todo o
pipeline (`extract_features_from_instance`, `run_pipeline_from_cleaned`). O dataset
resultante foi salvo em `data/processed/features_nofilter_window_class.parquet` (449.397
janelas, 88 features — idêntico ao dataset filtrado em estrutura).

**Melhores hiperparâmetros encontrados (sem filtro):**
`n_estimators=500, max_depth=4, learning_rate=0.1, subsample=0.8,
colsample_bytree=1.0, min_child_weight=3`

### 13.3 Resultados Comparativos

**Métricas globais:**

| Métrica | XGBoost Gaussiano | XGBoost Sem Filtro | Delta |
|---|---|---|---|
| F1-macro | **0.9082** | 0.9065 | −0.0017 |
| F1-weighted | **0.9340** | 0.9339 | −0.0001 |
| Accuracy | 0.9364 | **0.9345** | −0.0019 |
| F1-macro (CV) | 0.8866 | **0.8890** | +0.0024 |

O impacto global é mínimo (−0.0017 no F1-macro), confirmando que o filtro Gaussiano
não é determinante para o desempenho médio do modelo. A análise por classe, porém,
revela comportamentos opostos dependendo da natureza de cada falha.

**F1 por classe:**

| Classe | Nome | Gaussiano | Sem Filtro | Delta |
|---|---|---|---|---|
| 0 | Normal | 0.9145 | 0.9091 | −0.005 |
| 1 | BSW (Ativo) | 0.9389 | **0.9407** | +0.002 |
| 2 | DHSV (Ativo) | 0.9729 | **0.9733** | +0.000 |
| 3 | Golfadas (Ativo) | 0.9674 | **0.9757** | **+0.008** |
| 4 | Inst. Fluxo (Ativo) | 0.8988 | **0.9063** | +0.007 |
| 5 | Prod. Rápida (Ativo) | 0.9819 | **0.9853** | +0.003 |
| 6 | PCK Restrição (Ativo) | 0.9901 | **0.9917** | +0.002 |
| 7 | PCK Incrustação (Ativo) | **0.5712** | 0.5260 | **−0.045** |
| 8 | Hidrato Produção (Ativo) | 0.8497 | **0.8508** | +0.001 |
| 9 | Hidrato Serviço (Ativo) | 0.9895 | **0.9899** | +0.000 |
| 101 | BSW (Trans.) | **0.9289** | 0.9281 | −0.001 |
| 102 | DHSV (Trans.) | 0.8285 | **0.8646** | **+0.036** |
| 105 | Prod. Rápida (Trans.) | 0.8899 | **0.8955** | +0.006 |
| 106 | PCK Restrição (Trans.) | 0.9775 | **0.9794** | +0.002 |
| 107 | PCK Incrustação (Trans.) | 0.9047 | 0.9074 | +0.003 |
| 108 | Hidrato Produção (Trans.) | **0.8724** | 0.8278 | **−0.045** |
| 109 | Hidrato Serviço (Trans.) | **0.9628** | 0.9586 | −0.004 |

### 13.4 Análise por Tipo de Falha

Os resultados revelam um padrão físico claro: o impacto do filtro depende da
**dinâmica temporal** da falha.

**Falhas de dinâmica rápida — sem filtro é melhor:**

- **Classe 102 — DHSV Transiente (+0.036):** O fechamento espúrio da válvula DHSV
  produz uma queda abrupta de pressão em poucos segundos. O filtro Gaussiano atenua
  essa transição rápida, reduzindo as features `diff1_std` e `diff2_std` que capturam
  justamente essa taxa de variação. Sem filtro, essas features retêm o sinal completo
  da transição, tornando a classe mais fácil de distinguir.

- **Classe 3 — Golfadas Severas (+0.008):** O padrão de golfadas consiste em picos
  periódicos de pressão. O sensor QGL (vazão de gás lift) mostrou visualmente a maior
  diferença entre os filtros — o filtro Gaussiano reduz a amplitude dos picos, enquanto
  sem filtro `max_zscore` e `std` capturam os picos com maior fidelidade.

- **Classe 4 — Instabilidade de Fluxo (+0.007):** Similar às golfadas, a instabilidade
  gera oscilações que o filtro parcialmente suaviza.

**Falhas de dinâmica lenta — filtro Gaussiano é melhor:**

- **Classe 7 — PCK Incrustação (−0.045):** A incrustação no choke de produção é um
  processo gradual que ocorre ao longo de horas ou dias. O modelo precisa detectar uma
  tendência sutil de aumento de restrição. Sem filtro, o ruído de alta frequência do
  sensor (visível em P-ANULAR no gráfico de comparação) eleva artificialmente `std` e
  `diff1_std`, mascarando a tendência lenta. O filtro Gaussiano remove esse ruído e
  torna a tendência mais visível para o modelo.

- **Classe 108 — Hidrato Produção Transiente (−0.045):** O transiente de hidrato é
  caracterizado por uma queda gradual de temperatura (efeito Joule-Thomson no choke)
  e aumento progressivo de pressão diferencial — dinâmica de horas. O ruído nos
  sensores P-ANULAR e P-MON-CKP, sem filtro, introduz variabilidade espúria nas
  features que confunde o modelo durante essa fase de acúmulo lento.

**Sensores mais afetados pela escolha do filtro:**

Com base na comparação visual dos filtros realizada na instância WELL-00028
(Classe 8 — Hidrato na Linha de Produção):

| Sensor | Comportamento sem filtro | Impacto nas features |
|---|---|---|
| QGL | Alta variabilidade preservada | `std`, `max_zscore` maiores — beneficia classes com oscilações rápidas |
| P-ANULAR | Picos periódicos preservados | `max_zscore`, `iqr` maiores — pode ser ruído ou sinal real |
| T-MON-CKP | Sinal suave — sem diferença | Nenhum impacto prático |
| P-JUS-CKGL | Sinal suave — sem diferença | Nenhum impacto prático |

### 13.5 Análise SHAP — Mudança de Importância entre Modelos

Os gráficos SHAP gerados para cada classe (salvos em `results/figures/shap/nofilter/`)
revelam duas padrões principais:

**Classes onde o filtro não importa** (DHSV Ativo, Hidrato Serviço, PCK Restrição):
Os bee swarm plots do modelo Gaussiano e do modelo Sem Filtro são praticamente
idênticos — as mesmas features dominam, com magnitudes similares. Isso ocorre porque
esses eventos são detectáveis por tendências de larga escala que ambos os
pré-processamentos preservam igualmente.

**Classes onde o filtro muda as features dominantes** (DHSV Transiente, Golfadas,
PCK Incrustação): Os SHAP values mostram redistribuição de importância entre features
de `diff1_std`/`diff2_std` (derivadas, sensíveis ao ruído) e `mean`/`std` (tendência
global). Sem filtro, as features de derivada ganham importância nas classes de dinâmica
rápida (benefício) mas perdem discriminabilidade nas classes de dinâmica lenta (custo).

### 13.6 Conclusão (Gaussiano vs Sem Filtro)

O filtro Gaussiano com `sigma=2.0` é uma escolha robusta para o problema como um todo:
minimiza a degradação global (apenas −0.0017 no F1-macro) enquanto protege especificamente
as duas classes mais raras e de detecção mais difícil — PCK Incrustação (classe 7) e
Hidrato Produção Transiente (classe 108) — onde a perda sem filtro é de −0.045 em ambas.

Para um sistema de monitoramento em tempo real, onde a detecção de hidratos e incrustações
é crítica por razões de segurança e custo operacional, essa proteção justifica a escolha
do filtro Gaussiano como padrão no pipeline.

---

### 13.7 Filtro Estatístico Adaptativo (σ=0.5)

Para completar o ablation study, foi testado um terceiro regime de pré-processamento: o
**filtro estatístico adaptativo** (implementado em `_apply_statistical_filter`), parametrizado
com `sigma=0.5` (threshold ≈ 1.41 z-score).

O filtro opera de forma causal (forward pass) seguido de um backward pass para eliminar
defasagem de fase, combinando o resultado via `filtfilt` adaptativo. O coeficiente de
mistura `alpha` é calculado por:

```
alpha = erf(|x_anterior - u| / (2√2 × sigma))
saida = (1 - alpha) * x_anterior + alpha * u
```

Com `sigma=0.5`, o filtro suaviza variações pequenas (ruído típico com |diff| < 0.032 z-score)
e preserva transientes grandes (|diff| > 1.41 z-score), diferentemente do Gaussiano, que
suaviza independentemente da magnitude da variação.

**Melhores hiperparâmetros encontrados (filtro estatístico):**
`n_estimators=500, max_depth=4, learning_rate=0.1, subsample=0.8,
colsample_bytree=1.0, min_child_weight=3`

(idênticos ao XGBoost Gaussiano — confirma que a busca de hiperparâmetros converge para
a mesma configuração independentemente do pré-processamento)

---

### 13.8 Comparação Tripla — Gaussiano vs Sem Filtro vs Estatístico

**Métricas globais:**

| Métrica | Gaussiano | Sem Filtro | Estatístico | Melhor |
|---|:---:|:---:|:---:|:---:|
| F1-macro | **0.9082** | 0.9065 | 0.9067 | Gaussiano |
| F1-weighted | 0.9361 | 0.9339 | **0.9401** | **Estatístico** |
| Accuracy | 0.9364 | 0.9345 | **0.9402** | **Estatístico** |

O filtro estatístico supera ambos os outros em F1-weighted e Accuracy — indicando que
é mais preciso para as classes frequentes (que dominam essas métricas). No F1-macro
(que trata todas as classes igualmente), o Gaussiano permanece ligeiramente superior.

**F1 por classe — comparação tripla:**

| Cl. | Estado | Gauss. | S/Filt. | Estat. | Estat.−Gauss. | Estat.−S/Filt. |
|:---:|--------|:------:|:-------:|:------:|:-------------:|:--------------:|
| 0 | Normal | 0.9145 | 0.9091 | **0.9168** | +0.002 | +0.008 |
| 1 | BSW (Ativo) | 0.9389 | **0.9407** | 0.9403 | +0.001 | −0.000 |
| 2 | DHSV (Ativo) | **0.9729** | 0.9733 | 0.9648 | −0.008 | −0.009 |
| 3 | Golfadas (Ativo) | 0.9674 | **0.9757** | 0.9588 | −0.009 | −0.017 |
| 4 | Inst. Fluxo (Ativo) | 0.8988 | **0.9063** | 0.8894 | −0.009 | −0.017 |
| 5 | Prod. Rápida (Ativo) | **0.9819** | 0.9853 | 0.9804 | −0.002 | −0.005 |
| 6 | PCK Restrição (Ativo) | 0.9901 | **0.9917** | 0.9911 | +0.001 | −0.001 |
| **7** | **PCK Incrust. (Ativo)** | 0.5712 | 0.5260 | **0.5761** | **+0.005** | **+0.050** |
| 8 | Hidrato Prod. (Ativo) | **0.8497** | 0.8508 | 0.8443 | −0.005 | −0.007 |
| 9 | Hidrato Serv. (Ativo) | **0.9895** | 0.9899 | 0.9879 | −0.002 | −0.002 |
| 101 | BSW (Trans.) | 0.9289 | 0.9281 | **0.9544** | **+0.026** | **+0.026** |
| **102** | **DHSV (Trans.)** | 0.8285 | **0.8646** | 0.7718 | **−0.057** | **−0.093** |
| 105 | Prod. Rápida (Trans.) | **0.8899** | 0.8955 | 0.8869 | −0.003 | −0.009 |
| 106 | PCK Restrição (Trans.) | 0.9775 | **0.9794** | 0.9786 | +0.001 | −0.001 |
| 107 | PCK Incrust. (Trans.) | 0.9047 | 0.9074 | **0.9210** | **+0.016** | **+0.014** |
| **108** | **Hidrato Prod. (Trans.)** | 0.8724 | 0.8278 | **0.8914** | **+0.019** | **+0.064** |
| 109 | Hidrato Serv. (Trans.) | **0.9628** | 0.9586 | 0.9605 | −0.002 | +0.002 |

**Padrão observado:**

O filtro estatístico apresenta comportamento misto que reflete seu mecanismo adaptativo:

- **Ganhos significativos:** classes 101 (+0.026), 107 (+0.016), 108 (+0.019) e classe 7 (+0.005) — transientes lentos e classes de detecção difícil se beneficiam da suavização adaptativa, que remove ruído sem atenuar tendências de larga escala.

- **Perda crítica — Classe 102 DHSV Transiente (−0.057 vs Gaussiano, −0.093 vs Sem Filtro):** O fechamento abrupto da válvula DHSV produz uma variação de pressão que, após z-scoring, frequentemente ultrapassa o threshold de 1.41 z-score — e portanto o filtro estatístico **preserva** esse transiente. Contudo, os resultados indicam que o mecanismo backward pass introduz um efeito de antecipação artificial (suavização "futura" retroativa) que confunde as features de derivada (`diff1_std`) para esse tipo específico de evento.

- **Classes de dinâmica rápida (3, 4):** O filtro estatístico é ligeiramente inferior ao sem-filtro porque o sigma=0.5 ainda suaviza variações no limiar do threshold.

**Figuras geradas:** `results/figures/shap/statistical_vs_nofilter/shap_classe{C}.png` (17 figuras)
e `shap_delta_f1_statistical_vs_nofilter.png` (resumo do delta F1 por classe).

---

### 13.9 Conclusão Final do Estudo Comparativo

| Filtro | F1-macro | F1-weighted | Melhor para |
|--------|:--------:|:-----------:|------------|
| Gaussiano (`sigma=2.0`) | **0.9082** | 0.9361 | Classes raras de detecção difícil (7, 108); consistência global |
| Sem Filtro | 0.9065 | 0.9339 | Falhas de dinâmica rápida (102 DHSV Trans., 3 Golfadas) |
| **Estatístico** (`sigma=0.5`) | 0.9067 | **0.9401** | F1-weighted e Accuracy; classes 101, 107, 108 |

Nenhum filtro domina em todas as classes simultaneamente — cada regime favorece um
subconjunto de falhas com características temporais distintas. Para o objetivo principal
do TCC (maximizar F1-macro com ênfase em classes raras), o **filtro Gaussiano permanece
a escolha padrão** por sua consistência. O filtro estatístico representa uma alternativa
viável quando se prioriza precisão nas classes frequentes (F1-weighted).

---

## 14. Artefatos Gerados

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
| `data/processed/features_nofilter_window_class.parquet` | Features sem filtro — ablation study |
| `results/models/xgboost_nofilter_window_class.joblib` | XGBoost treinado sem filtro |
| `results/models/imputer_nofilter_window_class.joblib` | Imputador de NaN — XGBoost sem filtro |
| `results/metrics/xgboost_nofilter_metrics.json` | Métricas detalhadas — XGBoost sem filtro |
| `results/figures/shap/nofilter/shap_nofilter_classeXXX.png` | SHAP por classe — Gaussiano vs Sem Filtro (17 figuras) |
| `data/processed/features_statistical_window_class.parquet` | Features com filtro estatístico (σ=0.5) — ablation study |
| `results/models/xgboost_statistical_window_class.joblib` | XGBoost treinado com filtro estatístico |
| `results/models/imputer_statistical_window_class.joblib` | Imputador de NaN — XGBoost filtro estatístico |
| `results/metrics/xgboost_statistical_metrics.json` | Métricas detalhadas — XGBoost filtro estatístico |
| `results/figures/shap/statistical_vs_nofilter/shap_classeXXX.png` | SHAP comparativo — Estatístico vs Sem Filtro (17 figuras) |
| `results/figures/shap/statistical_vs_nofilter/shap_delta_f1_statistical_vs_nofilter.png` | Delta F1 por classe — Estatístico vs Sem Filtro |
| `results/metrics/cnn1d_metrics.json` | Métricas detalhadas — CNN-1D (FCN), 5 folds OOF |
| `results/figures/confusion_matrix/confusion_matrix_cnn1d_estado_operacional.png` | Matriz de confusão — CNN-1D |
| `results/metrics/lstm_metrics.json` | Métricas detalhadas — CNN-LSTM, 5 folds OOF |
| `results/metrics/lstm_oof_true.npy` | Labels verdadeiros OOF — CNN-LSTM |
| `results/metrics/lstm_oof_pred.npy` | Predições OOF — CNN-LSTM |
| `results/figures/confusion_matrix/confusion_matrix_lstm_estado_operacional.png` | Matriz de confusão — CNN-LSTM |
