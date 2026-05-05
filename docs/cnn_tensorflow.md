# TensorFlow na CNN-1D — Guia de Implementação e Otimizações

Este documento explica cada funcionalidade do TensorFlow utilizada no script `train_cnn1d_v2.py`, em ordem de aparição no pipeline de treino.

---

## 1. O Pipeline de Dados (`tf.data`)

O `tf.data` é o sistema do TensorFlow para construir pipelines de dados eficientes. Em vez de carregar todos os dados na RAM de uma vez, ele cria uma "correia transportadora" que entrega lotes de dados à GPU sob demanda.

### 1.1 `tf.data.Dataset.from_generator`

```python
ds = tf.data.Dataset.from_generator(
    generator,
    output_signature=(
        tf.TensorSpec(shape=(WINDOW_SIZE, N_SENSORS), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    ),
)
```

**O que faz:** Cria um dataset a partir de uma função Python geradora. O gerador produz uma janela por vez (`yield`), sem precisar de todas as janelas em memória.

**Por que usamos:** O conjunto completo de janelas de treino ocupa ~3,5 GB. Com o gerador, nunca alocamos esse array — o TensorFlow busca janelas do gerador conforme a GPU precisa.

**`output_signature`:** Informa ao TensorFlow o formato e tipo de cada elemento. Necessário porque o TensorFlow compila o pipeline antecipadamente e precisa saber os shapes.

---

### 1.2 `.shuffle(buffer_size, reshuffle_each_iteration)`

```python
ds = ds.shuffle(buffer_size=20_000, reshuffle_each_iteration=True)
```

**O que faz:** Mantém um buffer de 20.000 janelas em memória e, a cada vez que o modelo pede um lote, sorteia aleatoriamente dentro do buffer antes de adicionar novos elementos.

**Visualização do funcionamento:**

```
Gerador produz janelas →  [Buffer: 20.000 janelas misturadas]  → GPU recebe lotes de 256
                                    ↑
                          À medida que janelas saem, novas entram
```

**Por que 20.000?** É um compromisso entre qualidade do shuffle e uso de RAM. Com janelas de ~9,6 KB cada, 20.000 janelas ocupam ~190 MB — suficiente para misturar bem sem comprometer a memória.

**`reshuffle_each_iteration=True`:** Reembaralha o buffer a cada vez que o dataset é percorrido. Garante que os lotes sejam diferentes a cada epoch.

---

### 1.3 `.repeat()`

```python
ds = ds.repeat()
```

**O que faz:** Faz o dataset recomeçar automaticamente do início quando o gerador esgota, criando um fluxo infinito de dados.

**Por que é crítico para performance:** Sem `.repeat()`, quando o gerador termina uma passagem pelos dados, o buffer de shuffle fica vazio e precisa ser repreenchido do zero antes da próxima epoch — custando ~325 segundos de CPU. Com `.repeat()`, o gerador recomeça antes que o buffer esvazie, mantendo-o sempre cheio.

```
Sem .repeat():    [Buffer cheio] → [Buffer vazio] → [Reenchimento: 325s] → [Buffer cheio]
Com .repeat():    [Buffer cheio] → [Buffer 70%]  → [Gerador reinicia]   → [Buffer cheio]
```

O número de passos por epoch é controlado por `steps_per_epoch` no `model.fit()`.

---

### 1.4 `.map(função, num_parallel_calls)`

```python
ds = ds.map(
    lambda x, y: (x, y, tf.gather(cw_tensor, tf.cast(y, tf.int32))),
    num_parallel_calls=tf.data.AUTOTUNE,
)
```

**O que faz:** Aplica uma transformação a cada elemento do dataset. Aqui usamos para adicionar o peso de classe (`sample_weight`) a cada janela.

**`tf.gather(cw_tensor, y)`:** Funciona como indexação de array — dado o índice de classe `y` (ex: 5), retorna o peso correspondente do tensor `cw_tensor` (ex: 3.2). É uma operação vetorizada no TensorFlow, mais eficiente que um `if/else` Python.

**`num_parallel_calls=tf.data.AUTOTUNE`:** Executa o mapeamento em múltiplas threads em paralelo. O `AUTOTUNE` deixa o TensorFlow decidir automaticamente quantas threads usar com base na carga da CPU e GPU.

---

### 1.5 `.batch(256)`

```python
ds = ds.batch(256)
```

**O que faz:** Agrupa 256 janelas individuais em um tensor de shape `(256, 300, 8)` para envio à GPU. A GPU processa todas as 256 janelas em paralelo — é aí que vem o ganho de velocidade.

**Por que 256?** É um valor padrão equilibrado. Lotes maiores aceleram o treino mas exigem mais VRAM; lotes menores são mais lentos mas atualizam os pesos com mais frequência.

---

### 1.6 `.prefetch(tf.data.AUTOTUNE)`

```python
ds = ds.batch(256).prefetch(tf.data.AUTOTUNE)
```

**O que faz:** Enquanto a GPU está processando o lote atual, a CPU já prepara o próximo lote. Elimina o tempo de espera entre lotes.

```
Sem prefetch:   [GPU treina lote 1] → [CPU prepara lote 2] → [GPU treina lote 2] → ...
Com prefetch:   [GPU treina lote 1]
                [CPU prepara lote 2] (em paralelo)
                                     [GPU treina lote 2]
                                     [CPU prepara lote 3] (em paralelo)
```

**`AUTOTUNE`:** O TensorFlow monitora os tempos de CPU e GPU e ajusta automaticamente quantos lotes são pré-carregados.

---

## 2. Pesos de Classe com `tf.gather`

```python
cw_tensor = tf.constant(
    [cw_dict.get(i, 1.0) for i in range(N_CLASSES)], dtype=tf.float32
)
```

**Por que não usamos `class_weight` do Keras?** O argumento `class_weight` do `model.fit()` não é compatível com `tf.data.Dataset`. A alternativa é incorporar o peso diretamente como terceiro elemento do dataset (`sample_weight`), que o Keras processa automaticamente quando o dataset emite tuplas de 3 elementos `(X, y, peso)`.

**Como funciona na prática:**
- `cw_tensor` é um array de 19 pesos, um por classe
- Para cada janela com label `y=5`, `tf.gather(cw_tensor, 5)` retorna o peso da classe 5
- O Keras multiplica a loss daquela amostra pelo peso antes de atualizar os pesos do modelo

---

## 3. A Arquitetura no TensorFlow (Keras)

### 3.1 `BatchNormalization`

```python
x = BatchNormalization()(x)
```

**O que faz:** Após cada camada convolucional, normaliza as ativações para que tenham média ~0 e desvio padrão ~1, por mini-lote.

**Por que importa:** Sem isso, os valores das ativações podem crescer ou encolher ao longo das camadas, tornando o treino instável. O BatchNorm mantém os valores em uma faixa saudável, permitindo taxas de aprendizado maiores e convergência mais rápida.

**Detalhe:** O BatchNorm tem parâmetros treináveis (`gamma` e `beta`) que permitem à rede desfazer a normalização se necessário. Por isso aparece como "Non-trainable params: 1,024" no summary — são os parâmetros de média e variância calculados como média móvel durante o treino.

### 3.2 `GlobalAveragePooling1D`

```python
x = GlobalAveragePooling1D()(x)
```

**O que faz:** Recebe um tensor de shape `(batch, 300, 128)` e calcula a média ao longo dos 300 timesteps para cada um dos 128 canais, produzindo `(batch, 128)`.

**Comparação com Flatten:**
- `Flatten`: transforma `(300, 128)` em `(38.400,)` — vetor gigante, propenso a overfitting
- `GlobalAveragePooling`: transforma `(300, 128)` em `(128,)` — resumo compacto, mais robusto

### 3.3 `use_bias=False` nas camadas Conv1D

```python
Conv1D(128, 8, padding="same", use_bias=False)
```

O bias é desabilitado nas camadas convolucionais porque o `BatchNormalization` logo após já tem seu próprio parâmetro de deslocamento (`beta`). Manter os dois seria redundante.

---

## 4. Compilação e XLA

### 4.1 `model.compile`

```python
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
```

**`sparse_categorical_crossentropy`:** Versão da cross-entropy que aceita labels como inteiros (0, 1, 2, ...) em vez de vetores one-hot. Equivalente a `categorical_crossentropy` com `to_categorical(y)`, mas mais eficiente em memória.

**`Adam`:** Otimizador adaptativo que ajusta a taxa de aprendizado individualmente para cada parâmetro. É o padrão para redes neurais — combina as vantagens do Momentum (aceleração na direção do gradiente) com o RMSProp (escalonamento por magnitude do gradiente).

### 4.2 XLA — Accelerated Linear Algebra

```
Compiled cluster using XLA! This line is logged at most once...
```

**O que é:** XLA é um compilador que transforma operações TensorFlow em código nativo otimizado para GPU. Na primeira execução, o TensorFlow analisa o grafo computacional e compila um kernel CUDA especializado.

**Impacto:** A epoch 1 é mais lenta porque inclui a compilação XLA (~10s). A partir da epoch 2, o kernel compilado é reutilizado — daí a aceleração visível entre epoch 1 e epoch 2.

---

## 5. Callbacks de Treino

### 5.1 `EarlyStopping`

```python
EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
```

**O que faz:** Monitora o `val_loss` ao final de cada epoch. Se não melhorar por 10 epochs consecutivas, interrompe o treino e restaura os pesos da epoch com menor `val_loss`.

**`restore_best_weights=True`:** Fundamental — sem isso, o modelo ficaria com os pesos da última epoch (que pode ter overfitting), não com os pesos da melhor epoch.

### 5.2 `ReduceLROnPlateau`

```python
ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5)
```

**O que faz:** Se `val_loss` não melhora por 5 epochs, reduz a taxa de aprendizado à metade. Permite que o modelo "refine" em regiões próximas ao mínimo com passos menores.

**Interação com EarlyStopping:** O ReduceLROnPlateau tem `patience=5` e o EarlyStopping `patience=10`. Ou seja: a taxa de aprendizado é reduzida primeiro, dando ao modelo mais 5 epochs de chance com lr menor antes do treino ser encerrado.

```
Sem melhora por 5 epochs → lr = lr × 0.5
Sem melhora por 10 epochs → treino encerra
```

---

## 6. GPU e Memória

### 6.1 Alocação de VRAM

```
Created device GPU:0 with 4045 MB memory
```

O TensorFlow reserva 4 GB dos 6 GB da RTX 2060 para si ao iniciar. O restante (~2 GB) fica disponível para o sistema.

**Por que não usa tudo?** O TensorFlow deixa uma margem para evitar conflitos com outros processos (display, SO). É possível configurar para usar mais com `tf.config.experimental.set_memory_growth`, mas para nossa FCN (modelo pequeno, ~1 MB de parâmetros) 4 GB é mais que suficiente.

### 6.2 `tf.keras.backend.clear_session()`

```python
tf.keras.backend.clear_session()
```

Chamado ao final de cada fold. Libera o grafo Keras da memória e reinicia os contadores internos. Sem isso, cada novo modelo do fold seguinte acumularia memória do anterior, podendo causar OOM após vários folds.

---

## 7. `steps_per_epoch`

```python
n_train_windows = int(df_meta[df_meta["instance_id"].isin(fit_iids)]["window_label"].notna().sum())
steps_per_epoch = max(1, n_train_windows // 256)
```

**Por que é necessário com `.repeat()`:** Como o dataset é infinito (`.repeat()`), o Keras não sabe quando uma "epoch" termina. O `steps_per_epoch` define isso: após `N` lotes de 256 janelas, o Keras considera a epoch completa, avalia o conjunto de validação e verifica os callbacks.

**Como é calculado:** Estimamos o número de janelas de treino a partir dos metadados (`df_meta`) e dividimos pelo tamanho do lote. É uma aproximação — algumas janelas são filtradas em `extract_windows` — mas é suficientemente precisa para definir o tamanho da epoch.

---

## 8. Resumo do Fluxo Completo

```
cleaned.parquet
      │
      ▼ (lotes de 200K linhas, pyarrow)
instance_raw (pandas, temporário)
      │
      ▼ (normalização Z-score + filtro Gaussiano, numpy)
instance_arrays + instance_class (numpy, permanente em RAM)
      │
      ├──► collect_windows() → X_val, y_val (materializado, ~350 MB)
      ├──► collect_windows() → X_test, y_test (materializado, ~860 MB)
      │
      └──► make_train_dataset()
                │
                ▼
         from_generator → shuffle(20K) → repeat() → map(pesos) → batch(256) → prefetch
                │                                                                   │
                │ (CPU produz janelas continuamente)              (GPU treina) ◄────┘
                │
                ▼
         model.fit(steps_per_epoch=N)
                │
                ├── EarlyStopping (patience=10, val_loss)
                └── ReduceLROnPlateau (patience=5, factor=0.5)
```
