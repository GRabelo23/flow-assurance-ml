# CLAUDE.md — TCC: Garantia de Escoamento com 3W Dataset

## Contexto do Projeto

TCC de Engenharia Mecatrônica (UnB) — Análise e Modelagem Integrada de Dados de Garantia de Escoamento.

**Objetivo:** Pipeline de ML com duas abordagens complementares:
1. **Diagnóstico de falha** — dada uma janela de 300 s de sensores, classificar em qual dos 10 tipos de evento o poço está inserido (Normal + 9 falhas)
2. **Detecção de estado operacional** — classificar o estado instantâneo da janela: operação normal, transiente (falha se aproximando) ou evento ativo, por tipo de falha (19 estados)

**Dataset:** Parquets brutos em `D:\Documentos\UnB\Projeto Final de Curso\3W\dataset\`
organizados por classe (pastas 0 a 9).

---

## Estrutura do Projeto

```
TCC/
├── CLAUDE.md                    ← este arquivo
├── plan.md                      ← plano metodológico do pipeline
├── config.py                    ← caminhos e constantes centralizados
├── requirements.txt             ← dependências Python
├── project_state.md             ← estado atual do projeto
├── todo.md                      ← checklist do que falta fazer
├── src/
│   ├── data_loader.py           ← carregamento dos parquets brutos
│   ├── feature_engineering.py  ← janela deslizante, features, rotulagem
│   ├── evaluation.py           ← métricas e plots de avaliação
│   └── visualization.py        ← gráficos padronizados para o TCC
├── scripts/                     ← scripts standalone para tarefas longas
│   ├── train_random_forest.py      ← treina RF (já executado)
│   ├── plot_confusion_matrix.py    ← gera matriz de confusão do RF
│   └── run_pipeline_window_class.py ← gera features com rotulagem por estado
├── notebooks/
│   ├── 01_analise_exploratoria.ipynb
│   ├── 02_limpeza_preparacao.ipynb
│   ├── 03_engenharia_features.ipynb
│   ├── 04_modelagem.ipynb
│   ├── 05_avaliacao.ipynb
│   └── 06_interpretacao.ipynb
├── data/processed/
│   ├── cleaned.parquet              ← séries temporais limpas
│   ├── features.parquet             ← rotulagem por instância (10 classes)
│   └── features_window_class.parquet ← rotulagem por estado (19 classes)
└── results/
    ├── models/                  ← modelos treinados (.joblib)
    ├── metrics/                 ← métricas em CSV/JSON
    └── figures/                 ← gráficos prontos para o TCC
```

---

## Duas Abordagens de Rotulagem

### Abordagem 1 — Por Instância (`features.parquet`)
Cada janela herda a `fault_class` do poço inteiro (0–9).
- **Label:** qual tipo de falha ocorreu nesse poço?
- **Uso:** diagnóstico retroativo — "esse poço teve qual evento?"
- **10 classes:** Normal, BSW, DHSV, Golfadas, Instabilidade, Produtividade, PCK Restrição, PCK Incrustação, Hidrato Produção, Hidrato Serviço

### Abordagem 2 — Por Estado (`features_window_class.parquet`)
Cada janela recebe a **moda da coluna `class`** dentro daquela janela.
- **Label:** qual é o estado operacional nesse momento?
- **Uso:** monitoramento em tempo real — detecta progressão normal → transiente → ativo
- **19 classes:** 0 (normal), 1–9 (evento ativo por tipo), 101–109 (transiente por tipo)
- Janelas onde 100% dos timestamps têm `class=NaN` são descartadas

---

## Classes do 3W Dataset

| Classe | Evento | Ativo | Transiente |
|--------|--------|-------|-----------|
| 0 | Operação normal | — | — |
| 1 | Aumento abrupto de BSW | class=1 | class=101 |
| 2 | Fechamento espúrio da DHSV | class=2 | class=102 |
| 3 | Golfadas severas | class=3 | class=103 |
| 4 | Instabilidade de fluxo | class=4 | class=104 |
| 5 | Perda rápida de produtividade | class=5 | class=105 |
| 6 | Restrição rápida no PCK | class=6 | class=106 |
| 7 | Incrustação no PCK | class=7 | class=107 |
| 8 | Hidrato na linha de produção | class=8 | class=108 |
| 9 | Hidrato na linha de serviço | class=9 | class=109 |

---

## Decisões Técnicas Importantes

| Decisão | Por quê |
|---------|---------|
| `GroupKFold` por `instance_id` | Janelas do mesmo poço nunca aparecem em treino e teste ao mesmo tempo |
| Z-score por instância | Poços operam em faixas absolutas distintas; modelo aprende padrões de mudança |
| Sensores constantes → 0.0 | Evita que nível absoluto (0 vs 38 MPa) vaze como feature |
| `batch_size=30` no pipeline | Limita uso de RAM a ~500 MB por lote (RAM disponível: ~2.7 GB) |
| `class_weight='balanced'` | Compensa desbalanceamento entre classes |
| Outliers preservados como `max_zscore` | Picos são assinatura da falha, não ruído |
| Forward-fill ≤ 60 s | Respeita causalidade; buracos maiores permanecem como NaN |

---

## Convenções de Código

- **Variáveis, funções e classes:** inglês (padrão scikit-learn/pandas)
- **Comentários, docstrings e outputs de células:** português
- Saídas intermediárias → `data/processed/`
- Resultados finais → `results/`
- Tarefas longas (pipeline completo, treino) → `scripts/` em vez de notebooks

---

## Ambiente

```bash
pip install -r requirements.txt
python -c "import config; print(config.RAW_DATA_DIR)"
```

---

## Comunicação

- Sempre comunicar em **português brasileiro**
- Explicar conceitos de forma didática, como se o usuário tivesse 15 anos
- Evitar jargões técnicos sem explicação
- Contextualizar o "por quê" de cada decisão metodológica
