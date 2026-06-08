# Path Finder — CNN para Predição de Caminhos em STM32

Pipeline completo de treinamento, destilação e implantação de redes neurais convolucionais
para predição de caminhos em mapas de grade — do gerador de datasets C até a inferência
em microcontroladores STM32 via quantização INT8.

---

## Visão Geral

O objetivo é treinar um modelo U-Net capaz de, dado um mapa com obstáculos e posições
de início e fim, predizer o caminho ótimo equivalente ao calculado pelo algoritmo A\*.
O modelo professor (maior, mais preciso) é destilado num modelo aluno (menor, adequado
para embarcados), que é quantizado e implantado no STM32 via STM32 Cube AI Studio.

```
┌─────────────────────────────────────────────────────────────┐
│  AStar/           C — gerador de datasets via A*            │
│    └─ *.csv  ──────────────────────────────────────────┐    │
│                                                        ▼    │
│  Python_Trainer/  Python — treinamento e otimização        │
│    ├─ Coach.py          treina U-Net teacher               │
│    ├─ Distiller.py      destila para U-Net student         │
│    ├─ ConvertKeras2ONNX converte + quantiza INT8           │
│    └─ *.onnx  ─────────────────────────────────────┐       │
│                                                    ▼       │
│  MCU/             C — firmware STM32 com inferência        │
└─────────────────────────────────────────────────────────────┘
```

---

## Estrutura do Repositório

```
TCC/
├── AStar/                        # Gerador de datasets (C, A*)
│   ├── main.c / astar.c / astar.h
│   ├── Makefile
│   ├── datasets/                 # CSVs gerados (não versionados)
│   └── README.md
│
├── Python_Trainer/               # Pipeline de ML (Python)
│   ├── pipeline.py               # Orquestrador — executa tudo em sequência
│   ├── Coach.py                  # Treinamento do modelo teacher (U-Net)
│   ├── Distiller.py              # Destilação de conhecimento (teacher → student)
│   ├── ConvertKeras2ONNX.py      # Keras → ONNX + quantização INT8 estática
│   ├── convertKeras2TFLite.py    # Keras → TFLite (alternativa)
│   ├── Metrics.py                # Métricas e losses customizadas
│   ├── Visualizer.py             # Geração de imagens de resultado
│   │
│   ├── tests/                    # Avaliação e inferência
│   │   ├── test_keras_model.py   # Avalia modelo .keras no conjunto de teste
│   │   ├── test_tflite_model.py  # Avalia modelo .tflite no conjunto de teste
│   │   ├── test_onnx_model.py    # Avalia modelo .onnx no conjunto de teste
│   │   ├── test_models.py        # Compara todos os modelos de uma pasta
│   │   ├── validate_device.py    # Valida no STM32 via stedgeai CLI
│   │   ├── infer.py              # Inferência CLI universal com métricas
│   │   └── README.md
│   │
│   ├── utils/                    # Conversão de dados e ferramentas de suporte
│   │   ├── csv_to_npz.py         # CSV de mapas → NPZ comprimido
│   │   ├── npy_to_header.py      # NPY/NPZ → header C (x_val_data.h)
│   │   ├── csv_to_header.py      # CSV de mapas → header C (A* validation)
│   │   ├── bin_to_map.py         # Dump binário STM32 → heatmap PNG
│   │   ├── quantize_onnx.py      # Exemplo de quantização ONNX manual
│   │   └── README.md
│   │
│   ├── validations/              # Análise visual e comparações
│   │   ├── compare_models.py     # A* vs Modelo 1 vs Modelo 2 vs Heatmap embarcado
│   │   ├── compare_results.py    # Comparação teacher vs student
│   │   ├── compare_maps.py       # Colagem esperado (SDL2) vs atingido (MCU)
│   │   ├── analyze_training_results.py  # Grade sfilter × depth
│   │   ├── best_model_comparison.py     # Melhor modelo por tamanho de dataset
│   │   ├── plot_result.py        # Dump binário A* → plot Catppuccin
│   │   ├── generate_architecture_diagram.py  # Diagrama da U-Net
│   │   └── README.md
│   │
│   └── README.md
│
├── MCU/                          # Firmware STM32 (CubeIDE)
│   └── H7/                       # Projeto STM32H743
│
├── Anotacoes/                    # Logs, anotações e PDFs do TCC
├── Bibliografia/                 # Referências bibliográficas
├── Images/                       # Imagens usadas no relatório
├── Overleaf/                     # Capítulos LaTeX do TCC
└── README.md
```

---

## Formato dos Dados

### CSV gerado pelo AStar (entrada do pipeline)

| Coluna | Descrição |
|---|---|
| `id` | ID do mapa (também é o seed para posições de início/fim) |
| `difficulty` | Densidade de obstáculos |
| `start_x`, `start_y` | Coluna e linha do ponto de início |
| `end_x`, `end_y` | Coluna e linha do ponto de fim |
| `height`, `width` | Dimensões do mapa em células |
| `map` | String flat de `H×W` dígitos: `0`=vazio `1`=parede `2`=caminho `3`=início `4`=fim |

### Entrada da rede neural (X)

Array float32 `(N, 64, 64, 3)` — 3 canais por pixel:
- Canal 0 — obstáculos
- Canal 1 — posição de início
- Canal 2 — posição de fim

### Saída esperada (Y)

Array float32 `(N, 64, 64, 1)` — máscara binária do caminho A\*.

### Nomenclatura das pastas de resultado

```
result_W064xH064_D04_S000000_E005000
         │    │   │   │         └── Índice final do dataset
         │    │   │   └──────────── Índice inicial do dataset
         │    │   └──────────────── Profundidade (depth) do U-Net
         │    └──────────────────── Altura do mapa
         └───────────────────────── Largura do mapa
```

---

## Requisitos

```bash
# Instalar dependências Python (via uv ou pip)
cd Python_Trainer
uv sync
# ou:
pip install tensorflow onnx tf2onnx onnxruntime scikit-learn pandas matplotlib pillow scipy
```

Para compilar o gerador de datasets (Linux / WSL2):

```bash
cd AStar
make          # build sem visualização
make visualize  # build com SDL2 (requer: sudo apt install libsdl2-dev)
```

Para validação no STM32, é necessário o **STM32 Cube AI Studio** com `stedgeai`.
Configure o caminho em `Python_Trainer/tests/validate_device.py`:

```python
STEDGEAI_PATH = r"C:\ST\STEdgeAI\4.0\Utilities\windows\stedgeai.exe"
```

---

## Tutorial de Uso

### Etapa 0 — Gerar o dataset (AStar)

```bash
cd AStar
./astar          # gera CSV em datasets/
./astar_vis      # modo com visualização SDL2
```

Edite os parâmetros em `main.c` (tamanho do grid, difficulty, range de IDs).
O arquivo gerado tem o formato `W064xH064_D<diff>_S<start>_E<end>.csv`.

### Etapa 1 — Converter CSV para NPZ

```bash
cd Python_Trainer
python utils/csv_to_npz.py ../AStar/datasets/W064xH064_D03_S000000_E050000.csv
```

### Etapa 2 — Pipeline completo (recomendado)

Edite `Python_Trainer/pipeline.py` com os datasets e configurações desejadas:

```python
DATASETS = ['../AStar/datasets/W064xH064_D03_S000000_E050000.csv']

tasks = [
    # (sfilter, depth, dataset_index)
    (64, 4, 0),
]
```

Depois execute:

```bash
cd Python_Trainer
python pipeline.py
```

O pipeline executa automaticamente: **treinamento → destilação → conversão ONNX**.

### Etapa 3 — Execução por etapa (alternativa)

```bash
cd Python_Trainer

# 3a. Treinar o teacher
python Coach.py

# 3b. Destilar o student
python Distiller.py

# 3c. Converter para ONNX + quantizar INT8
python ConvertKeras2ONNX.py
```

### Etapa 4 — Avaliar os modelos

```bash
# Avaliação rápida no conjunto de teste (edite results_path no topo do arquivo)
python tests/test_keras_model.py
python tests/test_tflite_model.py
python tests/test_onnx_model.py

# Inferência com métricas detalhadas (CLI)
python tests/infer.py \
    --npz  results/result_W064xH064_D04_S000000_E005000/X_test.npy \
    --model results/result_W064xH064_D04_S000000_E005000/best_path_finder_Unet1.keras \
    --samples 10 --all
```

### Etapa 5 — Comparação teacher vs student

```bash
# Comparação visual: A* | Teacher | Student | Heatmap STM32
python validations/compare_models.py \
    --npz    results/result_.../X_test.npy \
    --model1 results/result_.../best_path_finder_Unet1.keras  --name1 "Teacher" \
    --model2 results/result_.../student_distilled.keras        --name2 "Student" \
    -o ./comparativo -c 10

# Comparação das predições já geradas pelos test_*.py
python validations/compare_results.py \
    --results results/result_W064xH064_D04_S000000_E005000
```

### Etapa 6 — Validar no STM32

Configure os workspaces em `tests/validate_device.py`:

```python
BOARDS = {
    "stm32h743": {
        "target":     "stm32h7",
        "board_name": "NUCLEO-H743ZI2",
        "workspace":  r"C:\Users\...\workspace\TesteH7",
    },
    "stm32n657": {
        "target":      "stm32n6",
        "board_name":  "STM32N6570-DK",
        "extra_flags": ["--st-neural-art", "--use-onnx-simplifier"],
        "workspace":   r"C:\Users\...\workspace\TesteN6\...",
    },
}
```

Depois execute:

```bash
python tests/validate_device.py               # todos os modelos, todos os boards
python tests/validate_device.py --host-only   # sem hardware físico conectado
python tests/validate_device.py --board stm32h743
```

Gera `validation_report.json` com análise estática (MACCs, RAM, Flash) e métricas
de validação no host e no device.

---

## Métricas

Definidas em `Python_Trainer/Metrics.py`:

| Métrica | Descrição |
|---|---|
| `iou_metric` | Jaccard Index — métrica principal de segmentação |
| `continuity_metric` | Penaliza caminhos fragmentados |
| `segment_count_metric` | Decaimento exponencial por segmento desconexo |
| `reachability_metric` | 1.0 se o caminho predito conecta início ao fim |
| `bce_dice_loss` | Loss de treinamento: BCE + Dice (50/50) |

---

## Modelos

### Teacher — `best_path_finder_Unet<N>.keras`

U-Net com encoder/decoder de 4 níveis e filtros configuráveis via `sfilter` e `depth`.
Treinado em GPU, tamanho típico ~10–30 MB.

### Student — `student_distilled.keras`

U-Net compacto (filtros fixos 8→128), treinado via destilação de conhecimento usando
as predições do teacher como soft labels (divergência KL). Tamanho ~1–2 MB.

### Modelos quantizados

| Formato | Arquivo | Uso |
|---|---|---|
| ONNX FP32 | `best_path_finder_Unet1.onnx` | Inferência PC/GPU |
| ONNX INT8 | `*_quantized_static.onnx` | Deploy STM32 via stedgeai |
| TFLite FP32 | `*.tflite` | Dispositivos Android/ARM |
| TFLite INT8 | `*_int8.tflite` | Dispositivos Android/ARM quantizados |

---

## Documentação Adicional

Cada subcomponente possui seu próprio README com documentação detalhada:

| Componente | README |
|---|---|
| Gerador de datasets A\* | [AStar/README.md](AStar/README.md) |
| Pipeline ML completo | [Python_Trainer/README.md](Python_Trainer/README.md) |
| Scripts de avaliação | [Python_Trainer/tests/README.md](Python_Trainer/tests/README.md) |
| Utilitários de conversão | [Python_Trainer/utils/README.md](Python_Trainer/utils/README.md) |
| Análise e visualização | [Python_Trainer/validations/README.md](Python_Trainer/validations/README.md) |
