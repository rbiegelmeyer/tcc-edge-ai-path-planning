# Python_Trainer

Pipeline completo de treinamento, destilação e otimização de redes neurais para
**segmentação semântica de caminhos** em mapas de grade — com foco em deployment
em microcontroladores STM32 via quantização INT8.

---

## Visão geral

O objetivo do pipeline é treinar um modelo U-Net capaz de, dado um mapa com
obstáculos, início e fim, predizer o caminho ótimo (equivalente ao A\*) diretamente
como uma máscara binária 64×64. O modelo treinado é então destilado, quantizado
e validado no hardware alvo.

```
CSV de mapas
    │
    ▼  csv_to_npz.py
 Dataset NPZ
    │
    ▼  Coach.py
 Modelo Teacher (.keras)
    │
    ├──▶ ConvertKeras2TFLite.py  →  .tflite + _int8.tflite
    │
    ├──▶ ConvertKeras2ONNX.py   →  .onnx + _quantized_static.onnx
    │
    └──▶ Distiller.py
              │
              ▼
         Modelo Student (.keras)
              │
              └──▶ ConvertKeras2ONNX.py  →  student_distilled_quantized_static.onnx
                        │
                        ▼  validate_device.py
                   Validação STM32 (stedgeai)
```

---

## Estrutura de diretórios

```
Python_Trainer/
├── Coach.py                  # Treinamento do U-Net teacher
├── Distiller.py              # Destilação de conhecimento (teacher → student)
├── ConvertKeras2ONNX.py      # Conversão Keras → ONNX + quantização estática
├── ConvertKeras2TFLite.py    # Conversão Keras → TFLite (float32 + INT8)
├── Metrics.py                # Métricas e losses customizadas (IoU, Dice, Reachability)
├── Visualizer.py             # Utilitários de visualização usados pelo Coach
├── pipeline.py               # Orquestrador do pipeline completo
│
├── tests/                    # Avaliação e inferência dos modelos
│   ├── test_keras_model.py   # Testa modelo Keras no conjunto de teste
│   ├── test_tflite_model.py  # Testa modelo TFLite no conjunto de teste
│   ├── test_onnx_model.py    # Testa modelo ONNX no conjunto de teste
│   ├── test_models.py        # Compara todos os modelos de uma pasta
│   ├── validate_device.py    # Valida no STM32 via stedgeai CLI
│   ├── infer.py              # Inferência CLI universal com métricas detalhadas
│   └── README.md
│
├── utils/                    # Conversão de dados e ferramentas de suporte
│   ├── csv_to_npz.py         # CSV de mapas → NPZ comprimido
│   ├── npy_to_header.py      # NPY/NPZ → header C (x_val_data.h)
│   ├── csv_to_header.py      # CSV de mapas → header C (A* validation)
│   ├── bin_to_map.py         # Dump binário STM32 → heatmap PNG
│   ├── quantize_onnx.py      # Exemplo de quantização ONNX manual
│   └── README.md
│
├── validations/              # Análise visual e comparações
│   ├── compare_models.py     # A* vs Modelo 1 vs Modelo 2 vs Heatmap embarcado
│   ├── compare_results.py    # Comparação teacher vs student (imagens existentes)
│   ├── analyze_training_results.py  # Hiper parâmetros: sfilter × depth
│   ├── best_model_comparison.py     # Melhor modelo por tamanho de dataset
│   ├── compare_maps.py       # Colagem esperado (SDL2 BMP) vs atingido (MCU PNG)
│   ├── plot_result.py        # Dump binário A* → plot com paleta Catppuccin
│   ├── generate_architecture_diagram.py  # Diagrama da arquitetura U-Net
│   └── README.md
│
└── results/                  # Gerado pelo pipeline (não versionado)
    └── result_<config>/
        ├── best_path_finder_Unet1.keras
        ├── student_distilled.keras
        ├── *.onnx / *.tflite
        ├── X_test.npy / Y_test.npy
        ├── X_val.npy  / Y_val.npy
        └── test/             # Visualizações geradas pelos scripts de teste
```

---

## Quick start

### 1. Configurar o ambiente

```bash
cd Python_Trainer
uv sync          # ou: pip install -r requirements.txt
```

### 2. Preparar o dataset

```bash
# Converte CSV de mapas para NPZ
python utils/csv_to_npz.py dados/W064xH064_D03_S000000_E050000.csv
```

### 3. Executar o pipeline completo

Edite `pipeline.py` para configurar os datasets e tarefas de treinamento:

```python
DATASETS = ['dados/W064xH064_D03_S000000_E050000.csv']

tasks = [
    # (sfilter, depth, dataset_index)
    (64, 4, 0),
]
```

Depois execute:

```bash
python pipeline.py
```

Ou execute as etapas individualmente:

```bash
# Treinar o teacher
python Coach.py

# Destilar o student
python Distiller.py

# Converter para ONNX + quantizar
python ConvertKeras2ONNX.py
```

### 4. Avaliar o modelo

```bash
# Inferência com métricas (modelo ONNX, 10 amostras)
python tests/infer.py \
    --npz  results/result_.../X_test.npy \
    --model results/result_.../best_path_finder_Unet1.onnx \
    --samples 10 --all

# Comparação teacher vs student no hardware
python validations/compare_models.py \
    --npz    dados/validacao.npz \
    --model1 results/result_.../best_path_finder_Unet1.keras \
    --model2 results/result_.../student_distilled.keras \
    --name1  "Teacher" --name2 "Student"
```

### 5. Validar no STM32

Configure `STEDGEAI_PATH` e os workspaces em `tests/validate_device.py`, depois:

```bash
python tests/validate_device.py --host-only   # sem hardware físico
python tests/validate_device.py              # com STM32 conectado via ST-Link
```

---

## Módulos principais

### `Coach.py`
Treina a U-Net teacher com busca de hiperparâmetros (sfilter × depth). Salva
checkpoints, logs de treinamento (JSON + CSV) e visualizações do conjunto de teste.

**Entradas:** arquivo CSV de mapas com colunas `id, start_x, start_y, end_x, end_y, map`.

**Saídas:** `results/result_<W>x<H>_D<depth>_S<start>_E<end>/`

### `Distiller.py`
Treina um U-Net compacto (student) usando as predições do teacher como soft labels
via divergência KL. Aplica early stopping e salva `student_distilled.keras`.

### `ConvertKeras2ONNX.py`
Converte `.keras` para ONNX (opset 13) e aplica quantização estática INT8 com dados
de calibração extraídos do conjunto de teste.

### `ConvertKeras2TFLite.py`
Converte `.keras` para TFLite float32 e INT8 com quantização pós-treinamento.

### `Metrics.py`
Métricas e losses customizadas usadas durante o treinamento e avaliação:

| Nome | Descrição |
|---|---|
| `iou_metric` | Jaccard Index (IoU) — métrica principal |
| `continuity_metric` | Penaliza caminhos fragmentados |
| `segment_count_metric` | Score por número de segmentos desconexos |
| `reachability_metric` | Verifica se o caminho conecta início ao fim |
| `dice_loss` | Dice Loss para segmentação |
| `bce_dice_loss` | Combinação BCE + Dice Loss |

### `pipeline.py`
Orquestra as 3 etapas do pipeline para múltiplos datasets:
1. Treinamento do teacher (`Coach.py`)
2. Destilação do student (`Distiller.py`)
3. Conversão e quantização ONNX (`ConvertKeras2ONNX.py`)

---

## Formato dos dados

### Entrada do modelo (X)
Array float32 `(N, 64, 64, 3)`:
- Canal 0 — obstáculos (1.0 = parede)
- Canal 1 — posição de início (1.0 = pixel de início)
- Canal 2 — posição de fim (1.0 = pixel de fim)

### Saída esperada (Y)
Array float32 `(N, 64, 64, 1)`: máscara binária do caminho A\* (1.0 = caminho).

### Codificação no CSV
String flat de `H×W` dígitos: `0`=vazio, `1`=parede, `2`=caminho, `3`=início, `4`=fim.

---

## Nomenclatura das pastas de resultado

```
result_W064xH064_D04_S000000_E005000
         │    │   │   │         │
         │    │   │   │         └── Índice final do dataset (E = end)
         │    │   │   └──────────── Índice inicial do dataset (S = start)
         │    │   └──────────────── Profundidade do U-Net (D = depth)
         │    └──────────────────── Altura do mapa
         └───────────────────────── Largura do mapa
```

---

## Dependências principais

| Biblioteca | Uso |
|---|---|
| TensorFlow / Keras | Treinamento, inferência, conversão TFLite |
| ONNX Runtime | Inferência ONNX, quantização |
| tf2onnx | Conversão Keras → ONNX |
| NumPy / SciPy | Processamento de arrays, métricas |
| Matplotlib / Pillow | Visualizações e colagens |
| Pandas | Leitura e análise de logs CSV |
| STM32 Cube AI Studio | Validação no hardware (stedgeai CLI) |
