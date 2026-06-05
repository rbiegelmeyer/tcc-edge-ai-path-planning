# tests/

Scripts de avaliação e teste dos modelos treinados pelo pipeline. Cada script carrega
um subconjunto de teste (`X_test.npy` / `Y_test.npy`) e gera visualizações comparativas
salvas em `<results_path>/test/`.

---

## Pré-requisitos comuns

Todos os scripts assumem que a pasta de resultados (`results_path`) contém:

| Arquivo | Descrição |
|---|---|
| `X_test.npy` | Entradas do conjunto de teste — shape `(N, 64, 64, 3)` float32 |
| `Y_test.npy` | Rótulos ground-truth (caminhos A\*) — shape `(N, 64, 64, 1)` float32 |
| Modelo (`.keras` / `.tflite` / `.onnx`) | Checkpoint a avaliar |

> O conjunto de teste é gerado automaticamente pelo `Coach.py` durante o treinamento
> e salvo na mesma pasta do resultado.

Execute os scripts a partir do diretório `Python_Trainer/`:

```bash
cd Python_Trainer
python tests/<script>.py
```

---

## Scripts

### `test_keras_model.py`

Avalia o modelo **Keras** (`.keras`) no conjunto de teste e gera imagens comparativas.

**Configuração** (editar no topo do arquivo):

```python
results_path = './results/result_W064xH064_D04_S000000_E005000'
```

**Saída:**
```
<results_path>/test/<nome_do_modelo>/mapa_predicao_<i>.png
```

Cada imagem contém 3 painéis:
- **Input** — mapa de entrada com obstáculos, início e fim
- **Target** — caminho real calculado pelo A\*
- **Previsão** — máscara binarizada da CNN (threshold = 0.5)

O número de amostras visualizadas é 15% do conjunto de teste.

---

### `test_tflite_model.py`

Avalia o modelo **TFLite** (`.tflite`) no conjunto de teste, processando cada amostra
individualmente (o intérprete TFLite não suporta batch).

**Configuração:**

```python
results_path = './results/result_W064xH064_D04_S000000_E005000'
```

**Saída:**
```
<results_path>/test/tflite/mapa_predicao_<i>.png
```

> Mais lento que Keras/ONNX por processar uma amostra por vez, mas representa
> fielmente o comportamento do modelo em dispositivos embarcados.

---

### `test_onnx_model.py`

Avalia o modelo **ONNX** (`.onnx`) usando ONNX Runtime. Suporta execução em GPU
(CUDAExecutionProvider) e processa todo o dataset em batch.

**Configuração:**

```python
results_path = './results/result_W064xH064_D04_S000000_E005000'
```

**Saída:**
```
<results_path>/test/Unet1_onnx/mapa_predicao_<i>.png
```

**GPU:** Por padrão, tenta usar CUDA. Para forçar CPU, edite:

```python
session = ort.InferenceSession(model_name, providers=['CPUExecutionProvider'])
```

---

### `test_models.py`

Descobre e compara **todos os modelos** presentes na pasta de resultados (`.keras`,
`.tflite`, `.onnx`), gerando uma imagem por amostra com uma linha por modelo.

**Configuração:**

```python
results_path = './results/result_W064xH064_D05_S000000_E005000'
```

**Saída:**
```
<results_path>/test_models/mapa_predicao_<i>.png
```

**Detecção automática de formato** pelos magic bytes do arquivo:
- `HDF5` → Keras
- `TFL3` → TFLite
- `onnx` / `ir_version` → ONNX

> Dependência: importa `iou_metric`, `continuity_metric` e `path_quality_metric`
> do módulo `Metrics.py` na raiz do `Python_Trainer/`. Execute a partir desse diretório.

---

### `validate_device.py`

Valida modelos ONNX quantizados (`*_quantized_static.onnx`) no hardware STM32
via **STM32 Cube AI Studio** (CLI `stedgeai`).

Realiza 3 etapas por modelo e por board:
1. **Análise estática** — MACCs, RAM, Flash
2. **Validação no host** — simulação do target no PC
3. **Validação no device** — execução real via ST-Link (opcional)

**Configuração necessária** (editar no topo do arquivo):

```python
STEDGEAI_PATH = r"C:\ST\STEdgeAI\4.0\Utilities\windows\stedgeai.exe"
RESULTS_DIR   = "./results"
```

Boards pré-configurados: `stm32h743` (Nucleo-H743ZI2) e `stm32n657` (STM32N6570-DK).
Edite a chave `"workspace"` de cada board para apontar ao workspace do STM32 Cube AI Studio.

**CLI:**

```bash
# Valida todos os modelos quantizados em todos os boards
python tests/validate_device.py

# Apenas validação no host (sem device físico conectado)
python tests/validate_device.py --host-only

# Board específico
python tests/validate_device.py --board stm32h743

# Modelo específico
python tests/validate_device.py --model results/result_.../student_distilled_quantized_static.onnx

# Valida todos os ONNX (não só os quantizados)
python tests/validate_device.py --all-onnx

# Relatório em caminho customizado
python tests/validate_device.py --report meu_relatorio.json
```

**Saída:** `validation_report.json` com análise, métricas e logs de cada etapa.

---

### `infer.py`

Script de inferência e visualização de uso geral. Suporta todos os formatos de modelo
(`.keras`, `.h5`, `.onnx`, `.tflite`) e calcula métricas detalhadas.

**CLI:**

```bash
# Inferência básica em 5 amostras
python tests/infer.py --npz data.npz --model model.keras

# Alias curto, saída customizada, 10 amostras plotadas
python tests/infer.py -n data.npz -m model.onnx -o ./saida -s 10

# Calcular métricas em TODO o dataset e plotar 5 amostras
python tests/infer.py -n data.npz -m model.onnx --all -s 5

# Threshold de binarização customizado
python tests/infer.py -n data.npz -m model.tflite --threshold 0.4
```

| Argumento | Padrão | Descrição |
|---|---|---|
| `--npz` / `-n` | obrigatório | Arquivo NPZ com arrays `X` (e opcionalmente `Y`) |
| `--model` / `-m` | obrigatório | Modelo (`.keras`, `.h5`, `.onnx`, `.tflite`) |
| `--output` / `-o` | `<dir NPZ>/inference_results/` | Diretório de saída |
| `--samples` / `-s` | `5` | Amostras a plotar individualmente |
| `--all` / `-a` | `false` | Calcular métricas em todo o dataset |
| `--threshold` / `-t` | `0.5` | Limiar de binarização da predição |
| `--batch` / `-b` | `32` | Batch size para inferência |

**Saída:**
```
<output>/<model_stem>_sample_<i>.png    — comparação individual (Input | Target | Previsão)
<output>/<model_stem>_summary.png       — histogramas de distribuição das métricas
```

**Métricas calculadas:**

| Métrica | Descrição |
|---|---|
| IoU (Jaccard) | Sobreposição entre predição e ground-truth |
| BCE+Dice Loss | Loss combinada (requer ground-truth) |
| Continuidade | Penaliza caminhos fragmentados |
| Segmentos | Score por número de segmentos desconexos |
| Alcançabilidade | 1.0 se o caminho conecta início ao fim |
