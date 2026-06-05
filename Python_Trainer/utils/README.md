# utils/

Utilitários de suporte ao pipeline: conversão de datasets, geração de headers C para
firmware embarcado, quantização ONNX manual e visualização de dumps binários do STM32.

Execute os scripts a partir do diretório `Python_Trainer/`:

```bash
cd Python_Trainer
python utils/<script>.py [argumentos]
```

---

## Scripts

### `csv_to_npz.py`

Converte um dataset CSV de mapas para o formato **NPZ comprimido**, pronto para uso
pelo `infer.py`, `compare_models.py` e `npy_to_header.py`.

**Formato esperado do CSV** (colunas obrigatórias):
`id, difficulty, start_x, start_y, end_x, end_y, height, width, map`

O campo `map` é uma string flat de `H×W` dígitos (0–4):
- `0` = vazio, `1` = parede, `2` = caminho (removido), `3` = início, `4` = fim

**Arrays gerados no NPZ:**

| Array | Shape | Dtype | Descrição |
|---|---|---|---|
| `x` | `(N, 64, 64, 3)` | float32 | Canal 0=obstáculos, 1=início, 2=fim |
| `ids` | `(N,)` | int32 | IDs originais dos mapas |
| `start_x`, `start_y` | `(N,)` | int16 | Coordenadas do ponto de início |
| `end_x`, `end_y` | `(N,)` | int16 | Coordenadas do ponto de fim |

**CLI:**

```bash
# Converte todo o CSV para NPZ (mesmo nome, extensão .npz)
python utils/csv_to_npz.py dataset.csv

# Saída customizada e limite de amostras
python utils/csv_to_npz.py dataset.csv --out validacao.npz --max 500

# Exemplo real
python utils/csv_to_npz.py W064xH064_D03_S100000_E100100.csv
```

| Argumento | Padrão | Descrição |
|---|---|---|
| `csv` | obrigatório | Arquivo CSV de entrada |
| `--out` | `<csv>.npz` | Arquivo NPZ de saída |
| `--max` | todos | Número máximo de amostras |

---

### `npy_to_header.py`

Converte um array `.npy` ou `.npz` de mapas para um **header C** (`x_val_data.h`)
para embed dos dados de validação no firmware do microcontrolador.

**Shape esperado:** `(N, 64, 64, 3)` — compatível com a saída de `csv_to_npz.py`.

**Conversão de dtype:**
- `float32` / `float64` [0, 1] → `int8` com escala 1/255 e zero-point −128
- `uint8` [0, 255] → `int8` (subtrai 128)
- `int8` → sem conversão

**Saída — estrutura do header:**
```c
#define X_VAL_COUNT      50
#define X_VAL_ITEM_SIZE  12288   // 64 * 64 * 3

static const int8_t x_val_data[50][12288] = {
    { /* mapa 0 */ },
    { /* mapa 1 */ },
    ...
};
```

**CLI:**

```bash
# A partir de .npy
python utils/npy_to_header.py X_val.npy

# A partir de .npz com chave explícita, limitando a 40 amostras
python utils/npy_to_header.py data.npz --key x --max 40 --out ./x_val_data.h

# Saída em caminho customizado
python utils/npy_to_header.py X_val.npy --out ./firmware/x_val_data.h
```

| Argumento | Padrão | Descrição |
|---|---|---|
| `npy` | obrigatório | Arquivo `.npy` ou `.npz` de entrada |
| `--key` | auto-detecta | Chave do array dentro de um `.npz` |
| `--out` | `./x_val_data.h` | Arquivo `.h` de saída |
| `--max` | todos | Número máximo de amostras |

> **Atenção Flash:** 40 amostras ≈ 480 KB. Com mais de ~115 amostras o arquivo
> ultrapassa 1.4 MB e pode não caber junto com os pesos na Flash do MCU.

---

### `csv_to_header.py`

Converte um dataset CSV de mapas para um **header C** com os mapas e metadados para
validação do algoritmo **A\*** no firmware.

**Saída — estrutura do header:**
```c
#define TEST_MAP_COUNT   50
#define TEST_MAP_HEIGHT  64
#define TEST_MAP_WIDTH   64
#define TEST_MAP_SIZE    (TEST_MAP_HEIGHT * TEST_MAP_WIDTH)

// Células: 0=vazio 1=parede 2=caminho 3=início 4=fim
static const int8_t test_maps[50][4096] = { ... };

typedef struct {
    uint32_t id;
    int16_t  start_row, start_col;
    int16_t  end_row,   end_col;
} TestMapMeta;

static const TestMapMeta test_maps_meta[50] = { ... };
```

> Inclua este header em **somente um** arquivo `.c` para evitar duplicação de dados
> na memória Flash.

**CLI:**

```bash
# Todos os mapas do CSV
python utils/csv_to_header.py W064xH064_D03_S100000_E100100.csv

# Primeiros 50 mapas, com skip dos 10 primeiros
python utils/csv_to_header.py dataset.csv -o test_maps.h -n 50 --skip 10

# Mapas específicos por índice de linha (0-based)
python utils/csv_to_header.py dataset.csv --ids 0 5 12 37

# Include guard customizado
python utils/csv_to_header.py dataset.csv --guard MY_TEST_MAPS_H
```

| Argumento | Padrão | Descrição |
|---|---|---|
| `csv` | obrigatório | Arquivo CSV de entrada |
| `--output` / `-o` | `<csv>_maps.h` | Arquivo `.h` de saída |
| `--count` / `-n` | todos | Número de mapas a exportar |
| `--skip` | `0` | Pular os primeiros N mapas |
| `--ids` | — | Índices de linha específicos (0-based) — ignora `--skip`/`--count` |
| `--guard` | derivado do nome | Nome do include guard |

---

### `bin_to_map.py`

Converte um **dump binário int8** de saída do modelo no STM32 em um heatmap PNG.
Visualiza os valores raw `[-128, 127]` com escala de temperatura (OrRd).

Opcionalmente sobrepõe o mapa de entrada (paredes, início e fim) sobre o heatmap.

**CLI:**

```bash
# Arquivo padrão (../AStar/temp.bin)
python utils/bin_to_map.py

# Arquivo de resultado específico
python utils/bin_to_map.py result.bin

# Heatmap com overlay do mapa de entrada
python utils/bin_to_map.py result.bin --input input.bin

# Grade não-quadrada
python utils/bin_to_map.py result.bin --input input.bin --height 32 --width 32
```

| Argumento | Padrão | Descrição |
|---|---|---|
| `bin` | `../AStar/temp.bin` | Arquivo `.bin` de saída do modelo |
| `--input` / `-i` | — | Mapa de entrada com valores 0–4 (paredes/início/fim) |
| `--height` / `-H` | `64` | Altura do mapa em células |
| `--width` / `-W` | `64` | Largura do mapa em células |

**Saída:**
```
<nome_do_bin>_map.png          — heatmap simples
<nome_do_bin>_map_overlay.png  — heatmap com overlay (se --input fornecido)
```

**Detecção automática do mapa de entrada:**
O arquivo `--input` pode ser um dump de memória maior; o script varre o arquivo
em busca de um bloco `H×W` bytes cujos valores estejam em `{0,1,2,3,4}` e que
contenha células de início (3) e fim (4).

---

### `quantize_onnx.py`

Script de **referência** para quantização manual de modelos ONNX usando o
`onnxruntime.quantization`. Demonstra dois modos:

| Modo | Saída | Calibração |
|---|---|---|
| Dinâmica | `*_dynamic_quantized.onnx` | Não requer dados |
| Estática INT8 | `*_static_quantized.onnx` | Requer amostras (`X_test.npy`) |

> Para uso integrado ao pipeline, prefira `ConvertKeras2ONNX.py`, que já
> executa a quantização estática automaticamente.

**Como usar:**

1. Edite as variáveis no topo do arquivo:
   ```python
   results_path       = './results/result_W064xH064_D01_S000000_E005000'
   checkpoint_filepath = f'{results_path}/best_path_finder_Unet1_1'
   ```
2. Execute:
   ```bash
   python utils/quantize_onnx.py
   ```

**Nota:** O `input_name` no `PathDataReader` (`'entrada_mapa'`) deve corresponder
ao nome do tensor de entrada do seu modelo. Verifique com:
```python
import onnxruntime as ort
sess = ort.InferenceSession("modelo.onnx")
print(sess.get_inputs()[0].name)
```
