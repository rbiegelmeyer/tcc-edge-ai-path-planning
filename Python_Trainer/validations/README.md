# validations/

Scripts de análise, comparação visual e geração de diagramas para os resultados do
pipeline. Ao contrário dos scripts em `tests/`, estes não calculam métricas de
acurácia em tempo real — foco em **visualização e análise post-hoc**.

Execute a partir do diretório `Python_Trainer/`:

```bash
cd Python_Trainer
python validations/<script>.py [argumentos]
```

---

## Scripts

### `compare_models.py`

Comparação visual com **4 painéis** por amostra:

```
[ A* Ground Truth ]  [ Modelo 1 ]  [ Modelo 2 ]  [ Heatmap Embarcado ]
```

O painel do heatmap embarcado exibe o dump binário int8 do STM32 (saída real do
modelo quantizado rodando no hardware), com overlay de paredes, início e fim.

**CLI:**

```bash
python validations/compare_models.py \
    --npz     data.npz \
    --model1  unet.keras     --name1 "U-Net D4" \
    --model2  student.onnx   --name2 "Student INT8" \
    --result  output.bin     --input-bin input.bin  --name-bin "STM32H743" \
    -o ./comparativo  -c 10

# Amostras específicas por índice
python validations/compare_models.py --npz data.npz -m1 m1.onnx -m2 m2.onnx --ids 0 5 42

# Layout em grade 2×2
python validations/compare_models.py --npz data.npz -m1 m1.onnx -m2 m2.onnx --layout square
```

| Argumento | Padrão | Descrição |
|---|---|---|
| `--npz` / `-n` | obrigatório | NPZ com `X` (e opcionalmente `Y`) |
| `--model1` / `-m1` | obrigatório | Modelo 1 (`.keras`/`.h5`/`.onnx`/`.tflite`) |
| `--model2` / `-m2` | obrigatório | Modelo 2 |
| `--name1`, `--name2` | `"Modelo 1/2"` | Título dos painéis dos modelos |
| `--result` / `-r` | — | Binário com heatmap embarcado (int8) |
| `--input-bin` / `-i` | — | Binário do mapa de entrada (valores 0–4) |
| `--name-bin` | `"Embarcado"` | Título do painel heatmap |
| `--name-gt` | `"A* (Ground Truth)"` | Título do painel ground truth |
| `--ids` | — | Índices específicos (substitui `--count`/`--skip`/`--all`) |
| `--count` / `-c` | `5` | Número de amostras |
| `--skip` / `-s` | `0` | Pular os primeiros N índices |
| `--all` / `-a` | `false` | Processar todas as amostras |
| `--output` / `-o` | `./comparativo/` | Diretório de saída |
| `--prefix` | `compare` | Prefixo dos arquivos PNG |
| `--layout` / `-l` | `horizontal` | Layout: `horizontal`, `vertical`, `square` |
| `--no-ticks` | `false` | Ocultar ticks do grid |
| `--threshold` / `-t` | `0.5` | Limiar de binarização |
| `--batch` / `-b` | `32` | Batch size para inferência |
| `--height` / `-H` | `64` | Altura do mapa (para leitura dos binários) |
| `--width` / `-W` | `64` | Largura do mapa (para leitura dos binários) |

**Saída:**
```
<output>/compare_<idx:04d>.png
```

Cada imagem exibe IoU e Alcançabilidade no título de cada painel de modelo quando
ground-truth está disponível.

---

### `compare_results.py`

Gera imagens comparativas entre o **modelo professor (teacher)** e o **modelo aluno
(student)** a partir das imagens já geradas pelos scripts de teste.

Requer que os scripts de teste (`test_keras_model.py` ou equivalentes) já tenham
sido executados para ambos os modelos.

**Modos:**

| Modo | Descrição | Saída |
|---|---|---|
| `vertical` (padrão) | Professor acima, Aluno abaixo, com faixas coloridas | `test/comparacao_teacher_student/comparacao_<i>.png` |
| `4painel` | Input \| Alvo Real \| Pred. Teacher \| Pred. Student lado a lado | `test/comparacao_4paineis/comparacao_4paineis_<i>.png` |

**CLI:**

```bash
# Modo vertical (padrão), auto-detecta pasta do student
python validations/compare_results.py \
    --results ./results/result_W064xH064_D01_S000000_E005000

# Modo 4 painéis com nomes explícitos
python validations/compare_results.py \
    --results ./results/result_W064xH064_D01_S000000_E005000 \
    --teacher Unet \
    --student distilled_Student_Unet \
    --modo 4painel
```

| Argumento | Padrão | Descrição |
|---|---|---|
| `--results` | obrigatório | Pasta de resultado do pipeline |
| `--teacher` | `Unet` | Subpasta do teacher em `test/` |
| `--student` | auto-detecta `distilled_*` | Subpasta do student em `test/` |
| `--modo` | `vertical` | Modo de comparação: `vertical` ou `4painel` |

---

### `analyze_training_results.py`

Visualiza os resultados da **busca de hiperparâmetros** (grade `sfilter × depth`)
a partir do arquivo `training_log.csv`.

**Modos:**

| Modo | Descrição |
|---|---|
| `--dataset NOME` | Gráficos de IoU e Val Loss por profundidade (depth), uma linha por sfilter |
| `--compare` | Evolução do melhor resultado à medida que o dataset cresce |

**Formato esperado do `training_log.csv`** (colunas mínimas):

```
dataset, sfilter, depth, total, best_val_loss, best_val_iou_metric
```

**CLI:**

```bash
# Análise de um dataset específico
python validations/analyze_training_results.py \
    --dataset W064xH064_D03_S000000_E050000

# Comparação entre todos os datasets
python validations/analyze_training_results.py --compare

# CSV alternativo e saída customizada
python validations/analyze_training_results.py \
    --dataset W064xH064_D03_S000000_E025000 \
    --csv meu_log.csv \
    --output resultado.png
```

| Argumento | Padrão | Descrição |
|---|---|---|
| `--dataset` | — | Nome do dataset a analisar (mutuamente exclusivo com `--compare`) |
| `--compare` | — | Compara evolução entre todos os datasets |
| `--csv` | `validations/training_log.csv` | Caminho do CSV de log |
| `--output` | `validations/results/<dataset>/hyperparameter_analysis.png` | Saída da imagem |

---

### `best_model_comparison.py`

Seleciona o **melhor modelo** (menor `val_loss`) de cada dataset e plota dois gráficos
comparativos mostrando o benefício de aumentar o volume de dados de treinamento.

**Saídas:**
```
validations/results/best_model_comparison_iou.png    — IoU de validação × amostras
validations/results/best_model_comparison_loss.png   — Loss de validação × amostras
```

Cada ponto é anotado com `sfilter`, `depth` e o valor da métrica. Percentuais de
melhoria entre datasets consecutivos são exibidos entre os pontos.

**CLI:**

```bash
python validations/best_model_comparison.py

# CSV e caminhos de saída alternativos
python validations/best_model_comparison.py \
    --csv training_log.csv \
    --output-iou iou_comparativo.png \
    --output-loss loss_comparativo.png
```

---

### `compare_maps.py`

Gera **colagens lado a lado** comparando o mapa esperado (imagens BMP do visualizador
SDL2 do STM32) com o mapa atingido pelo MCU (PNGs gerados pelo `plot_result.py`).

**Convenção de nomes:**

| Pasta | Padrão de arquivo | Descrição |
|---|---|---|
| `--esperados` | `<id>.bmp` | Resultado esperado (SDL2) |
| `--atingido` | `map_solved_<id>_plot.png` | Resultado do MCU (`plot_result.py`) |

O `id` do arquivo BMP menos o `--id-offset` deve corresponder ao `id` do PNG.

**CLI:**

```bash
# Pastas padrão (esperados/, atingido/, saída: comparativo/)
python validations/compare_maps.py

# Pastas customizadas e offset
python validations/compare_maps.py \
    --esperados ./sdl_bmps \
    --atingido  ./mcu_plots \
    --output    ./colagens \
    --id-offset 100000 \
    --gap 12
```

| Argumento | Padrão | Descrição |
|---|---|---|
| `--esperados` | `esperados` | Pasta com BMPs SDL2 |
| `--atingido` | `atingido` | Pasta com PNGs do MCU (`map_solved_*_plot.png`) |
| `--output` | `comparativo` | Pasta de saída |
| `--id-offset` | `100000` | Offset: `id_bmp - offset = id_atingido` |
| `--gap` | `8` | Espaço em pixels entre as duas imagens na colagem |

---

### `plot_result.py`

Converte um **dump binário** do STM32 (`.bin`) em um plot do mapa de resultado A\*,
usando a paleta **Catppuccin Mocha** (igual ao visualizador SDL2).

Auto-detecta o bloco de mapa `64×64` dentro do dump procurando o primeiro bloco alinhado
em 4 bytes com valores em `{0,1,2,3,4}` que contenha caminho (2), início (3) e fim (4).

**CLI:**

```bash
# Arquivo padrão (../AStar/matrix_data.bin)
python validations/plot_result.py

# Arquivo de resultado específico
python validations/plot_result.py result.bin

# Com overlay de paredes do dump de entrada
python validations/plot_result.py result.bin input.bin
```

**Saída:**
```
<nome_do_bin>_plot.png
```

**Codificação de células:**
```
0 = vazio   1 = parede   2 = caminho   3 = início   4 = fim
```

---

### `generate_architecture_diagram.py`

Gera um diagrama de **arquitetura da U-Net** em alta resolução, mostrando cada operação,
conexões skip, setas de fluxo encoder/decoder e legenda.

```bash
# Saída padrão: unet_architecture.png (no diretório atual)
python validations/generate_architecture_diagram.py

# Caminho de saída customizado
python validations/generate_architecture_diagram.py --output diagrama_unet.png
```

| Argumento | Padrão | Descrição |
|---|---|---|
| `--output` | `unet_architecture.png` | Caminho do PNG de saída |

O diagrama mostra:
- **Encoder** (esquerda): 4 níveis com 32→64→128→256 filtros
- **Bottleneck** (centro): 512 filtros
- **Decoder** (direita): 4 níveis espelhados com conexões skip tracejadas
- **Legenda** com código de cores por tipo de operação
