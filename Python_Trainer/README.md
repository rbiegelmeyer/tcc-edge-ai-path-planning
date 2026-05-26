# Path Finder — Treinamento e Implantação em STM32

Pipeline de treinamento de redes neurais convolucionais para predição de caminhos em mapas de grade, com destilação de conhecimento e exportação para microcontroladores STM32.

---

## Visão Geral

O objetivo é treinar um modelo U-Net capaz de, dado um mapa com obstáculos e posições de início e fim, predizer o caminho ótimo equivalente ao calculado pelo algoritmo A*. O modelo professor (maior, mais preciso) é depois destilado num modelo aluno (menor, adequado para embarcados), que é quantizado e implantado em STM32 via STM32 Cube AI Studio.

```
CSV (mapas A*)
      │
      ▼
  [Coach.py]          → Treina modelo professor (U-Net completo)
      │
      ▼
  [Distiller.py]      → Destila para modelo aluno (U-Net reduzido)
      │
      ▼
  [ConvertKeras2ONNX] → Exporta para ONNX + quantização INT8 estática
      │
      ▼
  [validate_device]   → Valida no STM32H743 / STM32N657 via stedgeai
```

---

## Estrutura de Arquivos

```
Python_Trainer/
│
├── pipeline.py            # Ponto de entrada principal — executa tudo em sequência
├── Coach.py               # Treinamento do modelo professor (U-Net)
├── Distiller.py           # Destilação de conhecimento (professor → aluno)
├── ConvertKeras2ONNX.py   # Conversão Keras → ONNX + quantização INT8
├── ConvertKeras2TFLite.py # Conversão Keras → TFLite (alternativa)
│
├── Metrics.py             # Métricas customizadas e função de loss
├── Visualizer.py          # Geração de imagens de resultado
├── compare_results.py     # Colagem comparativa professor vs aluno
├── validate_device.py     # Validação dos modelos ONNX no STM32
│
└── results/               # Gerado automaticamente
    └── result_<dataset>/
        ├── X_train.npy / Y_train.npy
        ├── X_val.npy   / Y_val.npy
        ├── X_test.npy  / Y_test.npy
        ├── path_finder_Unet.keras          # Checkpoint do professor
        ├── student_distilled.keras         # Checkpoint do aluno
        ├── student_distilled_quantized_static.onnx
        └── test/
            ├── Unet/                       # Visualizações do professor
            ├── distilled_Student_Unet/     # Visualizações do aluno
            └── comparacao_teacher_student/ # Comparações lado a lado
```

---

## Formato dos Dados de Entrada

Os dados de entrada são arquivos `.csv` gerados pelo gerador A* (`../AStar/`). Cada linha representa um mapa:

| Coluna    | Descrição                                      |
|-----------|------------------------------------------------|
| `map`     | String com os valores de cada célula concatenados |
| `width`   | Largura do mapa em pixels                      |
| `height`  | Altura do mapa em pixels                       |
| `start_x` / `start_y` | Posição de início                |
| `end_x`   / `end_y`   | Posição de destino               |
| `difficulty` | Nível de dificuldade do mapa              |

Valores de célula no mapa: `0` = livre, `1` = obstáculo, `2` = caminho A*.

A rede recebe entrada com **3 canais** por pixel:
- Canal 0: obstáculos (0 ou 1)
- Canal 1: pixel de início (1 somente na célula de início)
- Canal 2: pixel de fim (1 somente na célula de destino)

---

## Requisitos

```bash
pip install tensorflow onnx tf2onnx onnxruntime scikit-learn pandas matplotlib pillow scipy
```

Para validação no STM32, é necessário o **STM32 Cube AI Studio** instalado com o executável `stedgeai` acessível. Configure o caminho em `validate_device.py`:

```python
STEDGEAI_PATH = r"C:\ST\STEdgeAI\4.0\Utilities\windows\stedgeai.exe"
```

---

## Tutorial de Uso

### 1. Pipeline Completa (recomendado)

Edite a lista `DATASETS` em `pipeline.py` com os arquivos CSV desejados e execute:

```bash
cd Python_Trainer
python pipeline.py
```

A pipeline executa automaticamente treinamento → destilação → conversão ONNX para cada dataset.

> Para rodar só o treinamento (sem destilação/conversão), mantenha o `continue` na linha correspondente em `pipeline.py`.

---

### 2. Execução por Etapa

#### Etapa 1 — Treinamento do Professor

```bash
python Coach.py
```

Ou ajuste o dataset no bloco `if __name__ == '__main__':` ao final do arquivo. Ao concluir, gera:
- `results/<dataset>/path_finder_Unet.keras`
- Imagens de predição em `results/<dataset>/test/Unet/`
- Gráficos de treinamento em `results/<dataset>/test/Unet/graficos_treinamento.png`

#### Etapa 2 — Destilação do Aluno

```bash
python Distiller.py
```

Requer o checkpoint do professor gerado na etapa anterior. Gera:
- `results/<dataset>/student_distilled.keras`
- Imagens em `results/<dataset>/test/distilled_Student_Unet/`

#### Etapa 3 — Conversão para ONNX

```bash
python ConvertKeras2ONNX.py
```

Gera o modelo ONNX com quantização INT8 estática, pronto para implantação.

#### Etapa 4 — Comparação Visual

```bash
python compare_results.py --results ./results/result_W064xH064_D01_S000000_E005000
```

Gera imagens comparativas empilhando professor (acima) e aluno (abaixo) para cada mapa de teste. Saída em `test/comparacao_teacher_student/`.

Opções:
```bash
# Especificar subpastas manualmente
python compare_results.py \
    --results ./results/result_W064xH064_D01_S000000_E005000 \
    --teacher Unet \
    --student distilled_Student_Unet
```

#### Etapa 5 — Validação no STM32

```bash
# Valida todos os modelos quantizados em todos os boards
python validate_device.py

# Somente no host (sem hardware físico)
python validate_device.py --host-only

# Board específico
python validate_device.py --board stm32h743

# Modelo específico
python validate_device.py --model ./results/.../student_distilled_quantized_static.onnx
```

Gera um relatório JSON em `validation_report.json`.

---

## Métricas

Definidas em `Metrics.py`:

| Métrica | Descrição |
|--------|-----------|
| `iou_metric` | Interseção sobre União entre predição e alvo |
| `continuity_metric` | Penaliza caminhos fragmentados pelo número de extremidades |
| `segment_count_metric` | Decaimento exponencial por segmento desconectado |
| `path_quality_metric` | Combinação ponderada das três anteriores (IoU 40%, continuidade 30%, segmentos 30%) |
| `reachability_metric` | Fração de mapas onde o caminho previsto conecta início ao fim (pós-treino) |
| `bce_dice_loss` | Loss de treinamento: BCE + Dice Loss combinados (50/50) |

A `reachability_metric` é a métrica mais direta do objetivo real — ela usa componentes conectados para verificar se o caminho previsto realmente vai do ponto A ao ponto B.

---

## Modelos

### Professor — `Unet`
U-Net completo com 4 níveis de encoder/decoder, filtros 64→128→256→512→1024. Adequado para treinamento em GPU. Tamanho: ~30 MB.

### Aluno — `Student_Unet`
U-Net reduzido com 4 níveis, filtros 16→32→64→128→256. Otimizado para inferência embarcada. Tamanho: ~1.6 MB. Treinado via destilação de conhecimento usando o professor como supervisão soft.

---

## Configuração dos Boards STM32

Edite o dicionário `BOARDS` em `validate_device.py`:

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

O `workspace` deve ser um projeto previamente configurado no STM32 Cube AI Studio para o board correspondente.
