"""
quantize_onnx.py — Exemplo de quantização dinâmica e estática (INT8) de modelos ONNX
usando o ONNX Runtime Quantization Tool.

Este script serve como referência e ponto de partida para quantizar manualmente um
modelo ONNX exportado pelo pipeline. Para quantização integrada ao pipeline, use
ConvertKeras2ONNX.py, que já executa a quantização estática automaticamente.

Como usar:
    1. Edite `results_path` para apontar para a pasta que contém o modelo ONNX.
    2. Edite `checkpoint_filepath` para o nome base do checkpoint.
    3. Execute:
           python utils/quantize_onnx.py

    Pré-requisito: o arquivo <checkpoint>.onnx deve existir na pasta de resultados.

Saídas geradas:
    <checkpoint>_dynamic_quantized.onnx   — quantização dinâmica (mais rápida de gerar)
    <checkpoint>_static_quantized.onnx    — quantização estática INT8 com calibração

Diferença entre os modos:
    Dinâmica : pesos quantizados em tempo de compilação; ativações em tempo de execução.
               Não requer dados de calibração. Bom ponto de partida.
    Estática : pesos E ativações quantizados com base em amostras reais (X_test.npy).
               Menor tamanho e maior velocidade em hardware dedicado (NPU/ARM).
               Requer CalibrationDataReader com amostras representativas.

Nota sobre PathDataReader:
    Cada amostra precisa da dimensão de batch explícita (1, H, W, C). O input_name
    'entrada_mapa' deve corresponder ao nome do tensor de entrada do seu modelo ONNX
    (verificar com: session.get_inputs()[0].name).
"""

import onnx
from onnxruntime.quantization import quantize_static, quantize_dynamic, QuantType, CalibrationDataReader
import numpy as np

results_path = f'./results/result_W064xH064_D01_S000000_E005000'
checkpoint_filepath = f'{results_path}/best_path_finder_Unet1_1'

model_fp32_path = f'{checkpoint_filepath}.onnx'
model_int8_path = f'{checkpoint_filepath}_dynamic_quantized.onnx'

# Perform dynamic quantization
quantize_dynamic(
    model_input=model_fp32_path,
    model_output=model_int8_path
    # per_channel=False,
    # reduce_range=False,
    # weight_type=QuantType.QUInt8
)

data_input_filename = f'{results_path}/X_test.npy'
data_output_filename = f'{results_path}/Y_test.npy'
X_test = np.load(data_input_filename)
Y_test = np.load(data_output_filename)


class PathDataReader(CalibrationDataReader):
    def __init__(self, calibration_data, input_name):
        self.input_name = input_name
        # Criamos uma lista de dicionários onde cada imagem ganha a dimensão de batch (1, 64, 64, 1)
        self.data_list = [
            {input_name: img[np.newaxis, ...].astype(np.float32)} 
            for img in calibration_data
        ]
        self.data = iter(self.data_list)

    def get_next(self):
        return next(self.data, None)
dr = PathDataReader(X_test, 'entrada_mapa')


model_int8_path = f'{checkpoint_filepath}_static_quantized.onnx'
# Perform dynamic quantization
quantize_static(
    model_input=model_fp32_path,
    model_output=model_int8_path,
    calibration_data_reader=dr,
    per_channel=False,
    reduce_range=False,
    weight_type=QuantType.QInt8
)

# print(f"Quantized model saved to {model_int8_path}")

