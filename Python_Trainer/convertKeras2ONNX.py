import re

import numpy as np
import onnx
import tensorflow as tf
import tf2onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantType,
    quantize_static,
)

from Metrics import continuity_metric, iou_metric, path_quality_metric


def _parse_dims(results_path):
    """Extrai largura e altura do nome do diretório de resultados."""
    match = re.search(r'W(\d+)xH(\d+)', results_path)
    W = int(match.group(1))
    H = int(match.group(2))
    return W, H


class PathDataReader(CalibrationDataReader):
    """Fornece amostras de calibração para a quantização estática."""

    def __init__(self, calibration_data, input_name):
        self.data = iter([
            {input_name: img[np.newaxis, ...].astype(np.float32)}
            for img in calibration_data
        ])

    def get_next(self):
        return next(self.data, None)


def convert(results_path):
    W, H = _parse_dims(results_path)
    student_checkpoint = f'{results_path}/student_distilled'

    # 1. Carrega o modelo Keras com as métricas customizadas
    model = tf.keras.models.load_model(
        f'{student_checkpoint}.keras',
        custom_objects={
            'iou_metric': iou_metric,
            'continuity_metric': continuity_metric,
            'path_quality_metric': path_quality_metric,
        },
    )

    # 2. Prepara o modelo para conversão (nomes de saída e shape explícito)
    model.output_names = ['output']
    model.build([None, H, W, 3])

    # 3. Converte de Keras para ONNX (opset 13 para compatibilidade ampla)
    input_signature = [
        tf.TensorSpec([None, H, W, 3], tf.float32, name='entrada_mapa')
    ]
    onnx_model, _ = tf2onnx.convert.from_keras(
        model, input_signature, opset=13
    )
    onnx_path = f'{student_checkpoint}.onnx'
    onnx.save(onnx_model, onnx_path)
    print(f'ONNX salvo em: {onnx_path}')

    # 4. Carrega dados de teste para calibração da quantização
    X_test = np.load(f'{results_path}/X_test.npy')
    dr = PathDataReader(X_test, 'entrada_mapa')

    # 5. Quantização estática INT8 — reduz tamanho e latência de inferência
    quantized_path = f'{student_checkpoint}_quantized_static.onnx'
    quantize_static(
        model_input=onnx_path,
        model_output=quantized_path,
        calibration_data_reader=dr,
        per_channel=False,
        reduce_range=False,
        weight_type=QuantType.QInt8,
    )
    print(f'ONNX quantizado salvo em: {quantized_path}')

    return quantized_path


if __name__ == '__main__':
    convert('./results/result_W064xH064_D00_S000000_E005000')
