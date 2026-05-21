import numpy as np
import tensorflow as tf
import tf2onnx
import onnx
from onnxruntime.quantization import quantize_static, quantize_dynamic, QuantType, CalibrationDataReader
from iou_metric import iou_metric

# Regex
import re

# Isolar informações do caminho do arquivo
def get_W_H_D(text):
    padrao_regex = r'W(\d+)xH(\d+)_D(\d+)'

    match = re.search(padrao_regex, results_path)
    W = int(match.group(1))
    H = int(match.group(2))
    D = int(match.group(3))

    return W, H, D

results_path = f'./results/result_W064xH064_D00_S000000_E005000'
checkpoint_filepath = f'{results_path}/student_distilled'

W, H, D = get_W_H_D(results_path)


# 1. Carregue o modelo que o Checkpoint salvou
model = tf.keras.models.load_model(
    f'{checkpoint_filepath}.keras',
    # Recarregar a métrica customizada)
    custom_objects={'iou_metric': iou_metric}
)

# Precisa disso para a conversão ser possivel
model.output_names = ['output']
model.build([None, 64, 64, 3]) # Depende do UpScalling

input_signature=[tf.TensorSpec([None, H, W, 3], tf.float32, name='entrada_mapa')]
# Use from_function for tf functions
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)
onnx.save(onnx_model, f'{checkpoint_filepath}.onnx')

# Quantization
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

model_path_fp32 = f'{checkpoint_filepath}.onnx'
model_path_int8 = f'{checkpoint_filepath}_quantized_static.onnx'
# Perform dynamic quantization
quantize_static(
    model_input=model_path_fp32,
    model_output=model_path_int8,
    calibration_data_reader=dr,
    per_channel=False,
    reduce_range=False,
    weight_type=QuantType.QInt8
)
