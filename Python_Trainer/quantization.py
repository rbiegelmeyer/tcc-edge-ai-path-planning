import onnx
from onnxruntime.quantization import quantize_static, quantize_dynamic, QuantType, CalibrationDataReader
import numpy as np

results_path = f'./results/result_W064xH064_D01_S000000_E001000'
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

data_input_filename = f'{results_path}/X_test.npy'
data_output_filename = f'{results_path}/Y_test.npy'
X_test = np.load(data_input_filename)
Y_test = np.load(data_output_filename)

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

