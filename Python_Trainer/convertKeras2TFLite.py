import os
import numpy as np
import tensorflow as tf
from iou_metric import iou_metric

# Evita logs detalhados do driver da GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


results_path = f'./results/result_W064xH064_D01_S000000_E005000'
checkpoint_filepath = f'{results_path}/best_path_finder_Unet1'

# Sample training data.
data_input_filename = f'{results_path}/X_test.npy'
data_output_filename = f'{results_path}/Y_test.npy'
X_test = np.load(data_input_filename)
Y_test = np.load(data_output_filename)

def representative_data_gen():
    # Usa 100 amostras para calibrar as escalas da rede
    for i in range(100):
        data = np.expand_dims(X_test[i], axis=0)
        yield [data]

# 1. Carregue o modelo que o Checkpoint salvou
model = tf.keras.models.load_model(
    f'{checkpoint_filepath}.keras',
    # Recarregar a métrica customizada)
    custom_objects={'iou_metric': iou_metric}
)

# Convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open(f'{checkpoint_filepath}.tflite', 'wb') as f:
  f.write(tflite_model)


converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
tflite_model = converter.convert()

# Save the model.
with open(f'{checkpoint_filepath}_quantized.tflite', 'wb') as f:
  f.write(tflite_model)
