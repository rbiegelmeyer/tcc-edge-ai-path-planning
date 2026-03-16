
import tensorflow as tf
import numpy as np
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

results_path = './results/result_W064xH064_D01_S000000_E001000/'
model_basename = f'{results_path}/best_path_finder_Unet1'
model_name = f'{model_basename}.tflite'
data_input_filename = f'{results_path}/X_test.npy'
data_output_filename = f'{results_path}/Y_test.npy'

# Métrica crucial para segmentação (IoU)
def iou_metric(y_true, y_pred, smooth=1e-6):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

X_test = np.load(data_input_filename)
Y_test = np.load(data_output_filename)


def run_tflite_inference(X_data):
    """Executa a inferência para um conjunto de dados."""

    # Preparação do Interpretador TFLite
    interpreter = tf.lite.Interpreter(model_path=model_name)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    preds = []
    for i in range(len(X_data)):
        # Preparar input (Batch, H, W, C)
        input_tensor = np.expand_dims(X_data[i], axis=0).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        
        output = interpreter.get_tensor(output_details[0]['index'])
        preds.append(output[0]) # Remove dimensão de batch do resultado
    return np.array(preds)

print("\nAvaliando modelo TFLite no Conjunto de Teste...")
all_predictions = run_tflite_inference(X_test)

# Cálculo manual das métricas
# ious = [iou_metric(Y_test[i], (all_predictions[i] > 0.5).astype(np.float32)) for i in range(len(Y_test))]
# avg_iou = np.mean(ious)


# --- Visualização de Amostras de Teste ---
def visualize_tflite_results(X_data, Y_true, predictions, results_path, num_samples=5):
    """Visualiza o input, o caminho real e a previsão do modelo."""

    images_result = f'{results_path}/test/tflite/'
    print(f"Dados de teste gerados em: {images_result}")
    os.makedirs(images_result, exist_ok=True)

    for i in range(num_samples):
        plt.figure()

        # 1. Input Map (Início, Fim, Obstáculos)
        plt.subplot(1, 3, 1)
        # O squeeze remove a dimensão do canal (50, 100, 1) -> (50, 100)
        plt.imshow(X_data[i].squeeze(), cmap='gray')
        plt.title('Input\n(Início/Fim/Obstáculos)')
        plt.axis('off')

        # 2. Ground Truth (Caminho Real A*)
        plt.subplot(1, 3, 2)
        plt.imshow(Y_true[i].squeeze(), cmap='hot')
        plt.title('Target\n(Caminho Real A*)')
        plt.axis('off')

        # 3. Prediction (Caminho Previsto pela CNN)
        plt.subplot(1, 3, 3)
        # Binarizar a previsão (valores > 0.5 são considerados 'caminho')
        predicted_path = (predictions[i].squeeze() > 0.5).astype(np.float32)
        plt.imshow(predicted_path, cmap='hot')
        plt.title('Previsão\n(Caminho da CNN)')
        plt.axis('off')

        # plt.show()
        plt.savefig(f'{images_result}/mapa_predicao_{i}.png')
        plt.close()  # Libera a memória da figura

# Visualizar uma porcentagem das amostras
num_viz = int(len(X_test) * 0.15)
visualize_tflite_results(X_test, Y_test, all_predictions, results_path, num_samples=num_viz)