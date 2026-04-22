
import os
import numpy as np

import tensorflow as tf
import onnxruntime as ort
from iou_metric import iou_metric


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

results_path = './results/result_W064xH064_D01_S000000_E005000'
model_basename = f'{results_path}/best_path_finder_Unet1'
model_name = f'{model_basename}.tflite'
data_input_filename = f'{results_path}/X_test.npy'
data_output_filename = f'{results_path}/Y_test.npy'

X_test = np.load(data_input_filename)
Y_test = np.load(data_output_filename)

def run_keras_inference(X_data):

    keras_model_path = f'{model_basename}.keras'

    # Carregar o melhor modelo para avaliação
    model = tf.keras.models.load_model(
        keras_model_path,
        custom_objects={'iou_metric': iou_metric} # Recarregar a métrica customizada
    )

    predictions = model.predict(X_data)

    # Avaliação final no conjunto de Teste
    # print("\nAvaliação Final no Conjunto de Teste:")
    # loss, acc, iou = model.evaluate(X_test, Y_test, verbose=0)
    # print(f"Loss: {loss:.4f} | Acurácia de Pixel: {acc:.4f} | IoU (Jaccard): {iou:.4f}")

    return predictions


def run_tflite_inference(X_data):
    """Executa a inferência para um conjunto de dados."""

    tflite_model_path = f'{model_basename}.tflite'

    # Preparação do Interpretador TFLite
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
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

def run_tflite_quantized_inference(X_data):
    """Executa a inferência para um conjunto de dados."""

    tflite_model_path = f'{model_basename}_quantized.tflite'

    # Preparação do Interpretador TFLite
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
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

def run_onnx_inference(X_data):
    # 1. Criar a sessão de inferência
    # Para GPU, use providers=['CUDAExecutionProvider']
    onnx_model_path = f'{model_basename}.onnx'
    session = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider'])

    # 2. Descobrir os nomes e formatos de entrada (input)
    input_name = session.get_inputs()[0].name
    # input_shape = session.get_inputs()[0].shape
    # print(f"Nome da entrada: {input_name}, Formato: {input_shape}")

    # output_name = session.get_outputs()[0].name
    # output_shape = session.get_outputs()[0].shape
    # print(f"Nome da saída: {output_name}, Formato: {output_shape}")

    # 4. Rodar a inferência
    # print("Rodando a inferência...")
    outputs = session.run(None, {input_name: X_test})

    return outputs[0]

def run_onnx_quantized_static_inference(X_data):
    # 1. Criar a sessão de inferência
    # Para GPU, use providers=['CUDAExecutionProvider']
    onnx_model_path = f'{model_basename}_quantized_static.onnx'
    session = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider'])

    # 2. Descobrir os nomes e formatos de entrada (input)
    input_name = session.get_inputs()[0].name
    # input_shape = session.get_inputs()[0].shape
    # print(f"Nome da entrada: {input_name}, Formato: {input_shape}")

    # output_name = session.get_outputs()[0].name
    # output_shape = session.get_outputs()[0].shape
    # print(f"Nome da saída: {output_name}, Formato: {output_shape}")

    # 4. Rodar a inferência
    # print("Rodando a inferência...")
    outputs = session.run(None, {input_name: X_test})

    return outputs[0]

def identificar_modelo(caminho):
    with open(caminho, 'rb') as f:
        header = f.read(100) # Lê os primeiros 100 bytes
        
    if b'HDF5' in header:
        return "Keras (H5)"
    elif b'TFL3' in header:
        return "TFLite"
    elif b'onnx' in header.lower() or b'ir_version' in header.lower():
        return "ONNX"
    else:
        return "Formato desconhecido"

def visualize_tflite_results(X_data, Y_true, model_path, position=(0, 0, 0), results_path, num_samples=5):
    """Visualiza o input, o caminho real e a previsão do modelo."""
    images_result = f'{results_path}/test_models/'
    print(f"Dados de teste gerados em: {images_result}")
    os.makedirs(images_result, exist_ok=True)

    model_type = identificar_modelo(model_path)
    if model_type == "Keras (H5)":
        model = tf.keras.models.load_model(
            keras_model_path,
            custom_objects={'iou_metric': iou_metric} # Recarregar a métrica customizada
        )
        predictions = model.predict(X_data)

    elif model_type == "TFLite":
        interpreter = tf.lite.Interpreter(model_path=model_path)    # Carrega o modelo TFLite
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

        predictions = np.array(preds)

    elif model_type == "ONNX":
        session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])  # Carrega o modelo ONNX
        input_name = session.get_inputs()[0].name
        predictions = session.run(None, {input_name: X_test})[0]
        
    else:
        return 0



# --- Visualização de Amostras de Teste ---
def visualize_tflite_results(X_data, Y_true, keras_results, tflite_results, tflite_quantized_results, onnx_results, onnx_quantized_static_results, results_path, num_samples=5):
    """Visualiza o input, o caminho real e a previsão do modelo."""

    images_result = f'{results_path}/test_models/'
    print(f"Dados de teste gerados em: {images_result}")
    os.makedirs(images_result, exist_ok=True)

    for i in range(num_samples):
        plt.figure()

        # 1. Input Map (Início, Fim, Obstáculos)
        # plt.subplot(2, 5, 1)
        # # O squeeze remove a dimensão do canal (50, 100, 1) -> (50, 100)
        # plt.imshow(X_data[i].squeeze(), cmap='gray')
        # # plt.title('Input\n(Início/Fim/Obstáculos)')
        # plt.title('Input')
        # plt.axis('off')

        # 2. Ground Truth (Caminho Real A*)
        plt.subplot(2, 4, 1)
        plt.imshow(Y_true[i].squeeze(), cmap='hot')
        # plt.title('Target\n(Caminho Real A*)')
        plt.title('Target')
        plt.axis('off')

        # 3. Prediction Keras (Caminho Previsto pela CNN)
        plt.subplot(2, 4, 2)
        # Binarizar a previsão (valores > 0.5 são considerados 'caminho')
        predicted_path = (keras_results[i].squeeze() > 0.5).astype(np.float32)
        plt.imshow(predicted_path, cmap='hot')
        plt.title('Previsão\nKeras')
        plt.axis('off')

        # 3. Prediction TFLite(Caminho Previsto pela CNN)
        plt.subplot(2, 4, 3)
        predicted_path = (tflite_results[i].squeeze() > 0.5).astype(np.float32)
        plt.imshow(predicted_path, cmap='hot')
        plt.title('Previsão\nTFLite')
        plt.axis('off')

        # 3. Prediction TFLite(Caminho Previsto pela CNN)
        plt.subplot(2, 4, 7)
        predicted_path = (tflite_quantized_results[i].squeeze() > 0.5).astype(np.float32)
        plt.imshow(predicted_path, cmap='hot')
        plt.title('Previsão\nTFLite Quantized')
        plt.axis('off')

        # 3. Prediction ONNX (Caminho Previsto pela CNN)
        plt.subplot(2, 4, 4)
        predicted_path = (onnx_results[i].squeeze() > 0.5).astype(np.float32)
        plt.imshow(predicted_path, cmap='hot')
        plt.title('Previsão\nONNX')
        plt.axis('off')

        plt.subplot(2, 4, 8)
        predicted_path = (onnx_quantized_static_results[i].squeeze() > 0.5).astype(np.float32)
        plt.imshow(predicted_path, cmap='hot')
        plt.title('Previsão\nONNX Quantized Static')
        plt.axis('off')

        plt.savefig(f'{images_result}/mapa_predicao_{i}.png')
        plt.close()  # Libera a memória da figura

# Visualizar uma porcentagem das amostras
num_viz = int(len(X_test) * 0.15)
X_test = X_test[0:num_viz]

print("\nAvaliando modelo Keras no Conjunto de Teste...")
keras_results = run_keras_inference(X_test)

print("\nAvaliando modelo TFLite no Conjunto de Teste...")
tflite_results = run_tflite_inference(X_test)

print("\nAvaliando modelo TFLite Quantized no Conjunto de Teste...")
tflite_quantized_results = run_tflite_quantized_inference(X_test)

print("\nAvaliando modelo ONNX no Conjunto de Teste...")
onnx_results = run_onnx_inference(X_test)

print("\nAvaliando modelo ONNX Quantized Static no Conjunto de Teste...")
onnx_quantized_static_results = run_onnx_quantized_static_inference(X_test)

print("\nAvaliando modelo Student ONNX Quantized Static no Conjunto de Teste...")
onnx_quantized_static_results = run_onnx_quantized_static_inference(X_test)

visualize_tflite_results(
    X_test, Y_test, 
    keras_results,
    tflite_results, tflite_quantized_results,
    onnx_results, onnx_quantized_static_results,
    results_path, num_viz)