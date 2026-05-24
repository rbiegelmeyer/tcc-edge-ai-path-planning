
import os
import numpy as np

import tensorflow as tf
import onnxruntime as ort
from metrics import iou_metric, continuity_metric, path_quality_metric


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

results_path = './results/result_W064xH064_D05_S000000_E005000'
model_basename = f'{results_path}/best_path_finder_Unet1'
model_name = f'{model_basename}.tflite'
data_input_filename = f'{results_path}/X_test.npy'
data_output_filename = f'{results_path}/Y_test.npy'

X_test = np.load(data_input_filename)
Y_test = np.load(data_output_filename)

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
    

def run_model(model_path, X_data):
    model_file = os.path.basename(model_path)
    model_type = identificar_modelo(model_path)

    if model_type == "Keras (H5)":
        print(f"Processando modelo Keras: {model_file}")
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'iou_metric': iou_metric}
        )
        predictions = model.predict(X_data)

    elif model_type == "TFLite":
        print(f"Processando modelo TFLite: {model_file}")
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # TFLite não suporta inferência em batch — processa uma amostra por vez
        output_shape = output_details[0]['shape'][1:]  # (H, W, C) sem a dimensão de batch
        predictions = np.empty((len(X_data), *output_shape), dtype=np.float32)
        for i, sample in enumerate(X_data):
            interpreter.set_tensor(input_details[0]['index'], sample[np.newaxis].astype(np.float32))
            interpreter.invoke()
            predictions[i] = interpreter.get_tensor(output_details[0]['index'])[0]

    elif model_type == "ONNX":
        print(f"Processando modelo ONNX: {model_file}")
        # Fallback para CPU caso CUDA não esteja disponível no ambiente
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(model_path, providers=providers)
        input_name = session.get_inputs()[0].name
        # ONNX suporta batch — processa todo o dataset de uma vez
        predictions = session.run(None, {input_name: X_data.astype(np.float32)})[0]

    else:
        return None  # Formato de modelo não suportado

    return predictions

def visualize_tflite_results(X_data, Y_true, model_folder_path, num_samples=5):
    """Visualiza o input, o caminho real e a previsão do modelo."""

    images_result = f'{model_folder_path}/test_models/'
    print(f"Dados de teste gerados em: {images_result}")
    os.makedirs(images_result, exist_ok=True)

    # get only ai models names in the folder
    model_folder_path = os.path.abspath(model_folder_path)
    
    ai_models = [f for f in os.listdir(model_folder_path) if f.endswith(('.keras', '.tflite', '.onnx'))]
    # remove not file
    ai_models = [f for f in ai_models if os.path.isfile(os.path.join(model_folder_path, f))]
    print(f"Modelos encontrados: {ai_models}")

    number_of_models = len(ai_models)

    results_of_models = []

    for j, model_file in enumerate(ai_models):

        model_path = os.path.join(model_folder_path, model_file) 
        predictions = run_model(model_path, X_data)  

        if predictions is None:
            continue

        results_of_models.append({
            'name': model_file,
            'predictions': predictions
        })

    for i in range(num_samples):
        plt.figure()

        for j in range(len(results_of_models)):
            model_path = results_of_models[j]['name']
            predictions = results_of_models[0]['predictions']
            predicted_path = (predictions[i].squeeze() > 0.5).astype(np.float32)

            plt.subplot(number_of_models, 1, j + 1)
            plt.imshow(predicted_path, cmap='hot')
            plt.title(model_path)
            plt.axis('off')

        plt.savefig(f'{images_result}/mapa_predicao_{i}.png')
        plt.close()  # Libera a memória da figura
    


# # --- Visualização de Amostras de Teste ---
# def visualize_tflite_results(X_data, Y_true, keras_results, tflite_results, tflite_quantized_results, onnx_results, onnx_quantized_static_results, results_path, num_samples=5):
#     """Visualiza o input, o caminho real e a previsão do modelo."""

#     images_result = f'{results_path}/test_models/'
#     print(f"Dados de teste gerados em: {images_result}")
#     os.makedirs(images_result, exist_ok=True)

#     for i in range(num_samples):
#         plt.figure()

#         # 1. Input Map (Início, Fim, Obstáculos)
#         # plt.subplot(2, 5, 1)
#         # # O squeeze remove a dimensão do canal (50, 100, 1) -> (50, 100)
#         # plt.imshow(X_data[i].squeeze(), cmap='gray')
#         # # plt.title('Input\n(Início/Fim/Obstáculos)')
#         # plt.title('Input')
#         # plt.axis('off')

#         # 2. Ground Truth (Caminho Real A*)
#         plt.subplot(2, 4, 1)
#         plt.imshow(Y_true[i].squeeze(), cmap='hot')
#         # plt.title('Target\n(Caminho Real A*)')
#         plt.title('Target')
#         plt.axis('off')

#         # 3. Prediction Keras (Caminho Previsto pela CNN)
#         plt.subplot(2, 4, 2)
#         # Binarizar a previsão (valores > 0.5 são considerados 'caminho')
#         predicted_path = (keras_results[i].squeeze() > 0.5).astype(np.float32)
#         plt.imshow(predicted_path, cmap='hot')
#         plt.title('Previsão\nKeras')
#         plt.axis('off')

#         # 3. Prediction TFLite(Caminho Previsto pela CNN)
#         plt.subplot(2, 4, 3)
#         predicted_path = (tflite_results[i].squeeze() > 0.5).astype(np.float32)
#         plt.imshow(predicted_path, cmap='hot')
#         plt.title('Previsão\nTFLite')
#         plt.axis('off')

#         # 3. Prediction TFLite(Caminho Previsto pela CNN)
#         plt.subplot(2, 4, 7)
#         predicted_path = (tflite_quantized_results[i].squeeze() > 0.5).astype(np.float32)
#         plt.imshow(predicted_path, cmap='hot')
#         plt.title('Previsão\nTFLite Quantized')
#         plt.axis('off')

#         # 3. Prediction ONNX (Caminho Previsto pela CNN)
#         plt.subplot(2, 4, 4)
#         predicted_path = (onnx_results[i].squeeze() > 0.5).astype(np.float32)
#         plt.imshow(predicted_path, cmap='hot')
#         plt.title('Previsão\nONNX')
#         plt.axis('off')

#         plt.subplot(2, 4, 8)
#         predicted_path = (onnx_quantized_static_results[i].squeeze() > 0.5).astype(np.float32)
#         plt.imshow(predicted_path, cmap='hot')
#         plt.title('Previsão\nONNX Quantized Static')
#         plt.axis('off')

#         plt.savefig(f'{images_result}/mapa_predicao_{i}.png')
#         plt.close()  # Libera a memória da figura

# Visualizar uma porcentagem das amostras
num_viz = int(len(X_test) * 0.15)
X_test = X_test[0:num_viz]




visualize_tflite_results(X_test, Y_test, results_path, num_viz)