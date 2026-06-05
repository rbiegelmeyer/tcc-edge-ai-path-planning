
import os
import numpy as np
import tensorflow as tf

# Correção para o erro de backend Tkinter/GUI
import matplotlib
# Configura o backend para 'Agg' (modo não-interativo/sem GUI)
# Faça isso antes de importar pyplot!
matplotlib.use('Agg')
# Se o seu código tiver importado pyplot antes, o erro persistirá.
# Portanto, mova esta linha para o topo do seu script.
import matplotlib.pyplot as plt

results_path = './results/result_W064xH064_D01_S000000_E001000'
model_basename = f'{results_path}/best_path_finder_Unet1'
model_name = f'{model_basename}.keras'
data_input_filename = f'{results_path}/X_test.npy'
data_output_filename = f'{results_path}/Y_test.npy'


# Métrica crucial para segmentação (IoU)
def iou_metric(y_true, y_pred, smooth=1e-6):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

# Carregar o melhor modelo para avaliação
best_model = tf.keras.models.load_model(
    model_name, 
    custom_objects={'iou_metric': iou_metric} # Recarregar a métrica customizada
)

X_test = np.load(data_input_filename)
Y_test = np.load(data_output_filename)

# Avaliação final no conjunto de Teste
print("\nAvaliação Final no Conjunto de Teste:")
loss, acc, iou = best_model.evaluate(X_test, Y_test, verbose=0)
print(f"Loss: {loss:.4f} | Acurácia de Pixel: {acc:.4f} | IoU (Jaccard): {iou:.4f}")



# --- Visualização de Amostras de Teste ---
def visualize_results(X_data, Y_true, model, num_samples=3):
    """Visualiza o input, o caminho real e a previsão do modelo."""

    predictions = model.predict(X_data[:num_samples])
    images_result = f'{results_path}/test/{model.name}'
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

# Visualize as primeiras 5 amostras do conjunto de teste
num_samples = int(len(X_test) * 0.15)
visualize_results(X_test, Y_test, best_model, num_samples=num_samples)