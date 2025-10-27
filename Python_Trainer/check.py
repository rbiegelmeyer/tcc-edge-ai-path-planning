# Correção para o erro de backend Tkinter/GUI
import matplotlib
# Configura o backend para 'Agg' (modo não-interativo/sem GUI)
# Faça isso antes de importar pyplot!
matplotlib.use('Agg') 
# Se o seu código tiver importado pyplot antes, o erro persistirá.
# Portanto, mova esta linha para o topo do seu script.

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


checkpoint_filepath = './best_path_finder_unet.keras'

# Métrica crucial para segmentação (IoU)
def iou_metric(y_true, y_pred, smooth=1e-6):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

# Carregar o melhor modelo para avaliação
best_model = tf.keras.models.load_model(
    checkpoint_filepath, 
    custom_objects={'iou_metric': iou_metric} # Recarregar a métrica customizada
)

X_test = np.load('X_test.npy')
Y_test = np.load('Y_test.npy')

# Avaliação final no conjunto de Teste
print("\nAvaliação Final no Conjunto de Teste:")
loss, acc, iou = best_model.evaluate(X_test, Y_test, verbose=0)
print(f"Loss: {loss:.4f} | Acurácia de Pixel: {acc:.4f} | IoU (Jaccard): {iou:.4f}")

# --- Visualização de Amostras de Teste ---
def visualize_results(X_data, Y_true, model, num_samples=3):
    """Visualiza o input, o caminho real e a previsão do modelo."""
    
    predictions = model.predict(X_data[:num_samples])
    
    for i in range(num_samples):
        plt.figure(figsize=(10, 5))
        
        # 1. Input Map (Início, Fim, Obstáculos)
        plt.subplot(1, 3, 1)
        # O squeeze remove a dimensão do canal (50, 100, 1) -> (50, 100)
        plt.imshow(X_data[i].squeeze(), cmap='gray') 
        plt.title('Input (Início/Fim/Obstáculos)')
        plt.axis('off')

        # 2. Ground Truth (Caminho Real A*)
        plt.subplot(1, 3, 2)
        plt.imshow(Y_true[i].squeeze(), cmap='hot') 
        plt.title('Target (Caminho Real A*)')
        plt.axis('off')

        # 3. Prediction (Caminho Previsto pela CNN)
        plt.subplot(1, 3, 3)
        # Binarizar a previsão (valores > 0.5 são considerados 'caminho')
        predicted_path = (predictions[i].squeeze() > 0.5).astype(np.float32)
        plt.imshow(predicted_path, cmap='hot')
        plt.title('Previsão da CNN (Caminho)')
        plt.axis('off')
        
        # plt.show()
        plt.savefig(f'mapa_predicao_{i}.png')
        plt.close() # Libera a memória da figura

# Visualize as primeiras 5 amostras do conjunto de teste
visualize_results(X_test, Y_test, best_model, num_samples=5)