"""
test_onnx_model.py — Avalia um modelo ONNX (.onnx) no conjunto de teste e gera
visualizações comparativas (Input | Target A* | Previsão CNN).

Como usar:
    1. Edite a variável `results_path` para apontar para a pasta do resultado desejado.
    2. Execute diretamente:
           python tests/test_onnx_model.py

    A pasta de resultados deve conter:
        best_path_finder_Unet1.onnx — modelo ONNX exportado
        X_test.npy                  — entradas do conjunto de teste (N, 64, 64, 3)
        Y_test.npy                  — rótulos do conjunto de teste (N, 64, 64, 1)

    Saída:
        <results_path>/test/Unet1_onnx/mapa_predicao_<i>.png
        — imagem com 3 painéis: Input | Target | Previsão binarizada (threshold 0.5)

Execução:
    Por padrão, a sessão ONNX Runtime tenta usar GPU (CUDAExecutionProvider).
    Se CUDA não estiver disponível, adicione 'CPUExecutionProvider' como fallback:
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']

    O modelo ONNX suporta inferência em batch — todo o dataset é processado de uma vez.

Exemplo de results_path:
    './results/result_W064xH064_D04_S000000_E005000'
"""

import os
import onnxruntime as ort
import numpy as np
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
model_name = f'{model_basename}.onnx'
data_input_filename = f'{results_path}/X_test.npy'
data_output_filename = f'{results_path}/Y_test.npy'

# 1. Criar a sessão de inferência
# Para GPU, use providers=['CUDAExecutionProvider']
session = ort.InferenceSession(model_name, providers=['CUDAExecutionProvider'])

# Carregando dados de testes
X_test = np.load(data_input_filename)
Y_test = np.load(data_output_filename)

# 2. Descobrir os nomes e formatos de entrada (input)
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
print(f"Nome da entrada: {input_name}, Formato: {input_shape}")

output_name = session.get_outputs()[0].name
output_shape = session.get_outputs()[0].shape
print(f"Nome da saída: {output_name}, Formato: {output_shape}")

# 4. Rodar a inferência
print("Rodando a inferência...")
outputs = session.run(None, {input_name: X_test})


# --- Visualização de Amostras de Teste ---
def visualize_results(X_data, Y_true, num_samples=3):
    """Visualiza o input, o caminho real e a previsão do modelo."""
    
    # predictions = model.predict(X_data[:num_samples])
    images_result = f'{results_path}/test/Unet1_onnx'
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
        predicted_path = (outputs[0][i].squeeze() > 0.5).astype(np.float32)
        plt.imshow(predicted_path, cmap='hot')
        plt.title('Previsão\n(Caminho da CNN)')
        plt.axis('off')
        
        # plt.show()
        plt.savefig(f'{images_result}/mapa_predicao_{i}.png')
        plt.close() # Libera a memória da figura


# Visualize as primeiras 5 amostras do conjunto de teste
num_samples = int(len(X_test) * 0.15)
visualize_results(X_test, Y_test, num_samples=num_samples)