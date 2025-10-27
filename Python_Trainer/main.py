import sys
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout, Conv2DTranspose

np.set_printoptions(threshold=sys.maxsize, precision=4, suppress=True)

def preprocess_data(df):
    """
    Transforma os dados brutos do CSV em tensores de Input (X) e Target (Y).
    X: Mapa com Obstáculos, Início (3) e Fim (4).
    Y: Máscara Binária do Caminho A* (1 para caminho, 0 para o resto).
    """
    X_list = []
    Y_list = []

    for index, row in df.iterrows():
        H, W = row['height'], row['width']
        start_x, start_y = row['start_x'], row['start_y']
        end_x, end_y = row['end_x'], row['end_y']
        map_str = row['map']

        # 1. Converter a string do mapa para um array 2D
        # Assumindo que a string tem H*W caracteres e é lida linha por linha
        map_array = np.array(list(map_str), dtype=int).reshape(H, W)

        # 2. Criar a Máscara do Caminho (Target Y)
        # Assumimos que '2' é o marcador do caminho percorrido pelo A*
        path_mask = (map_array == 2).astype(np.float32)
        # print(path_mask)
        Y_list.append(path_mask)

        # 3. Criar o Mapa de Entrada (Input X)
        input_map = map_array.copy()

        # Codificação do Input X:
        # Obstáculo ('1') -> 1
        # Espaço Livre ('0') -> 0
        # Caminho ('2') é apagado ou deixado como 0 (pois é o que a CNN deve prever)
        input_map[input_map == 2] = 0

        # Marcar Início e Fim com valores distintos (codificação one-hot-like)
        # Usamos 3 e 4 para serem normalizados depois
        input_map[start_y, start_x] = 3
        input_map[end_y, end_x] = 4
        
        X_list.append(input_map)

    # Converter para arrays NumPy
    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)

    # Adicionar o canal (necessário para CNNs 2D: [batch, height, width, channels])
    X = np.expand_dims(X, axis=-1)
    Y = np.expand_dims(Y, axis=-1)

    # 4. Normalização (importante)
    # Normaliza o Input X para o range [0, 1]
    # O valor máximo codificado é 4 (para o ponto final)
    X_normalized = X / 4.0

    return X_normalized, Y

# Métrica crucial para segmentação (IoU)
def iou_metric(y_true, y_pred, smooth=1e-6):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

def build_unet(input_size=(64, 64, 1)):
    """Constrói a arquitetura U-Net, otimizada para mapas 2D."""
    
    inputs = Input(input_size)

    #Let's create a function for one step of the encoder block, so as to increase the reusability when making custom unets
    def encoder_block(filters, inputs):
        x = Conv2D(filters, kernel_size = (3,3), padding = 'same', strides = 1, activation = 'relu')(inputs)
        s = Conv2D(filters, kernel_size = (3,3), padding = 'same', strides = 1, activation = 'relu')(x)
        p = MaxPooling2D(pool_size = (2,2), padding = 'same')(s)
        return s, p #p provides the input to the next encoder block and s provides the context/features to the symmetrically opposte decoder block
    #Baseline layer is just a bunch on Convolutional Layers to extract high level features from the downsampled Image
    def baseline_layer(filters, inputs):
        x = Conv2D(filters, kernel_size = (3,3), padding = 'same', strides = 1, activation = 'relu')(inputs)
        x = Conv2D(filters, kernel_size = (3,3), padding = 'same', strides = 1, activation = 'relu')(x)
        return x
    #Decoder Block
    def decoder_block(filters, connections, inputs):
        x = Conv2DTranspose(filters, kernel_size = (2,2), padding = 'same', activation = 'relu', strides = 2)(inputs)
        skip_connections = concatenate([x, connections], axis = -1)
        x = Conv2D(filters, kernel_size = (2,2), padding = 'same', activation = 'relu')(skip_connections)
        x = Conv2D(filters, kernel_size = (2,2), padding = 'same', activation = 'relu')(x)
        return x
    
    #defining the encoder
    s1, p1 = encoder_block(64, inputs = inputs)
    s2, p2 = encoder_block(128, inputs = p1)
    s3, p3 = encoder_block(256, inputs = p2)
    s4, p4 = encoder_block(512, inputs = p3)

    #Setting up the baseline
    baseline = baseline_layer(1024, p4)

    #Defining the entire decoder
    d1 = decoder_block(512, s4, baseline)
    d2 = decoder_block(256, s3, d1)
    d3 = decoder_block(128, s2, d2)
    d4 = decoder_block(64, s1, d3)
    
    # Saída: Um canal de saída, ativado por Sigmoid para máscara binária (0 ou 1)
    output = Conv2D(1, (1, 1), activation='sigmoid')(d4)

    model = Model(inputs=inputs, outputs=output, name='Unet')
    
    # Compilação: Binary Cross-Entropy é ideal para a máscara binária
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy', iou_metric])
    
    return model


df_raw = pd.read_csv('../AStar/result.csv')

# 0 Free
# 1 Obstacle
# 2 Path
# 3 Start
# 4 Goal

# print(df_raw)

X, Y = preprocess_data(df_raw)

# Divisão dos dados
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

np.save('X_test.npy', X_test)
np.save('Y_test.npy', Y_test)


print(f"Shape do X_train (Input): {X_train.shape}")
print(f"Shape do Y_train (Target/Máscara): {Y_train.shape}")
print(f"Valor Máximo em X_train (deve ser ~1.0): {np.max(X_train)}")


# Dimensões do seu mapa
H, W = X_train.shape[1], X_train.shape[2]
model = build_unet(input_size=(H, W, 1))

# model.summary() # Descomente para ver a arquitetura detalhada




from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Definir callbacks de segurança
# 1. Parada Antecipada: Monitora a perda de validação. Se não melhorar por 10 épocas, para.
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

# 2. Checkpoint do Modelo: Salva o melhor modelo que alcançou a menor 'val_loss'.
checkpoint_filepath = './best_path_finder_unet.keras'
model_checkpoint = ModelCheckpoint(checkpoint_filepath, 
                                   monitor='val_loss', 
                                   save_best_only=True, 
                                   verbose=1)

# Treinamento
print("\nIniciando treinamento da CNN (U-Net)...")
history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=100, # Um número razoavelmente alto, EarlyStopping vai parar
    batch_size=8, # Ajustar conforme a memória da GPU
    callbacks=[early_stopping, model_checkpoint],
    verbose=2 # Exibe menos detalhes por época
)

# O melhor modelo está agora salvo em './best_path_finder_unet.keras'


# Correção para o erro de backend Tkinter/GUI
import matplotlib
# Configura o backend para 'Agg' (modo não-interativo/sem GUI)
# Faça isso antes de importar pyplot!
matplotlib.use('Agg') 
# Se o seu código tiver importado pyplot antes, o erro persistirá.
# Portanto, mova esta linha para o topo do seu script.

import matplotlib.pyplot as plt

checkpoint_filepath = './best_path_finder_unet.keras'

# Carregar o melhor modelo para avaliação
best_model = tf.keras.models.load_model(
    checkpoint_filepath, 
    custom_objects={'iou_metric': iou_metric} # Recarregar a métrica customizada
)

# Avaliação final no conjunto de Teste
print("\nAvaliação Final no Conjunto de Teste:")
loss, acc, iou = best_model.evaluate(X_test, Y_test, verbose=0)
print(f"Loss: {loss:.4f} | Acurácia de Pixel: {acc:.4f} | IoU (Jaccard): {iou:.4f}")

# --- Visualização de Amostras de Teste ---
def visualize_results(X_data, Y_true, model, num_samples=3):
    """Visualiza o input, o caminho real e a previsão do modelo."""
    
    predictions = model.predict(X_data[:num_samples])
    
    for i in range(num_samples):
        plt.figure()
        
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