# %%
# Imports e setup
import os
import sys
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import tensorflow as tf
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# from keras import layers
# import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Dropout, Conv2DTranspose
from tensorflow.keras.models import Model

from sklearn.model_selection import train_test_split


# 0 = Todos os logs (padrão)
# 1 = Filtra logs INFO
# 2 = Filtra logs INFO e WARNING
# 3 = Filtra logs INFO, WARNING e ERROR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Evita logs detalhados do driver da GPU
os.environ['AUTOGRAPH_VERBOSITY'] = '0'

# Desativa a inicialização duplicada de plugins XLA/CUDA (o "pulo do gato")
os.environ['TF_CPP_MAX_VLOG_LEVEL'] = '0'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import logging
# Desativa avisos do absl (usado internamente pelo TF)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

np.set_printoptions(threshold=sys.maxsize, precision=4, suppress=True)



MAP_OBSTACLE = 1
MAP_PATH     = 2

def preprocess_data(df):
    H, W = df['height'].iloc[0], df['width'].iloc[0]
    difficulty = df['difficulty'].iloc[0]
    n = len(df)

    X = np.zeros((n, H, W, 3), dtype=np.float32)  # [obstáculos, início, fim]
    Y = np.zeros((n, H, W, 1), dtype=np.float32)

    for i, row in enumerate(df.itertuples(index=False)):
        map_array = np.frombuffer(row.map.encode(), dtype=np.uint8) - ord('0')
        map_array = map_array.reshape(H, W)

        X[i, :, :, 0] = (map_array == MAP_OBSTACLE)        # canal obstáculos
        X[i, row.start_y, row.start_x, 1] = 1.0            # canal início
        X[i, row.end_y,   row.end_x,   2] = 1.0            # canal fim

        Y[i, :, :, 0] = (map_array == MAP_PATH)

    return X, Y, difficulty

def iou_metric(y_true, y_pred, smooth=1e-6):
    """
    Calcula a métrica Intersection over Union (IoU), também conhecida como Índice de Jaccard.

        Esta função avalia a sobreposição entre a máscara real (y_true) e a predição do modelo (y_pred).
        É a métrica padrão para problemas de segmentação e planejamento de caminhos, pois penaliza
        tanto os pixels não detectados (falsos negativos) quanto os pixels detectados incorretamente 
        (falsos positivos).

    Args:
        y_true: Ground truth (gabarito real).
        y_pred: Valores preditos pelo modelo (geralmente após ativação Sigmoid).
        smooth: Pequena constante para evitar divisão por zero quando as áreas forem nulas.

    Returns:
        Um valor escalar entre 0 e 1, onde 1 indica uma sobreposição perfeita.
    """
    
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

# Let's create a function for one step of the encoder block, so as to increase the reusability when making custom unets
def encoder_block(filters, inputs):
    x = Conv2D(filters, kernel_size=(3, 3), padding='same', strides=1, activation='relu')(inputs)
    x = Conv2D(filters, kernel_size=(3, 3), padding='same', strides=1, activation='relu')(x)
    p = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    return x, p  # p provides the input to the next encoder block and s provides the context/features to the symmetrically opposte decoder block
# Baseline layer is just a bunch on Convolutional Layers to extract high level features from the downsampled Image

def baseline_layer(filters, inputs):
    x = Conv2D(filters, kernel_size=(3, 3), padding='same', strides=1, activation='relu')(inputs)
    x = Conv2D(filters, kernel_size=(3, 3), padding='same', strides=1, activation='relu')(x)
    return x
# Decoder Block

def decoder_block(filters, connections, inputs):
    x = Conv2DTranspose(filters, kernel_size=(2, 2), padding='same', activation='relu', strides=2)(inputs)
    skip_connections = concatenate([x, connections], axis=-1)
    x = Conv2D(filters, kernel_size=(2, 2), padding='same', activation='relu')(skip_connections)
    x = Conv2D(filters, kernel_size=(2, 2), padding='same', activation='relu')(x)
    return x

def build_unet(input_size):
    """Constrói a arquitetura U-Net, otimizada para mapas 2D."""

    inputs = Input(input_size)

    # defining the encoder
    s1, p1 = encoder_block(64, inputs=inputs)
    s2, p2 = encoder_block(128, inputs=p1)
    s3, p3 = encoder_block(256, inputs=p2)
    s4, p4 = encoder_block(512, inputs=p3)

    # Setting up the baseline
    baseline = baseline_layer(1024, p4)

    # Defining the entire decoder
    d1 = decoder_block(512, s4, baseline)
    d2 = decoder_block(256, s3, d1)
    d3 = decoder_block(128, s2, d2)
    d4 = decoder_block(64, s1, d3)

    # Saída: Um canal de saída, ativado por Sigmoid para máscara binária (0 ou 1)
    output = Conv2D(1, kernel_size=(1, 1), activation='sigmoid')(d4)

    model = Model(inputs=inputs, outputs=output, name='Unet1')
    # model = keras.Sequential([(inputs=inputs, outputs=output, name='Unet')

    # Compilação: Binary Cross-Entropy é ideal para a máscara binária
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', iou_metric])

    model.summary()

    return model

def build_simplified_unet(input_size):
    """Constrói a arquitetura U-Net, otimizada para mapas 2D."""

    inputs = Input(input_size)

    # defining the encoder
    s1, p1 = encoder_block(64, inputs=inputs)
    s2, p2 = encoder_block(128, inputs=p1)

    # Setting up the baseline
    baseline = baseline_layer(256, p2)

    # Defining the entire decoder
    d3 = decoder_block(128, s2, baseline)
    d4 = decoder_block(64, s1, d3)

    # Saída: Um canal de saída, ativado por Sigmoid para máscara binária (0 ou 1)
    output = Conv2D(1, kernel_size=(1, 1), activation='sigmoid')(d4)

    model = Model(inputs=inputs, outputs=output, name='Light_Unet')
    # model = keras.Sequential([(inputs=inputs, outputs=output, name='Unet')

    # Compilação: Binary Cross-Entropy é ideal para a máscara binária
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', iou_metric])

    model.summary()

    return model

# %% Pré-processamento
# df_raw = pd.read_csv('../AStar/result.csv')
df_filename = '../AStar/result_W064xH064_D05_S000000_E005000.csv'
# df_filename = '../AStar/result_W064xH064_D01_S000000_E010000.csv'
df_raw = pd.read_csv(df_filename)
# Remove map duplicados
df_raw = df_raw.drop_duplicates(subset='map', keep='first')

# Processa os dados do dataframe
X, Y, difficulty = preprocess_data(df_raw)

# Divisão dos dados
# Train:        70%
# Test:         15%
# Validation:   15%
X_train, X_temp, Y_train, Y_temp = train_test_split(
    X, Y, test_size=0.3, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(
    X_temp, Y_temp, test_size=0.5, random_state=42)
print(f'Tamanho de treinamento: {len(X_train):5} ({len(Y_train)/len(Y)*100:.2f}%)')
print(f'Tamanho de validação:   {len(X_val):5} ({len(Y_val)/len(Y)*100:.2f}%)')
print(f'Tamanho de teste:       {len(X_test):5} ({len(Y_test)/len(Y)*100:.2f}%)')
print(f"Shape do X_train (Input): {X_train.shape}")
print(f"Shape do Y_train (Target/Máscara): {Y_train.shape}")
print(f"Valor Máximo em X_train (deve ser ~1.0): {np.max(X_train)}")

# Dimensões do seu mapa
H, W = X_train.shape[1], X_train.shape[2]

# Construção da U-Net
model = build_unet(input_size=(H, W, 3))

# Results Path
basename_for_results = os.path.basename(df_filename).split('.')[0]
results_path = f'./results/{basename_for_results}'
print(results_path)
# Cria o diretorio caso não exista
os.makedirs(results_path, exist_ok=True)

np.save(f'{results_path}/X_train.npy', X_train)
np.save(f'{results_path}/Y_train.npy', Y_train)
np.save(f'{results_path}/X_val.npy', X_val)
np.save(f'{results_path}/Y_val.npy', Y_val)
np.save(f'{results_path}/X_test.npy', X_test)
np.save(f'{results_path}/Y_test.npy', Y_test)
# np.savez(f'{results_path}/Z_test.npz', X_test, Y_test)

# %% Treinamento
#########################################
############## Treinamento ##############
#########################################
# Definir callbacks de segurança
# 1. Parada Antecipada: Monitora a perda de validação. Se não melhorar por 10 épocas, para.
early_stopping = EarlyStopping(monitor='val_iou_metric',
                               patience=50,
                               verbose=1,
                               restore_best_weights=True,
                               mode='max')

# 2. Checkpoint do Modelo: Salva o melhor modelo que alcançou a menor 'val_loss'.
checkpoint_filepath = f'{results_path}/path_finder_{model.name}.keras'
model_checkpoint = ModelCheckpoint(checkpoint_filepath,
                                   monitor='val_iou_metric',  # Foca no IoU de validação
                                   save_best_only=True,
                                   verbose=0,
                                   mode='max')

# Treinamento
print("\nIniciando treinamento da CNN (U-Net)...")
history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=1000,  # Um número razoavelmente alto, EarlyStopping vai parar
    batch_size=64,  # Ajustar conforme a memória da GPU
    callbacks=[early_stopping, model_checkpoint],
    verbose=1
)
# Carregar o melhor modelo para avaliação
best_model = tf.keras.models.load_model(
    checkpoint_filepath,
    # Recarregar a métrica customizada
    custom_objects={'iou_metric': iou_metric}
)

# Avaliação final no conjunto de Teste
print("\nAvaliação Final no Conjunto de Teste:")
loss, acc, iou = best_model.evaluate(X_test, Y_test, verbose=1)
print(f"Loss: {loss:.4f} | Acurácia de Pixel: {acc:.4f} | IoU (Jaccard): {iou:.4f}")

# %% Avaliação
# --- Visualização de Amostras de Teste ---
def visualize_results(X_data, Y_true, model, num_samples=3, prefixe=''):
    """Visualiza o input, o caminho real e a previsão do modelo."""

    predictions = model.predict(X_data[:num_samples])
    images_result = f'{results_path}/test/{prefixe}{model.name}'
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


def plot_training_results(history):
    # Criando uma figura com dois subplots lado a lado
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 1. Gráfico de IoU (Métrica Principal)
    ax1.plot(history.history['iou_metric'],
             label='Treino', color='blue', linewidth=2)
    ax1.plot(history.history['val_iou_metric'],
             label='Validação', color='orange', linewidth=2)
    ax1.set_title('Evolução do IoU por Época')
    ax1.set_xlabel('Épocas')
    ax1.set_ylabel('IoU')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 2. Gráfico de Loss (Função de Custo)
    ax2.plot(history.history['loss'], label='Treino',
             color='blue', linewidth=2)
    ax2.plot(history.history['val_loss'],
             label='Validação', color='orange', linewidth=2)
    ax2.set_title('Evolução da Loss por Época')
    ax2.set_xlabel('Épocas')
    ax2.set_ylabel('Binary Crossentropy')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    images_result = f'{results_path}/test/{model.name}'
    plt.savefig(f'{images_result}/graficos_treinamento.png',
                dpi=300)  # Salva para o seu relatório
    plt.close()


# Chame a função após o model.fit()
plot_training_results(history)