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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose
from tensorflow.keras.models import Model

from sklearn.model_selection import train_test_split

from Metrics import iou_metric, continuity_metric, path_quality_metric

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['AUTOGRAPH_VERBOSITY'] = '0'
os.environ['TF_CPP_MAX_VLOG_LEVEL'] = '0'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import logging
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
        map_array = np.frombuffer(str(row.map).zfill(H * W).encode(), dtype=np.uint8) - ord('0')
        map_array = map_array.reshape(H, W)

        X[i, :, :, 0] = (map_array == MAP_OBSTACLE)
        X[i, row.start_y, row.start_x, 1] = 1.0
        X[i, row.end_y,   row.end_x,   2] = 1.0

        Y[i, :, :, 0] = (map_array == MAP_PATH)

    return X, Y, difficulty


def encoder_block(filters, inputs):
    x = Conv2D(filters, kernel_size=(3, 3), padding='same', strides=1, activation='relu')(inputs)
    x = Conv2D(filters, kernel_size=(3, 3), padding='same', strides=1, activation='relu')(x)
    p = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    return x, p

def baseline_layer(filters, inputs):
    x = Conv2D(filters, kernel_size=(3, 3), padding='same', strides=1, activation='relu')(inputs)
    x = Conv2D(filters, kernel_size=(3, 3), padding='same', strides=1, activation='relu')(x)
    return x

def decoder_block(filters, connections, inputs):
    x = Conv2DTranspose(filters, kernel_size=(2, 2), padding='same', activation='relu', strides=2)(inputs)
    x = concatenate([x, connections], axis=-1)
    x = Conv2D(filters, kernel_size=(2, 2), padding='same', activation='relu')(x)
    x = Conv2D(filters, kernel_size=(2, 2), padding='same', activation='relu')(x)
    return x

def build_unet(input_size):
    inputs = Input(input_size)

    s1, p1 = encoder_block(64,  inputs=inputs)
    s2, p2 = encoder_block(128, inputs=p1)
    s3, p3 = encoder_block(256, inputs=p2)
    s4, p4 = encoder_block(512, inputs=p3)

    baseline = baseline_layer(1024, p4)

    d1 = decoder_block(512, s4, baseline)
    d2 = decoder_block(256, s3, d1)
    d3 = decoder_block(128, s2, d2)
    d4 = decoder_block(64,  s1, d3)

    output = Conv2D(1, kernel_size=(1, 1), activation='sigmoid')(d4)
    model = Model(inputs=inputs, outputs=output, name='Unet')
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', iou_metric, continuity_metric, path_quality_metric])
    model.summary()
    return model

def build_simplified_unet(input_size):
    inputs = Input(input_size)

    s1, p1 = encoder_block(64,  inputs=inputs)
    s2, p2 = encoder_block(128, inputs=p1)

    baseline = baseline_layer(256, p2)

    d3 = decoder_block(128, s2, baseline)
    d4 = decoder_block(64,  s1, d3)

    output = Conv2D(1, kernel_size=(1, 1), activation='sigmoid')(d4)
    model = Model(inputs=inputs, outputs=output, name='Light_Unet')
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', iou_metric, continuity_metric, path_quality_metric])
    model.summary()
    return model


# %%
# Execução

def train(df_filename):
    # --- Dados ---
    df_raw = pd.read_csv(df_filename)
    df_raw = df_raw.drop_duplicates(subset='map', keep='first')
    X, Y, _ = preprocess_data(df_raw)

    TRAIN_RATIO, TEST_RATIO = 0.70, 0.15
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=1 - TRAIN_RATIO, random_state=42)
    X_val, X_test, Y_val, Y_test     = train_test_split(X_temp, Y_temp, test_size=TEST_RATIO / (1 - TRAIN_RATIO), random_state=42)

    print(f'Treino:    {len(X_train):5} ({len(X_train)/len(X)*100:.1f}%)')
    print(f'Validação: {len(X_val):5} ({len(X_val)/len(X)*100:.1f}%)')
    print(f'Teste:     {len(X_test):5} ({len(X_test)/len(X)*100:.1f}%)')

    H, W = X_train.shape[1], X_train.shape[2]

    # --- Paths ---
    basename = os.path.basename(df_filename).split('.')[0]
    results_path = f'./results/{basename}'
    os.makedirs(results_path, exist_ok=True)

    np.save(f'{results_path}/X_train.npy', X_train)
    np.save(f'{results_path}/Y_train.npy', Y_train)
    np.save(f'{results_path}/X_val.npy',   X_val)
    np.save(f'{results_path}/Y_val.npy',   Y_val)
    np.save(f'{results_path}/X_test.npy',  X_test)
    np.save(f'{results_path}/Y_test.npy',  Y_test)

    # --- Modelo ---
    model = build_unet(input_size=(H, W, 3))

    checkpoint_filepath = f'{results_path}/path_finder_{model.name}.keras'
    early_stopping = EarlyStopping(monitor='val_path_quality_metric', patience=50,
                                   verbose=1, restore_best_weights=True, mode='max')
    model_checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_path_quality_metric',
                                       save_best_only=True, verbose=0, mode='max')

    # --- Treinamento ---
    print(f'\nIniciando treinamento: {basename}')
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=1000,
        batch_size=64,
        callbacks=[early_stopping, model_checkpoint],
        verbose=2,
    )

    best_model = tf.keras.models.load_model(
        checkpoint_filepath,
        custom_objects={'iou_metric': iou_metric,
                        'continuity_metric': continuity_metric,
                        'path_quality_metric': path_quality_metric}
    )

    # --- Avaliação ---
    print('\nAvaliação Final no Conjunto de Teste:')
    loss, acc, iou, cont, quality = best_model.evaluate(X_test, Y_test, verbose=1)
    print(f'Loss: {loss:.4f} | Acurácia: {acc:.4f} | IoU: {iou:.4f} | Continuidade: {cont:.4f} | Quality: {quality:.4f}')

    _visualize_results(X_test, Y_test, best_model,
                       num_samples=max(1, int(len(X_test) * 0.15)),
                       results_path=results_path)
    _plot_training_results(history, model.name, basename, results_path)

    return results_path, checkpoint_filepath


def _visualize_results(X_data, Y_true, model, num_samples, results_path, prefixe=''):
    predictions = model.predict(X_data[:num_samples])
    images_result = f'{results_path}/test/{prefixe}{model.name}'
    os.makedirs(images_result, exist_ok=True)

    for i in range(num_samples):
        _, axes = plt.subplots(1, 3)
        axes[0].imshow(X_data[i, :, :, 0], cmap='gray')
        axes[0].set_title('Input\n(Obstáculos)')
        axes[1].imshow(Y_true[i].squeeze(), cmap='hot')
        axes[1].set_title('Target\n(Caminho Real)')
        axes[2].imshow((predictions[i].squeeze() > 0.5).astype(np.float32), cmap='hot')
        axes[2].set_title('Previsão\n(CNN)')
        for ax in axes:
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(f'{images_result}/mapa_predicao_{i}.png')
        plt.close()


def _plot_training_results(history, model_name, basename, results_path):
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    specs = [
        ('path_quality_metric', 'val_path_quality_metric', 'Path Quality (IoU + Continuidade)', 'Score'),
        ('iou_metric',          'val_iou_metric',          f'IoU por Época ({basename})',        'IoU'),
        ('continuity_metric',   'val_continuity_metric',   'Continuidade por Época',             'Score'),
        ('loss',                'val_loss',                'Loss por Época',                     'Binary Crossentropy'),
    ]

    for ax, (train_key, val_key, title, ylabel) in zip(axes, specs):
        ax.plot(history.history[train_key], label='Treino',    color='blue',   linewidth=2)
        ax.plot(history.history[val_key],   label='Validação', color='orange', linewidth=2)
        ax.set_title(title)
        ax.set_xlabel('Épocas'); ax.set_ylabel(ylabel)
        ax.legend(); ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    images_result = f'{results_path}/test/{model_name}'
    os.makedirs(images_result, exist_ok=True)
    plt.savefig(f'{images_result}/graficos_treinamento.png', dpi=300)
    plt.close()


if __name__ == '__main__':
    train('../AStar/result_W064xH064_D02_S000000_E100000.csv')
