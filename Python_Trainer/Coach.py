# %%
# Imports e setup
import os
import sys
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose
from tensorflow.keras.models import Model

from sklearn.model_selection import train_test_split

from Metrics import iou_metric, continuity_metric, segment_count_metric, path_quality_metric, reachability_metric, bce_dice_loss
from Visualizer import visualize_results, plot_training_results

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
    x = Conv2D(filters, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = Conv2D(filters, kernel_size=(3, 3), padding='same', activation='relu')(x)
    return x

def build_unet(input_size):
    inputs = Input(input_size)

    s1, p1 = encoder_block(32,  inputs=inputs)
    s2, p2 = encoder_block(64, inputs=p1)
    s3, p3 = encoder_block(128, inputs=p2)
    s4, p4 = encoder_block(256, inputs=p3)
    # s5, p5 = encoder_block(1024, inputs=p4)

    # baseline = baseline_layer(2048, p5)
    baseline = baseline_layer(512, p4)

    # d5 = decoder_block(1024, s5, baseline)
    # d4 = decoder_block(512, s4, d5)
    d4 = decoder_block(256, s4, baseline)
    d3 = decoder_block(128, s3, d4)
    d2 = decoder_block(64, s2, d3)
    d1 = decoder_block(32,  s1, d2)

    output = Conv2D(1, kernel_size=(1, 1), activation='sigmoid')(d1)
    model = Model(inputs=inputs, outputs=output, name='Unet')
    model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
                  loss=bce_dice_loss,
                  metrics=[iou_metric])
    model.summary()
    return model

# %%
# Execução

def train(df_filename):
    # --- Dados ---
    df_raw = pd.read_csv(df_filename, dtype={'map': str})
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
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=50,
        verbose=1,
        restore_best_weights=True,
        mode='min'
    )
    model_checkpoint = ModelCheckpoint(
        checkpoint_filepath,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=0
    )
    # --- Treinamento ---
    print(f'\nIniciando treinamento: {basename}')
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=250,
        batch_size=64,
        callbacks=[early_stopping, model_checkpoint],
        verbose=2,
    )

    best_model = tf.keras.models.load_model(
        checkpoint_filepath,
        custom_objects={
            'bce_dice_loss': bce_dice_loss,
            'iou_metric': iou_metric,
            # 'continuity_metric': continuity_metric,
            # 'segment_count_metric': segment_count_metric,
            # 'path_quality_metric': path_quality_metric
        }
    )

    # --- Avaliação ---
    print('\nAvaliação Final no Conjunto de Teste:')
    loss, iou = best_model.evaluate(X_test, Y_test, verbose=1)
    print(f'Loss: {loss:.4f} | IoU: {iou:.4f}')
    # loss, acc, iou, cont, seg, quality = best_model.evaluate(X_test, Y_test, verbose=1)
    # print(f'Loss: {loss:.4f} | Acurácia: {acc:.4f} | IoU: {iou:.4f} | '
    #       f'Continuidade: {cont:.4f} | Segmentos: {seg:.4f} | Quality: {quality:.4f}')

    predictions_test = best_model.predict(X_test, verbose=0)
    reach = reachability_metric(X_test, predictions_test)
    print(f'Alcançabilidade (início→fim): {reach:.4f}')

    visualize_results(X_test, Y_test, best_model, results_path,
                      num_samples=max(1, int(len(X_test) * 0.01)),
                      pred_label= 'Teacher')
    plot_training_results(history, model.name, basename, results_path)

    return results_path, checkpoint_filepath


if __name__ == '__main__':
    train('../AStar/result_W064xH064_D02_S000000_E100000.csv')
