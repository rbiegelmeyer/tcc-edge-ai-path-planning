# %%
# Imports e setup
import csv
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

DEVICE = 'GPU'
if DEVICE == 'CPU':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose
from tensorflow.keras.models import Model


from Metrics import iou_metric, continuity_metric, segment_count_metric, path_quality_metric, bce_dice_loss
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

def build_unet(input_size, sfilter=64, depth=4):
    inputs = Input(input_size)

    s = [None] * depth
    p = [None] * depth
    d = [None] * depth

    p0 = inputs

    for i in range(depth):
        s[i], p[i] = encoder_block(sfilter * (2 ** i), p0)
        p0 = p[i]

    p0 = baseline_layer(sfilter * (2 ** depth), p0)

    for i in range(depth - 1, -1, -1):
        d[i] = decoder_block(sfilter * (2 ** i), s[i], p0)
        p0 = d[i]

    # s1, p1 = encoder_block(16,  inputs=inputs)
    # s2, p2 = encoder_block(32, inputs=p1)
    # s3, p3 = encoder_block(64, inputs=p2)
    # s4, p4 = encoder_block(128, inputs=p3)
    # s5, p5 = encoder_block(256, inputs=p4)
    # s6, p6 = encoder_block(512, inputs=p5)

    # baseline = baseline_layer(1024, p6)

    # d6 = decoder_block(512, s6, baseline)
    # d5 = decoder_block(256, s5, d6)
    # d4 = decoder_block(128, s4, d5)
    # d3 = decoder_block(64, s3, d4)
    # d2 = decoder_block(32, s2, d3)
    # d1 = decoder_block(16,  s1, d2)

    output = Conv2D(1, kernel_size=(1, 1), activation='sigmoid')(d[0])
    model = Model(inputs=inputs, outputs=output, name='Unet')
    model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
                  loss=bce_dice_loss,
                  metrics=[iou_metric])
    model.summary()
    return model

# %%
# Execução

def train(df_filename, sfilter=64, depth=4):
    # --- Dados ---
    df_raw = pd.read_csv(df_filename, dtype={'map': str})
    df_raw = df_raw.drop_duplicates(subset='map', keep='first')
    X, Y, _ = preprocess_data(df_raw)

    TRAIN_RATIO, TEST_RATIO = 0.70, 0.15
    rng  = np.random.default_rng(42)
    idx  = rng.permutation(len(X))
    n_train = int(len(X) * TRAIN_RATIO)
    n_test  = int(len(X) * TEST_RATIO)
    X_train, Y_train = X[idx[:n_train]],          Y[idx[:n_train]]
    X_val,   Y_val   = X[idx[n_train:-n_test]],   Y[idx[n_train:-n_test]]
    X_test,  Y_test  = X[idx[-n_test:]],           Y[idx[-n_test:]]

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
    model = build_unet(input_size=(H, W, 3), sfilter=sfilter, depth=depth)

    checkpoint_filepath = f'{results_path}/path_finder_{model.name}.keras'
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=25,
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
    print(f'\n{"="*60}')
    print(f'  Dataset  : {basename}')
    print(f'  Amostras : {len(X)} total  |  {len(X_train)} treino  |  {len(X_val)} val  |  {len(X_test)} teste')
    print(f'  Mapa     : {W}x{H} px')
    print(f'  Modelo   : depth={depth}  sfilter={sfilter}  →  bottleneck={sfilter * (2 ** depth)}f')
    print(f'  Épocas   : {model.optimizer.__class__.__name__}  lr={model.optimizer.learning_rate.numpy():.0e}')
    print(f'{"="*60}')
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

    _save_training_log(
        basename,
        n_total=len(X), n_train=len(X_train), n_val=len(X_val), n_test=len(X_test),
        sfilter=sfilter, depth=depth,
        history=history,
        test_loss=loss, test_iou=iou,
    )

    visualize_results(X_test, Y_test, best_model, results_path,
                      num_samples=max(10, int(len(X_test) * 0.01)),
                      pred_label='Teacher', sfilter=sfilter, depth=depth)
    plot_training_results(history, model.name, basename, results_path, sfilter=sfilter, depth=depth)

    return results_path, checkpoint_filepath


def _save_training_log(basename, n_total, n_train, n_val, n_test,
                       sfilter, depth, history, test_loss, test_iou):
    h            = history.history
    total_epochs = len(h['loss'])
    best_epoch   = int(np.argmin(h['val_loss']))
    best_metrics = {k: float(v[best_epoch]) for k, v in h.items()}

    # --- Imprime resumo da melhor época ---
    metrics_str = '  '.join(
        f'{k}: {v:.4f}' for k, v in sorted(best_metrics.items())
    )
    print(f'\n{"─"*60}')
    print(f'  Melhor época: {best_epoch + 1}/{total_epochs}')
    print(f'  {metrics_str}')
    print(f'{"─"*60}')

    log = {
        'timestamp':    datetime.now().isoformat(timespec='seconds'),
        'dataset':      basename,
        'dataset_size': {'total': n_total, 'train': n_train, 'val': n_val, 'test': n_test},
        'config':       {'sfilter': sfilter, 'depth': depth},
        'training': {
            'total_epochs': total_epochs,
            'best_epoch':   best_epoch + 1,
            'best_metrics': best_metrics,
        },
        'test': {'loss': float(test_loss), 'iou': float(test_iou)},
    }

    # JSON acumulativo no mesmo nível dos scripts
    log_path = os.path.join(os.path.dirname(__file__), 'training_log.json')
    entries = []
    if os.path.exists(log_path):
        with open(log_path, 'r', encoding='utf-8') as f:
            entries = json.load(f)
    entries.append(log)
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)

    # CSV acumulativo (mesmas informações, formato tabular)
    csv_path = os.path.join(os.path.dirname(__file__), 'training_log.csv')
    row = {
        'timestamp':    log['timestamp'],
        'dataset':      basename,
        'total':        n_total,
        'train':        n_train,
        'val':          n_val,
        'test':         n_test,
        'sfilter':      sfilter,
        'depth':        depth,
        'total_epochs': total_epochs,
        'best_epoch':   best_epoch + 1,
        **{f'best_{k}': round(v, 6) for k, v in sorted(best_metrics.items())},
        'test_loss':    round(float(test_loss), 6),
        'test_iou':     round(float(test_iou), 6),
    }
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f'  Log ({len(entries)} entradas): {log_path}')
    print(f'  CSV: {csv_path}')


if __name__ == '__main__':
    train('../AStar/result_W064xH064_D02_S000000_E100000.csv')
