"""
Converte modelos Keras para TFLite (float32 e INT8 quantizado).

Uso:
    python ConvertKeras2TFLite.py --results ./results/W064xH064_D03_S000000_E025000
    python ConvertKeras2TFLite.py --results ./results/... --model teacher
"""

import argparse
import os

import numpy as np
import tensorflow as tf

from Metrics import bce_dice_loss, iou_metric

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def _representative_data_gen(X_cal, n_samples=200):
    """Gera amostras de calibração para quantização estática."""
    n = min(n_samples, len(X_cal))
    def gen():
        for i in range(n):
            yield [X_cal[i:i+1].astype(np.float32)]
    return gen


def convert(results_path, model_type='teacher'):
    """
    Converte modelo Keras para TFLite float32 e INT8.

    Args:
        results_path: pasta de resultados com os checkpoints e dados .npy
        model_type:   'teacher' (path_finder_Unet) ou 'student' (student_distilled)
    """
    if model_type == 'teacher':
        checkpoint = f'{results_path}/path_finder_Unet.keras'
        prefix     = f'{results_path}/path_finder_Unet'
    else:
        checkpoint = f'{results_path}/student_distilled.keras'
        prefix     = f'{results_path}/student_distilled'

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f'Checkpoint não encontrado: {checkpoint}')

    print(f'\nCarregando modelo: {checkpoint}')
    model = tf.keras.models.load_model(
        checkpoint,
        custom_objects={'bce_dice_loss': bce_dice_loss, 'iou_metric': iou_metric},
    )
    model.summary()

    X_val = np.load(f'{results_path}/X_val.npy')

    # ------------------------------------------------------------------
    # 1. TFLite float32
    # ------------------------------------------------------------------
    print('\n[1/2] Convertendo para TFLite float32...')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    tflite_path = f'{prefix}.tflite'
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    size_kb = os.path.getsize(tflite_path) / 1024
    print(f'  Salvo: {tflite_path}  ({size_kb:.1f} KB)')

    # ------------------------------------------------------------------
    # 2. TFLite INT8 quantizado (quantização estática)
    # ------------------------------------------------------------------
    print('\n[2/2] Convertendo para TFLite INT8 (quantização estática)...')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations          = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = _representative_data_gen(X_val)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type   = tf.int8
    converter.inference_output_type  = tf.int8
    tflite_quant = converter.convert()

    quant_path = f'{prefix}_int8.tflite'
    with open(quant_path, 'wb') as f:
        f.write(tflite_quant)
    size_kb = os.path.getsize(quant_path) / 1024
    print(f'  Salvo: {quant_path}  ({size_kb:.1f} KB)')

    return tflite_path, quant_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Conversão Keras → TFLite')
    parser.add_argument('--results', required=True,
                        help='Pasta de resultados (ex: ./results/W064xH064_D03_...)')
    parser.add_argument('--model', choices=['teacher', 'student'], default='teacher',
                        help='Modelo a converter (padrão: teacher)')
    args = parser.parse_args()

    convert(args.results, model_type=args.model)
