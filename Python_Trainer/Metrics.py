import numpy as np
import tensorflow as tf
from scipy.ndimage import label as scipy_label


def iou_metric(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)


def continuity_metric(y_true, y_pred, threshold=0.5, smooth=1e-6):
    """
    Penaliza caminhos fragmentados com base na contagem de extremidades.

    Um caminho único e contínuo tem exatamente 2 extremidades (pixels com 1 vizinho).
    Cada segmento desconectado adiciona 2 extremidades extras; pixels isolados
    (0 vizinhos) equivalem a 2 extremidades cada.

    Pontuação cai como 2 / n_extremidades:
      1 segmento  → 1.00 | 2 → 0.50 | 3 → 0.33 | 4 → 0.25
    """
    y_pred = tf.cast(y_pred, tf.float32)
    pred_binary = tf.cast(y_pred > threshold, tf.float32)

    kernel = tf.constant([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=tf.float32)
    kernel = tf.reshape(kernel, [3, 3, 1, 1])
    neighbors = tf.nn.conv2d(pred_binary, kernel, strides=[1, 1, 1, 1], padding='SAME')

    n_endpoints = tf.reduce_sum(pred_binary * tf.cast(neighbors == 1, tf.float32))
    n_isolated  = tf.reduce_sum(pred_binary * tf.cast(neighbors == 0, tf.float32))
    n_eff = n_endpoints + 2.0 * n_isolated

    has_pixels = tf.cast(tf.reduce_sum(pred_binary) >= 1.0, tf.float32)
    return (2.0 / tf.maximum(2.0, n_eff + smooth)) * has_pixels


def segment_count_metric(y_true, y_pred, threshold=0.5):
    """
    Pontua com base no número de segmentos desconectados.

    Usa a contagem de extremidades como proxy: n_segmentos ≈ n_extremidades / 2.
    Pixels isolados (0 vizinhos) contam como 1 segmento cada.

    Decaimento exponencial — penalidade drástica desde o segundo segmento:
      1 segmento  → exp( 0) = 1.00
      2 segmentos → exp(-1) ≈ 0.37
      3 segmentos → exp(-2) ≈ 0.14
      4 segmentos → exp(-3) ≈ 0.05
    """
    y_pred = tf.cast(y_pred, tf.float32)
    pred_binary = tf.cast(y_pred > threshold, tf.float32)

    kernel = tf.constant([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=tf.float32)
    kernel = tf.reshape(kernel, [3, 3, 1, 1])
    neighbors = tf.nn.conv2d(pred_binary, kernel, strides=[1, 1, 1, 1], padding='SAME')

    n_endpoints = tf.reduce_sum(pred_binary * tf.cast(neighbors == 1, tf.float32))
    n_isolated  = tf.reduce_sum(pred_binary * tf.cast(neighbors == 0, tf.float32))

    # Extremidades efetivas: endpoints normais + 2 por pixel isolado
    n_eff = n_endpoints + 2.0 * n_isolated

    # Estimativa de segmentos (mínimo 1)
    n_segments = tf.maximum(1.0, n_eff / 2.0)

    # Score 0 se nenhum pixel previsto; caso contrário, decaimento exponencial
    has_pixels = tf.cast(tf.reduce_sum(pred_binary) >= 1.0, tf.float32)
    return tf.exp(1.0 - n_segments) * has_pixels


def path_quality_metric(y_true, y_pred, alpha=0.4, beta=0.3):
    """
    Combina IoU, continuidade local e contagem de segmentos.

      alpha → peso do IoU           (padrão 0.4)
      beta  → peso da continuidade  (padrão 0.3)
      1-alpha-beta → peso da contagem de segmentos (padrão 0.3)
    """
    iou  = iou_metric(y_true, y_pred)
    # conn = continuity_metric(y_true, y_pred)
    # seg  = segment_count_metric(y_true, y_pred)
    # return alpha * iou + beta * conn + (1.0 - alpha - beta) * seg
    return iou


def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    return 1.0 - (2.0 * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)


def bce_dice_loss(y_true, y_pred, bce_weight=0.5):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    bce = tf.reduce_mean(bce)
    dice = dice_loss(y_true, y_pred)
    return bce_weight * bce + (1.0 - bce_weight) * dice


def reachability_metric(x_batch, y_pred_batch, threshold=0.5):
    """
    Fração de amostras em que o caminho previsto conecta início ao fim.

    Usa componentes conectados (scipy) para verificar se o pixel de início
    e o pixel de fim pertencem ao mesmo segmento na predição binarizada.
    Retorna valor entre 0.0 (nenhuma amostra conectada) e 1.0 (todas conectadas).

    Não é uma Keras metric — chame manualmente após predict().
    """
    scores = []
    for i in range(len(x_batch)):
        start_yx = np.argwhere(x_batch[i, :, :, 1] > 0.5)
        end_yx   = np.argwhere(x_batch[i, :, :, 2] > 0.5)
        if not len(start_yx) or not len(end_yx):
            scores.append(0.0)
            continue
        mask = (y_pred_batch[i].squeeze() > threshold).astype(int)
        labeled, _ = scipy_label(mask)
        label_start = labeled[start_yx[0][0], start_yx[0][1]]
        label_end   = labeled[end_yx[0][0],   end_yx[0][1]]
        scores.append(float(label_start != 0 and label_start == label_end))
    return float(np.mean(scores))
