import tensorflow as tf


def iou_metric(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)


def continuity_metric(y_true, y_pred, threshold=0.5, smooth=1e-6):
    """
    Mede a proporção de pixels que formam um caminho simples (sem bifurcações).

    Pixels ideais têm 1 ou 2 vizinhos:
      0 vizinhos → isolado (ruim)
      1 vizinho  → extremidade do caminho (ok)
      2 vizinhos → pixel interior contínuo (ideal)
      3+ vizinhos → blob ou bifurcação (ruim)
    """
    y_pred = tf.cast(y_pred, tf.float32)
    pred_binary = tf.cast(y_pred > threshold, tf.float32)

    kernel = tf.constant([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=tf.float32)
    kernel = tf.reshape(kernel, [3, 3, 1, 1])

    neighbors = tf.nn.conv2d(pred_binary, kernel, strides=[1, 1, 1, 1], padding='SAME')

    # Conta apenas pixels com 1 ou 2 vizinhos como "bem conectados"
    well_connected = pred_binary * tf.cast((neighbors >= 1) & (neighbors <= 2), tf.float32)
    total = tf.reduce_sum(pred_binary) + smooth
    return tf.reduce_sum(well_connected) / total


def path_quality_metric(y_true, y_pred, alpha=0.4):
    """Combina IoU (acerto de pixels) com continuidade do caminho."""
    iou  = iou_metric(y_true, y_pred)
    conn = continuity_metric(y_true, y_pred)
    return alpha * iou + (1.0 - alpha) * conn
