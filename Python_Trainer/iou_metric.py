import numpy as np
from tensorflow import reduce_sum


def iou_metric(y_true, y_pred, smooth=1e-6):
    # Garante que os inputs sejam arrays numpy
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # A lógica permanece a mesma: multiplicação elemento a elemento para interseção
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou

def iou_metric_tf(y_true, y_pred, smooth=1e-6):
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

    intersection = reduce_sum(y_true * y_pred)
    union = reduce_sum(y_true) + reduce_sum(y_pred) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou