import tensorflow as tf
import tf2onnx
import onnx

# Regex
import re


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

# Isolar informações do caminho do arquivo
def get_W_H_D(text):
    padrao_regex = r'W(\d+)xH(\d+)_D(\d+)'

    match = re.search(padrao_regex, results_path)
    W = int(match.group(1))
    H = int(match.group(2))
    D = int(match.group(3))

    return W, H, D

results_path = f'./results/result_W064xH064_D01_S000000_E001000'
checkpoint_filepath = f'{results_path}/best_path_finder_Unet1_1'

W, H, D = get_W_H_D(results_path)


# 1. Carregue o modelo que o Checkpoint salvou
model = tf.keras.models.load_model(
    f'{checkpoint_filepath}.keras',
    # Recarregar a métrica customizada)
    custom_objects={'iou_metric': iou_metric}
)

# Precisa disso para a conversão ser possivel
model.output_names = ['output']
model.build([None, 64, 64, 1]) # Depende do UpScalling

input_signature=[tf.TensorSpec([None, H, W, 1], tf.float32, name='entrada_mapa')]
# Use from_function for tf functions
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)
onnx.save(onnx_model, f'{checkpoint_filepath}.onnx')
