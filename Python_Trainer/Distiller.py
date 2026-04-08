import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout, Conv2DTranspose
from tensorflow.keras.models import Model

import matplotlib.pyplot as plt

class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student

    def compile(self, optimizer, metrics, student_loss_fn, distillation_loss_fn, alpha=0.1, temperature=3):
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha # Peso para a perda do aluno vs destilação
        self.temperature = temperature

    def train_step(self, data):
        x, y = data

        # 1. Inferência do Professor (não treinamos o professor aqui)
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # 2. Inferência do Aluno
            student_predictions = self.student(x, training=True)

            # 3. Loss padrão (Aluno vs Gabarito Real)
            student_loss = self.student_loss_fn(y, student_predictions)

            # 4. Loss de Destilação (Aluno vs Professor com Temperatura)
            # Aplicamos a suavização nas previsões
            distillation_loss = self.distillation_loss_fn(
                tf.nn.sigmoid(teacher_predictions / self.temperature),
                tf.nn.sigmoid(student_predictions / self.temperature),
            )

            # Loss Final: Equilíbrio entre aprender o real e imitar o mestre
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # 5. Atualização dos pesos do Aluno
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Atualizar métricas
        for metric in self.metrics:
            metric.update_state(y, student_predictions)

        return {m.name: m.result() for m in self.metrics}
    

def build_simplified_unet(input_size):
    """Constrói a arquitetura U-Net, otimizada para mapas 2D."""

    inputs = Input(input_size)

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

    # defining the encoder
    s1, p1 = encoder_block(16, inputs=inputs)
    s2, p2 = encoder_block(32, inputs=p1)
    s3, p3 = encoder_block(64, inputs=p2)
    s4, p4 = encoder_block(128, inputs=p3)

    # Setting up the baseline
    baseline = baseline_layer(256, p4)

    # Defining the entire decoder
    d1 = decoder_block(128, s4, baseline)
    d2 = decoder_block(64, s3, d1)
    d3 = decoder_block(32, s2, d2)
    d4 = decoder_block(16, s1, d3)

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




results_path = f'./results/result_W064xH064_D01_S000000_E005000'
checkpoint_filepath = f'{results_path}/best_path_finder_Unet1.keras'

data_test_input_filename = f'{results_path}/X_test.npy'
data_test_output_filename = f'{results_path}/Y_test.npy'
data_train_input_filename = f'{results_path}/X_train.npy'
data_train_output_filename = f'{results_path}/Y_train.npy'

X_test = np.load(data_test_input_filename)
Y_test = np.load(data_test_output_filename)
X_train = np.load(data_train_input_filename)
Y_train = np.load(data_train_output_filename)

H, W = X_train.shape[1], X_train.shape[2]

# Carregar o melhor modelo para avaliação
best_model = tf.keras.models.load_model(
    checkpoint_filepath,
    # Recarregar a métrica customizada
    custom_objects={'iou_metric': iou_metric}
)

# Construção da U-Net Simplificada
student_model = build_simplified_unet(input_size=(H, W, 1))

distiller = Distiller(student=student_model, teacher=best_model)

distiller.compile(
    optimizer=keras.optimizers.Adam(),
    metrics=[iou_metric],
    student_loss_fn=keras.losses.BinaryCrossentropy(),
    distillation_loss_fn=keras.losses.KLDivergence(), # KL Divergence é padrão para destilação
    alpha=0.1,
    temperature=5
)

# Treinamento
distiller.fit(X_train, Y_train, epochs=30)

distiller.student.save(f'{results_path}/student_distilled.keras')


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

num_samples = int(len(X_test) * 0.15)
visualize_results(X_test, Y_test, student_model, num_samples=num_samples, prefixe='distilled_')