import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose
from tensorflow.keras.models import Model

from Metrics import iou_metric, continuity_metric, segment_count_metric, path_quality_metric, reachability_metric, bce_dice_loss
from Visualizer import visualize_results


class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student

    def compile(self, optimizer, metrics, student_loss_fn, distillation_loss_fn, alpha=0.1, temperature=3):
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def call(self, x, training=False):
        return self.student(x, training=training)

    def train_step(self, data):
        x, y = data
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            student_predictions = self.student(x, training=True)
            student_loss = self.student_loss_fn(y, student_predictions)
            distillation_loss = self.distillation_loss_fn(
                tf.nn.sigmoid(teacher_predictions / self.temperature),
                tf.nn.sigmoid(student_predictions / self.temperature),
            )
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        gradients = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))

        for metric in self.metrics:
            metric.update_state(y, student_predictions)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        teacher_predictions = self.teacher(x, training=False)
        student_predictions = self.student(x, training=False)

        student_loss = self.student_loss_fn(y, student_predictions)
        distillation_loss = self.distillation_loss_fn(
            tf.nn.sigmoid(teacher_predictions / self.temperature),
            tf.nn.sigmoid(student_predictions / self.temperature),
        )
        loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        for metric in self.metrics:
            metric.update_state(y, student_predictions)
        return {**{m.name: m.result() for m in self.metrics}, "loss": loss}


def build_student(input_size):
    inputs = Input(input_size)

    def encoder_block(filters, x):
        x = Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
        return x, MaxPooling2D((2, 2), padding='same')(x)

    def baseline_layer(filters, x):
        x = Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
        return Conv2D(filters, (3, 3), padding='same', activation='relu')(x)

    def decoder_block(filters, skip, x):
        x = Conv2DTranspose(filters, (2, 2), padding='same', activation='relu', strides=2)(x)
        x = concatenate([x, skip], axis=-1)
        x = Conv2D(filters, (2, 2), padding='same', activation='relu')(x)
        return Conv2D(filters, (2, 2), padding='same', activation='relu')(x)

    s1, p1 = encoder_block(16,  inputs)
    s2, p2 = encoder_block(32,  p1)
    s3, p3 = encoder_block(64,  p2)
    s4, p4 = encoder_block(128, p3)

    base = baseline_layer(256, p4)

    d1 = decoder_block(128, s4, base)
    d2 = decoder_block(64,  s3, d1)
    d3 = decoder_block(32,  s2, d2)
    d4 = decoder_block(16,  s1, d3)

    output = Conv2D(1, (1, 1), activation='sigmoid')(d4)
    model = Model(inputs=inputs, outputs=output, name='Student_Unet')
    model.summary()
    return model


def distill(results_path, teacher_checkpoint_path):
    X_train = np.load(f'{results_path}/X_train.npy')
    Y_train = np.load(f'{results_path}/Y_train.npy')
    X_val   = np.load(f'{results_path}/X_val.npy')
    Y_val   = np.load(f'{results_path}/Y_val.npy')
    X_test  = np.load(f'{results_path}/X_test.npy')
    Y_test  = np.load(f'{results_path}/Y_test.npy')

    H, W = X_train.shape[1], X_train.shape[2]

    teacher = tf.keras.models.load_model(
        teacher_checkpoint_path,
        custom_objects={'bce_dice_loss': bce_dice_loss,
                        'iou_metric': iou_metric,
                        'continuity_metric': continuity_metric,
                        'segment_count_metric': segment_count_metric,
                        'path_quality_metric': path_quality_metric}
    )

    student = build_student(input_size=(H, W, 3))

    distiller = Distiller(student=student, teacher=teacher)
    distiller.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
        metrics=[iou_metric, continuity_metric, segment_count_metric, path_quality_metric],
        student_loss_fn=bce_dice_loss,
        distillation_loss_fn=keras.losses.KLDivergence(),
        alpha=0.1,
        temperature=5,
    )

    early_stopping = EarlyStopping(monitor='val_path_quality_metric', patience=15,
                                   verbose=1, restore_best_weights=True, mode='max')

    print(f'\nIniciando destilação em: {results_path}')
    distiller.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=120,
        batch_size=64,
        callbacks=[early_stopping],
        verbose=2,
    )

    # restore_best_weights=True garante que student tem os pesos do melhor epoch aqui
    student_checkpoint = f'{results_path}/student_distilled.keras'
    student.save(student_checkpoint)
    print(f'Modelo aluno salvo em: {student_checkpoint}')

    predictions_test = student.predict(X_test, verbose=0)
    reach = reachability_metric(X_test, predictions_test)
    print(f'Alcançabilidade (início→fim): {reach:.4f}')

    visualize_results(X_test, Y_test, student, results_path,
                      num_samples=max(1, int(len(X_test) * 0.30)),
                      prefix='distilled_', pred_label='Aluno')

    return student_checkpoint


if __name__ == '__main__':
    results_path = './results/result_W064xH064_D01_S000000_E005000'
    teacher_ckpt = f'{results_path}/path_finder_Unet.keras'
    distill(results_path, teacher_ckpt)
