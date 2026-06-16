import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose
from tensorflow.keras.models import Model

from Metrics import iou_metric, continuity_metric, segment_count_metric, path_quality_metric, bce_dice_loss
from Visualizer import visualize_results


class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.teacher      = teacher
        self.student      = student
        self.loss_tracker = keras.metrics.Mean(name='loss')
        self.iou_tracker  = keras.metrics.Mean(name='iou_metric')

    @property
    def metrics(self):
        return [self.loss_tracker, self.iou_tracker]

    def compile(self, optimizer, student_loss_fn, distillation_loss_fn, alpha=0.1, temperature=3):
        super(Distiller, self).compile(optimizer=optimizer)
        self.student_loss_fn      = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha                = alpha
        self.temperature          = temperature

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

        self.loss_tracker.update_state(loss)
        self.iou_tracker.update_state(iou_metric(y, student_predictions))
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

        self.loss_tracker.update_state(loss)
        self.iou_tracker.update_state(iou_metric(y, student_predictions))
        return {m.name: m.result() for m in self.metrics}


def build_student(input_size):
    inputs = Input(input_size)

    def encoder_block(filters, x):
        x = Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
        p = MaxPooling2D((2, 2), padding='same')(x)
        return x, p

    def baseline_layer(filters, x):
        x = Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
        return x

    def decoder_block(filters, skip, x):
        x = Conv2DTranspose(filters, (2, 2), padding='same', activation='relu', strides=2)(x)
        x = concatenate([x, skip], axis=-1)
        x = Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
        return x

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


def _plot_distiller_results(history, results_path):
    keys = list(history.history.keys())

    iou_train = next((k for k in keys if 'iou' in k and not k.startswith('val_')), None)
    iou_val   = next((k for k in keys if 'iou' in k and k.startswith('val_')), None)

    specs = []
    if iou_train and iou_val:
        specs.append((iou_train, iou_val, 'IoU (%) por Época', 'IoU (%)'))
    if 'loss' in keys and 'val_loss' in keys:
        specs.append(('loss', 'val_loss', 'Val. Loss (%) por Época', 'Val. Loss (%)'))

    if not specs:
        print(f'[Distiller] Nenhuma métrica disponível para plotar. Chaves: {keys}')
        return

    _, axes = plt.subplots(len(specs), 1, figsize=(8, 5 * len(specs)))
    if len(specs) == 1:
        axes = [axes]

    for ax, (train_key, val_key, title, ylabel) in zip(axes, specs):
        val_values = history.history[val_key]
        ax.plot(history.history[train_key], label='Treino',    color='blue',   linewidth=2)
        ax.plot(val_values,                 label='Validação', color='orange', linewidth=2)
        ax.set_title(title)
        ax.set_xlabel('Épocas')
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.6)

        if 'loss' in val_key:
            best_epoch = int(np.argmin(val_values))
            best_value = val_values[best_epoch]
            marker, marker_color = 'v', 'red'
            label_best = f'Mín: {best_value:.4f}\n(época {best_epoch + 1})'
        else:
            best_epoch = int(np.argmax(val_values))
            best_value = val_values[best_epoch]
            marker, marker_color = '^', 'green'
            label_best = f'Máx: {best_value:.4f}\n(época {best_epoch + 1})'

        ax.scatter(best_epoch, best_value, color=marker_color,
                   marker=marker, s=80, zorder=5, label=label_best)
        ax.annotate(
            f'{best_value:.4f}',
            xy=(best_epoch, best_value),
            xytext=(8, 8), textcoords='offset points',
            fontsize=8, color=marker_color, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=marker_color, lw=1.0),
        )
        ax.legend(fontsize=8)

    plt.tight_layout()
    out_dir = os.path.join(results_path, 'test', 'Student_Unet')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'training_results_distiller.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f'Gráfico de destilação salvo em: {out_path}')


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
        custom_objects={
            'bce_dice_loss': bce_dice_loss,
            'iou_metric': iou_metric,
            # 'continuity_metric': continuity_metric,
            # 'segment_count_metric': segment_count_metric,
            # 'path_quality_metric': path_quality_metric
        }
    )

    student = build_student(input_size=(H, W, 3))

    distiller = Distiller(student=student, teacher=teacher)
    distiller.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
        student_loss_fn=bce_dice_loss,
        distillation_loss_fn=keras.losses.KLDivergence(),
        alpha=0.1,
        temperature=5,
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=1,
        restore_best_weights=True,
        mode='min'
    )

    print(f'\nIniciando destilação em: {results_path}')
    history = distiller.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=100,
        batch_size=64,
        callbacks=[early_stopping],
        verbose=2,
    )

    _plot_distiller_results(history, results_path)

    # restore_best_weights=True garante que student tem os pesos do melhor epoch aqui
    student_checkpoint = f'{results_path}/student_distilled.keras'
    student.save(student_checkpoint)
    print(f'Modelo aluno salvo em: {student_checkpoint}')

    visualize_results(X_test, Y_test, student, results_path,
                      num_samples=max(20, int(len(X_test) * 0.01)),
                      prefix='distilled_', pred_label='Student')

    return student_checkpoint


if __name__ == '__main__':
    results_path = './results/result_W064xH064_D01_S000000_E005000'
    teacher_ckpt = f'{results_path}/path_finder_Unet.keras'
    distill(results_path, teacher_ckpt)
