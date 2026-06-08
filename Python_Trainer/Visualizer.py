import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


_PATH_COLOR = np.array([0.95, 0.45, 0.00], dtype=np.float32)
_OBST_COLOR = np.array([0.40, 0.40, 0.40], dtype=np.float32)
_LEGEND_ELEMENTS = [
    mlines.Line2D([], [], marker='o', color='w', markerfacecolor='limegreen',
                  markeredgecolor='black', markersize=8, label='Início'),
    mlines.Line2D([], [], marker='X', color='w', markerfacecolor='red',
                  markeredgecolor='darkred', markersize=8, label='Fim'),
]


def _composite(obst, path_mask):
    H, W = obst.shape
    img = np.ones((H, W, 3), dtype=np.float32)
    img[obst > 0.5] = _OBST_COLOR
    if path_mask is not None:
        img[path_mask > 0.5] = _PATH_COLOR
    return img


def _add_markers(ax, start_yx, end_yx):
    kw = dict(zorder=5, linewidths=1.2)
    if len(start_yx):
        sy, sx = start_yx[0]
        ax.scatter(sx, sy, s=70, c='limegreen', marker='o', edgecolors='black', **kw)
    if len(end_yx):
        ey, ex = end_yx[0]
        ax.scatter(ex, ey, s=70, c='red', marker='X', edgecolors='darkred', **kw)


def visualize_results(X_data, Y_true, model, results_path, num_samples=3, prefix='', pred_label='CNN', sfilter=None, depth=None):
    predictions = model.predict(X_data[:num_samples])
    images_result = f'{results_path}/test/{prefix}{model.name}'
    os.makedirs(images_result, exist_ok=True)

    for i in range(num_samples):
        obst      = X_data[i, :, :, 0]
        start_yx  = np.argwhere(X_data[i, :, :, 1] > 0.5)
        end_yx    = np.argwhere(X_data[i, :, :, 2] > 0.5)
        pred_mask = predictions[i].squeeze() > 0.5

        img_input  = _composite(obst, None)
        img_target = _composite(obst, Y_true[i].squeeze())
        img_pred   = _composite(obst, pred_mask)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, img, title in zip(
            axes,
            [img_input, img_target, img_pred],
            ['Input\n(Obstáculos)', 'Target\n(Caminho A*)', f'Previsão\n({pred_label})'],
        ):
            ax.imshow(img, interpolation='nearest')
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor('black')
                spine.set_linewidth(1)

        _add_markers(axes[0], start_yx, end_yx)
        _add_markers(axes[1], start_yx, end_yx)
        _add_markers(axes[2], start_yx, end_yx)
        axes[0].legend(handles=_LEGEND_ELEMENTS, loc='upper right',
                       fontsize=7, framealpha=0.85)

        if sfilter is not None and depth is not None:
            config_str = f'depth={depth}  |  sfilter={sfilter}  |  bottleneck={sfilter * (2 ** depth)}f  |  sample {i}'
        else:
            config_str = f'sample {i}'
        fig.text(0.5, 0.01, config_str, ha='center', va='bottom',
                 fontsize=7, color='#555555', style='italic')

        plt.tight_layout(pad=0.3)
        fig.subplots_adjust(wspace=0.04, bottom=0.08)
        config_tag = f'_d{depth}_f{sfilter}' if sfilter is not None and depth is not None else ''
        plt.savefig(f'{images_result}/prediction{config_tag}_{i}.png', dpi=150, bbox_inches='tight')
        plt.close()


def plot_training_results(history, model_name, basename, results_path, sfilter=None, depth=None):
    specs = [
        # ('path_quality_metric',  'val_path_quality_metric',  'Path Quality',   'Score'),
        ('iou_metric', 'val_iou_metric', 'IoU (%) por Época',      'IoU (%)'),
        # ('continuity_metric',    'val_continuity_metric',    'Continuidade',   'Score'),
        # ('segment_count_metric', 'val_segment_count_metric', 'Segmentos Únicos','Score'),
        ('loss',       'val_loss',       'Val. Loss (%) por Época', 'Val. Loss (%)'),
    ]

    _, axes = plt.subplots(len(specs), 1, figsize=(8, 5 * len(specs)))

    for ax, (train_key, val_key, title, ylabel) in zip(axes, specs):
        val_values = history.history[val_key]
        ax.plot(history.history[train_key], label='Treino',    color='blue',   linewidth=2)
        ax.plot(val_values,                 label='Validação', color='orange', linewidth=2)
        ax.set_title(title)
        ax.set_xlabel('Épocas')
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.6)

        # Marca o melhor valor de validação
        if 'loss' in val_key:
            best_epoch = int(np.argmin(val_values))
            best_value = val_values[best_epoch]
            marker = 'v'
            marker_color = 'red'
            label_best = f'Mín: {best_value:.4f}\n(época {best_epoch + 1})'
        else:
            best_epoch = int(np.argmax(val_values))
            best_value = val_values[best_epoch]
            marker = '^'
            marker_color = 'green'
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
    images_result = f'{results_path}/test/{model_name}'
    os.makedirs(images_result, exist_ok=True)
    suffix = f'_d{depth}_f{sfilter}' if depth is not None and sfilter is not None else ''
    plt.savefig(f'{images_result}/training_results{suffix}.png', dpi=300)
    plt.close()
