#!/usr/bin/env python3
"""
infer.py — Inferência e visualização de resultados para modelos de path-finding.

Uso:
    python infer.py --npz data.npz --model model.keras
    python infer.py -n data.npz -m model.onnx -o ./saida -s 10 --all
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path
from scipy.ndimage import convolve, label as scipy_label


# ── Paleta visual (mesma do Visualizer.py) ─────────────────────────────────────
_PATH_COLOR = np.array([0.95, 0.45, 0.00], dtype=np.float32)
_OBST_COLOR = np.array([0.40, 0.40, 0.40], dtype=np.float32)
_BG_COLOR   = np.array([1.00, 1.00, 1.00], dtype=np.float32)

_LEGEND_ELEMENTS = [
    mlines.Line2D([], [], marker='o', color='w', markerfacecolor='limegreen',
                  markeredgecolor='black', markersize=8, label='Início'),
    mlines.Line2D([], [], marker='X', color='w', markerfacecolor='red',
                  markeredgecolor='darkred', markersize=8, label='Fim'),
]


# ── Métricas (numpy/scipy) ─────────────────────────────────────────────────────
def _iou(y_true, y_pred, smooth=1e-6):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return float((intersection + smooth) / (union + smooth))


def _bce_dice(y_true, y_pred_raw, smooth=1e-6):
    eps = 1e-7
    p = np.clip(y_pred_raw, eps, 1 - eps)
    bce = -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
    intersection = np.sum(y_true * y_pred_raw)
    dice = 1.0 - (2.0 * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred_raw) + smooth)
    return float(0.5 * bce + 0.5 * dice)


def _endpoint_stats(pred_binary):
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.float32)
    neighbors = convolve(pred_binary.astype(np.float32), kernel, mode='constant', cval=0)
    n_endpoints = np.sum(pred_binary * (neighbors == 1))
    n_isolated  = np.sum(pred_binary * (neighbors == 0))
    return n_endpoints, n_isolated


def _continuity(pred_binary):
    n_endpoints, n_isolated = _endpoint_stats(pred_binary)
    n_eff = n_endpoints + 2.0 * n_isolated
    has_pixels = float(np.sum(pred_binary) >= 1)
    return float(2.0 / max(2.0, n_eff + 1e-6)) * has_pixels


def _segment_count(pred_binary):
    n_endpoints, n_isolated = _endpoint_stats(pred_binary)
    n_eff = n_endpoints + 2.0 * n_isolated
    n_segments = max(1.0, n_eff / 2.0)
    has_pixels = float(np.sum(pred_binary) >= 1)
    return float(np.exp(1.0 - n_segments)) * has_pixels


def _reachability(x_sample, pred_binary):
    start_yx = np.argwhere(x_sample[:, :, 1] > 0.5)
    end_yx   = np.argwhere(x_sample[:, :, 2] > 0.5)
    if not len(start_yx) or not len(end_yx):
        return 0.0
    labeled, _ = scipy_label(pred_binary)
    l_start = labeled[start_yx[0][0], start_yx[0][1]]
    l_end   = labeled[end_yx[0][0],   end_yx[0][1]]
    return float(l_start != 0 and l_start == l_end)


def compute_metrics(x, y_true, pred_raw, threshold):
    pred_bin = (pred_raw > threshold).astype(np.float32)
    pb = pred_bin.squeeze()
    m = {}
    if y_true is not None:
        yt = y_true.squeeze()
        yp = pred_raw.squeeze()
        m['IoU']       = _iou(yt, pb)
        m['BCE+Dice']  = _bce_dice(yt, yp)
    m['Continuidade']    = _continuity(pb)
    m['Segmentos']       = _segment_count(pb)
    m['Alcançabilidade'] = _reachability(x, pb)
    return m, pred_bin


# ── Carregamento de modelos ────────────────────────────────────────────────────
def load_model(model_path):
    ext = Path(model_path).suffix.lower()
    if ext in ('.keras', '.h5'):
        import tensorflow as tf
        # tenta importar custom objects do projeto; ignora se não encontrado
        try:
            sys.path.insert(0, str(Path(model_path).parent))
            from Metrics import bce_dice_loss, iou_metric
            custom = {'bce_dice_loss': bce_dice_loss, 'iou_metric': iou_metric}
        except ImportError:
            custom = {}
        return tf.keras.models.load_model(model_path, custom_objects=custom), 'keras'

    elif ext == '.onnx':
        import onnxruntime as ort
        session = ort.InferenceSession(
            model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        return session, 'onnx'

    elif ext == '.tflite':
        import tensorflow as tf
        interp = tf.lite.Interpreter(model_path=model_path)
        interp.allocate_tensors()
        return interp, 'tflite'

    else:
        raise ValueError(f'Formato não suportado: {ext}  (use .keras, .h5, .onnx ou .tflite)')


def run_inference(model, model_type, X, batch_size):
    if model_type == 'keras':
        return model.predict(X, batch_size=batch_size, verbose=0)

    elif model_type == 'onnx':
        input_name = model.get_inputs()[0].name
        chunks = []
        for i in range(0, len(X), batch_size):
            out = model.run(None, {input_name: X[i:i+batch_size].astype(np.float32)})[0]
            chunks.append(out)
        return np.concatenate(chunks, axis=0)

    elif model_type == 'tflite':
        in_det  = model.get_input_details()
        out_det = model.get_output_details()
        H, W = X.shape[1], X.shape[2]
        preds = np.zeros((len(X), H, W, 1), dtype=np.float32)
        for i, sample in enumerate(X):
            model.set_tensor(in_det[0]['index'], sample[np.newaxis].astype(np.float32))
            model.invoke()
            preds[i] = model.get_tensor(out_det[0]['index'])[0]
        return preds


# ── Visualização ───────────────────────────────────────────────────────────────
def _make_img(obst, path_mask=None):
    H, W = obst.shape
    img = np.ones((H, W, 3), dtype=np.float32) * _BG_COLOR
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


def plot_sample(idx, x, y_true, pred_raw, pred_bin, output_dir, model_name):
    obst     = x[:, :, 0]
    start_yx = np.argwhere(x[:, :, 1] > 0.5)
    end_yx   = np.argwhere(x[:, :, 2] > 0.5)

    has_truth = y_true is not None
    cols_data = [('Input\n(Obstáculos)', _make_img(obst, None))]
    if has_truth:
        cols_data.append(('Target\n(Caminho A*)', _make_img(obst, y_true.squeeze())))
    cols_data.append(('Previsão\n(CNN)', _make_img(obst, pred_bin.squeeze())))

    fig, axes = plt.subplots(1, len(cols_data), figsize=(4 * len(cols_data), 4))
    fig.patch.set_facecolor('white')
    if len(cols_data) == 1:
        axes = [axes]

    H, W = obst.shape
    for ax, (title, img) in zip(axes, cols_data):
        ax.imshow(img, interpolation='nearest')
        ax.set_title(title, fontsize=9, color='black')
        ax.set_facecolor('white')

        # Grade fina por célula (igual ao bin_to_map.py)
        ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
        ax.tick_params(which='minor', length=0)
        ax.grid(which='minor', color='#cccccc', linewidth=0.25)

        # Ticks maiores a cada 8 células
        ax.set_xticks(range(0, W, 8))
        ax.set_yticks(range(0, H, 8))
        ax.tick_params(colors='black', labelsize=6)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('#aaaaaa')
            spine.set_linewidth(0.8)
        _add_markers(ax, start_yx, end_yx)

    axes[0].legend(handles=_LEGEND_ELEMENTS, loc='upper right', fontsize=7,
                   framealpha=0.85, facecolor='white', labelcolor='black')

    plt.tight_layout(pad=0.3)
    fig.subplots_adjust(wspace=0.04)

    out = os.path.join(output_dir, f'{model_name}_sample_{idx:04d}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out


def plot_summary(all_metrics, output_dir, model_name):
    keys = list(all_metrics[0].keys())
    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, key in zip(axes, keys):
        vals = [m[key] for m in all_metrics]
        mean, std = np.mean(vals), np.std(vals)
        ax.hist(vals, bins=min(20, len(vals)), color='steelblue', edgecolor='black', alpha=0.8)
        ax.axvline(mean, color='red', linestyle='--', linewidth=1.5, label=f'Média: {mean:.4f}')
        ax.set_title(key, fontsize=10)
        ax.set_xlabel('Score')
        ax.set_ylabel('Frequência')
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.text(0.02, 0.97, f'std: {std:.4f}', transform=ax.transAxes,
                fontsize=7, va='top', color='#555555')

    fig.suptitle(f'Distribuição de Métricas — {model_name}  ({len(all_metrics)} amostras)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(output_dir, f'{model_name}_summary.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description='Inferência e visualização para modelos de path-finding.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--npz',       '-n', required=True,
                   help='Arquivo NPZ com arrays "X" e (opcionalmente) "Y"')
    p.add_argument('--model',     '-m', required=True,
                   help='Modelo (.keras, .h5, .onnx, .tflite)')
    p.add_argument('--output',    '-o', default=None,
                   help='Diretório de saída (padrão: <dir do NPZ>/inference_results/)')
    p.add_argument('--samples',   '-s', type=int, default=5,
                   help='Número de amostras para plotar individualmente (padrão: 5)')
    p.add_argument('--all',       '-a', action='store_true',
                   help='Calcular métricas em todo o dataset (não só nas amostras plotadas)')
    p.add_argument('--threshold', '-t', type=float, default=0.5,
                   help='Limiar de binarização da predição (padrão: 0.5)')
    p.add_argument('--batch',     '-b', type=int, default=32,
                   help='Batch size para inferência (padrão: 32)')
    return p.parse_args()


def main():
    args = parse_args()

    # Adiciona dir do script ao path para importar Metrics.py
    sys.path.insert(0, str(Path(__file__).parent))

    output_dir = args.output or str(Path(args.npz).parent / 'inference_results')
    os.makedirs(output_dir, exist_ok=True)

    # ── Carrega dados ──────────────────────────────────────────────────────────
    print(f'[+] Carregando {args.npz}')
    npz = np.load(args.npz)

    def _find_key(npz, *candidates):
        for k in candidates:
            if k in npz:
                return k
        return None

    x_key = _find_key(npz, 'X', 'x')
    y_key = _find_key(npz, 'Y', 'y')

    if x_key is None:
        print('[!] Erro: o arquivo NPZ deve conter o array "X" ou "x".')
        sys.exit(1)

    X = npz[x_key].astype(np.float32)
    Y = npz[y_key].astype(np.float32) if y_key is not None else None
    print(f'    X: {X.shape} (key="{x_key}")  |  Y: {Y.shape if Y is not None else f"ausente (key buscada: Y/y)"}')

    # ── Carrega modelo ─────────────────────────────────────────────────────────
    print(f'[+] Carregando modelo {args.model}')
    model, model_type = load_model(args.model)
    model_name = Path(args.model).stem
    print(f'    Tipo detectado: {model_type}')

    # ── Define escopo da inferência ────────────────────────────────────────────
    if args.all:
        X_infer = X
        Y_infer = Y
    else:
        n = max(args.samples, 1)
        X_infer = X[:n]
        Y_infer = Y[:n] if Y is not None else None

    print(f'[+] Executando inferência em {len(X_infer)} amostras...')
    preds_raw = run_inference(model, model_type, X_infer, args.batch)

    # ── Calcula métricas ───────────────────────────────────────────────────────
    print('[+] Calculando métricas...')
    all_metrics = []
    all_pred_bin = []
    for i in range(len(X_infer)):
        yt = Y_infer[i] if Y_infer is not None else None
        m, pb = compute_metrics(X_infer[i], yt, preds_raw[i], args.threshold)
        all_metrics.append(m)
        all_pred_bin.append(pb)

    # ── Imprime resumo ─────────────────────────────────────────────────────────
    print()
    print('── Métricas ' + '─' * 48)
    header = f'  {"Métrica":<18}  {"Média":>8}  {"Std":>8}  {"Mín":>8}  {"Máx":>8}'
    print(header)
    print('  ' + '─' * (len(header) - 2))
    for key in all_metrics[0].keys():
        vals = [m[key] for m in all_metrics]
        print(f'  {key:<18}  {np.mean(vals):8.4f}  {np.std(vals):8.4f}  '
              f'{np.min(vals):8.4f}  {np.max(vals):8.4f}')
    print('─' * 60)
    print()

    # ── Plota amostras individuais ─────────────────────────────────────────────
    n_plot = min(args.samples, len(X_infer))
    print(f'[+] Plotando {n_plot} amostras...')
    for i in range(n_plot):
        yt = Y_infer[i] if Y_infer is not None else None
        path = plot_sample(
            i, X_infer[i], yt,
            preds_raw[i], all_pred_bin[i],
            output_dir, model_name,
        )
        print(f'    {path}')

    # ── Histogramas de métricas ────────────────────────────────────────────────
    if len(all_metrics) > 1:
        summary = plot_summary(all_metrics, output_dir, model_name)
        print(f'[+] Resumo salvo em: {summary}')

    print(f'\n[✓] Concluído. Resultados em: {output_dir}')


if __name__ == '__main__':
    main()
