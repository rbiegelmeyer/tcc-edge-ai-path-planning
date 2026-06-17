#!/usr/bin/env python3
"""
compare_models.py — Comparação visual: A* vs Modelo 1 vs Modelo 2 vs Heatmap Embarcado.

Gera uma imagem por amostra com 4 painéis horizontais:
  [A* (Ground Truth) | Modelo 1 | Modelo 2 | Heatmap (bin_to_map)]

Uso:
    python compare_models.py \\
        --npz     data.npz \\
        --model1  unet.keras     --name1 "U-Net D4" \\
        --model2  student.onnx   --name2 "Student INT8" \\
        --result  output.bin     --input-bin input.bin  --name-bin "STM32H743" \\
        -o ./comparativo  -c 10

Binários (--result / --input-bin):
    Se o arquivo contiver N mapas concatenados (N × H × W bytes), o mapa do
    índice global i é lido no offset i × H × W.
    Se o arquivo contiver apenas um mapa (H × W bytes), ele é reutilizado para
    todos os índices.
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import Optional
from scipy.ndimage import convolve, label as scipy_label


# ── Paleta (tema claro — igual infer.py / bin_to_map.py) ─────────────────────
_PATH_COLOR = np.array([0.95, 0.45, 0.00], dtype=np.float32)   # laranja
_OBST_COLOR = np.array([0.40, 0.40, 0.40], dtype=np.float32)   # cinza
_BG_COLOR   = np.array([1.00, 1.00, 1.00], dtype=np.float32)   # branco
_WALL_RGBA  = (0.40, 0.40, 0.40)

_FIG_BG    = 'white'
_GRID_CLR  = '#cccccc'
_SPINE_CLR = '#aaaaaa'
_TEXT_CLR  = 'black'

# Colormap OrRd com -128 forçado para branco puro (igual ao fundo)
# Gradiente branco → cor do caminho (mesma do _PATH_COLOR)
_HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    'white_to_path',
    [(1.0, 1.0, 1.0), tuple(_PATH_COLOR[:3])],
    N=256,
)

_LEGEND_ELEMENTS = [
    mlines.Line2D([], [], marker='o', color='w', markerfacecolor='limegreen',
                  markeredgecolor='black', markersize=7, label='Início'),
    mlines.Line2D([], [], marker='X', color='w', markerfacecolor='red',
                  markeredgecolor='darkred', markersize=7, label='Fim'),
]


# ── Métricas ───────────────────────────────────────────────────────────────────
def _iou(y_true, y_pred, smooth=1e-6):
    i = np.sum(y_true * y_pred)
    u = np.sum(y_true) + np.sum(y_pred) - i
    return float((i + smooth) / (u + smooth))


def _reachability(x_sample, pred_binary):
    start_yx = np.argwhere(x_sample[:, :, 1] > 0.5)
    end_yx   = np.argwhere(x_sample[:, :, 2] > 0.5)
    if not len(start_yx) or not len(end_yx):
        return 0.0
    labeled, _ = scipy_label(pred_binary)
    ls = labeled[start_yx[0][0], start_yx[0][1]]
    le = labeled[end_yx[0][0],   end_yx[0][1]]
    return float(ls != 0 and ls == le)


# ── Carregamento de modelos ────────────────────────────────────────────────────
def load_model(model_path):
    ext = Path(model_path).suffix.lower()
    if ext in ('.keras', '.h5'):
        import tensorflow as tf
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from Metrics import bce_dice_loss, iou_metric
            custom = {'bce_dice_loss': bce_dice_loss, 'iou_metric': iou_metric}
        except ImportError:
            custom = {}
        return tf.keras.models.load_model(model_path, custom_objects=custom), 'keras'
    elif ext == '.onnx':
        import onnxruntime as ort
        return ort.InferenceSession(model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']), 'onnx'
    elif ext == '.tflite':
        import tensorflow as tf
        interp = tf.lite.Interpreter(model_path=model_path)
        interp.allocate_tensors()
        return interp, 'tflite'
    else:
        raise ValueError(f'Formato não suportado: {ext}')


def run_inference(model, model_type, X, batch_size):
    if model_type == 'keras':
        return model.predict(X, batch_size=batch_size, verbose=0)
    elif model_type == 'onnx':
        name = model.get_inputs()[0].name
        chunks = [model.run(None, {name: X[i:i+batch_size].astype(np.float32)})[0]
                  for i in range(0, len(X), batch_size)]
        return np.concatenate(chunks, axis=0)
    elif model_type == 'tflite':
        in_d, out_d = model.get_input_details(), model.get_output_details()
        H, W = X.shape[1], X.shape[2]
        preds = np.zeros((len(X), H, W, 1), dtype=np.float32)
        for i, s in enumerate(X):
            model.set_tensor(in_d[0]['index'], s[np.newaxis].astype(np.float32))
            model.invoke()
            preds[i] = model.get_tensor(out_d[0]['index'])[0]
        return preds


# ── Carregamento de binários ───────────────────────────────────────────────────
def _read_bin_map(data: bytes, global_idx: int, H: int, W: int) -> Optional[np.ndarray]:
    """Lê mapa de índice global_idx; se arquivo tiver só um mapa, usa offset 0."""
    size = H * W
    off  = global_idx * size
    chunk = data[off: off + size]
    if len(chunk) == size:
        return np.frombuffer(chunk, dtype=np.int8).reshape(H, W)
    if len(data) >= size:
        return np.frombuffer(data[:size], dtype=np.int8).reshape(H, W)
    return None


def _find_input_block(data: bytes, H: int, W: int) -> Optional[np.ndarray]:
    """Busca bloco válido {0-4} com início(3) e fim(4) — fallback para input-bin."""
    size, valid = H * W, {0, 1, 2, 3, 4}
    for off in range(0, len(data) - size + 1, 4):
        s = set(data[off: off + size])
        if s <= valid and {3, 4} <= s:
            return np.frombuffer(data[off: off + size], dtype=np.int8).reshape(H, W)
    return None


# ── Helpers visuais ────────────────────────────────────────────────────────────
def _make_img(obst, path_mask=None):
    img = np.ones((*obst.shape, 3), dtype=np.float32)
    img[obst > 0.5] = _OBST_COLOR
    if path_mask is not None:
        img[path_mask > 0.5] = _PATH_COLOR
    return img


def _add_markers(ax, start_yx, end_yx):
    kw = dict(zorder=6, linewidths=1.2)
    if len(start_yx):
        sy, sx = start_yx[0]
        ax.scatter(sx, sy, s=70, c='limegreen', marker='o', edgecolors='black', **kw)
    if len(end_yx):
        ey, ex = end_yx[0]
        ax.scatter(ex, ey, s=70, c='red', marker='X', edgecolors='darkred', **kw)


def _apply_grid(ax, H, W, show_ticks=True):
    ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
    ax.tick_params(which='minor', length=0)
    ax.grid(which='minor', color=_GRID_CLR, linewidth=0.25)
    if show_ticks:
        ax.set_xticks(range(0, W, 8))
        ax.set_yticks(range(0, H, 8))
        ax.tick_params(colors=_TEXT_CLR, labelsize=6)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(_SPINE_CLR)
        spine.set_linewidth(0.8)


def _overlay_walls_markers(ax, input_map):
    """Sobrepõe paredes, início e fim do mapa de entrada sobre o heatmap."""
    H, W = input_map.shape
    wall_rgba = np.zeros((H, W, 4), dtype=np.float32)
    wall_rgba[:, :, 0] = _WALL_RGBA[0]
    wall_rgba[:, :, 1] = _WALL_RGBA[1]
    wall_rgba[:, :, 2] = _WALL_RGBA[2]
    wall_rgba[:, :, 3] = np.where(input_map == 1, 1.0, 0.0)
    ax.imshow(wall_rgba, interpolation='nearest', origin='upper', aspect='equal')
    kw = dict(zorder=6, linewidths=1.2)
    starts = np.argwhere(input_map == 3)
    ends   = np.argwhere(input_map == 4)
    if len(starts):
        r, c = starts[0]
        ax.scatter(c, r, s=70, c='limegreen', marker='o', edgecolors='black', **kw)
    if len(ends):
        r, c = ends[0]
        ax.scatter(c, r, s=70, c='red', marker='X', edgecolors='darkred', **kw)


# ── Plot de uma amostra (4 painéis) ───────────────────────────────────────────
def plot_comparison(global_idx, x, y_true,
                    pred1_raw, pred2_raw,
                    result_map, input_map,
                    title_gt, title_m1, title_m2, title_bin,
                    threshold, output_dir, prefix,
                    layout='horizontal', show_ticks=True):

    H, W     = x.shape[0], x.shape[1]
    obst     = x[:, :, 0]
    start_yx = np.argwhere(x[:, :, 1] > 0.5)
    end_yx   = np.argwhere(x[:, :, 2] > 0.5)

    pred1_bin = (pred1_raw.squeeze() > threshold).astype(np.float32)
    pred2_bin = (pred2_raw.squeeze() > threshold).astype(np.float32)

    # Títulos com IoU quando ground truth disponível
    def _title_with_iou(base, pb):
        if y_true is not None:
            iou  = _iou(y_true.squeeze(), pb)
            reac = _reachability(x, pb)
            return f'{base}\nIoU={iou:.3f}  Alc={reac:.0f}'
        return base

    label_m1 = _title_with_iou(title_m1, pred1_bin)
    label_m2 = _title_with_iou(title_m2, pred2_bin)

    if layout == 'vertical':
        fig, axes = plt.subplots(4, 1, figsize=(4.5, 17))
        axes = list(axes)
        adjust_kw = dict(hspace=0.06)
    elif layout == 'square':
        fig, ax_grid = plt.subplots(2, 2, figsize=(9, 9))
        axes = [ax_grid[0, 0], ax_grid[0, 1], ax_grid[1, 0], ax_grid[1, 1]]
        adjust_kw = dict(hspace=0.06, wspace=0.06)
    else:  # horizontal
        fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))
        axes = list(axes)
        adjust_kw = dict(wspace=0.06)
    fig.patch.set_facecolor(_FIG_BG)

    # ── Painel 1: A* Ground Truth ─────────────────────────────────────────────
    # Prioridade: Y do NPZ → input_map do binário → apenas obstáculos do NPZ
    ax = axes[0]
    ax.set_facecolor(_FIG_BG)
    if y_true is not None:
        p1_obst     = obst
        p1_path     = y_true.squeeze()
        p1_start_yx = start_yx
        p1_end_yx   = end_yx
    elif input_map is not None:
        p1_obst     = (input_map == 1).astype(np.float32)
        p1_path     = np.isin(input_map, [2, 3, 4]).astype(np.float32)
        p1_start_yx = np.argwhere(input_map == 3)
        p1_end_yx   = np.argwhere(input_map == 4)
    else:
        p1_obst     = obst
        p1_path     = None
        p1_start_yx = start_yx
        p1_end_yx   = end_yx
    ax.imshow(_make_img(p1_obst, p1_path), interpolation='nearest')
    ax.set_title(title_gt, fontsize=11, color=_TEXT_CLR)
    _apply_grid(ax, H, W, show_ticks)
    _add_markers(ax, p1_start_yx, p1_end_yx)
    ax.legend(handles=_LEGEND_ELEMENTS, loc='upper right', fontsize=8,
              framealpha=0.85, facecolor='white', labelcolor=_TEXT_CLR)

    # ── Painel 2: Modelo 1 ────────────────────────────────────────────────────
    ax = axes[1]
    ax.set_facecolor(_FIG_BG)
    ax.imshow(_make_img(obst, pred1_bin), interpolation='nearest')
    ax.set_title(label_m1, fontsize=11, color=_TEXT_CLR)
    _apply_grid(ax, H, W, show_ticks)
    _add_markers(ax, start_yx, end_yx)

    # ── Painel 3: Modelo 2 ────────────────────────────────────────────────────
    ax = axes[2]
    ax.set_facecolor(_FIG_BG)
    ax.imshow(_make_img(obst, pred2_bin), interpolation='nearest')
    ax.set_title(label_m2, fontsize=11, color=_TEXT_CLR)
    _apply_grid(ax, H, W, show_ticks)
    _add_markers(ax, start_yx, end_yx)

    # ── Painel 4: Heatmap binário (bin_to_map) ────────────────────────────────
    ax = axes[3]
    ax.set_facecolor(_FIG_BG)
    if result_map is not None:
        # Heatmap int8 com temperatura (saída do modelo embarcado)
        im = ax.imshow(result_map.astype(np.float32),
                       cmap=_HEATMAP_CMAP, vmin=-128, vmax=127,
                       interpolation='nearest', origin='upper', aspect='equal')
        if input_map is not None:
            _overlay_walls_markers(ax, input_map)
        # Colorbar dentro do painel para não alterar largura
        cax  = ax.inset_axes([1.02, 0.05, 0.04, 0.90])
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.yaxis.set_tick_params(color=_TEXT_CLR, labelsize=6)
        cbar.outline.set_edgecolor(_SPINE_CLR)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=_TEXT_CLR)
        cbar.set_label('int8', color=_TEXT_CLR, fontsize=7, labelpad=4)
    elif input_map is not None:
        # Fallback: exibe o mapa de entrada com estilo discreto
        p4_obst = (input_map == 1).astype(np.float32)
        p4_path = np.isin(input_map, [2]).astype(np.float32)
        ax.imshow(_make_img(p4_obst, p4_path), interpolation='nearest')
        p4_starts = np.argwhere(input_map == 3)
        p4_ends   = np.argwhere(input_map == 4)
        _add_markers(ax, p4_starts, p4_ends)
    else:
        ax.text(0.5, 0.5, 'Sem dados\nbinários', ha='center', va='center',
                fontsize=10, color='#888888', transform=ax.transAxes)
    ax.set_title(title_bin, fontsize=11, color=_TEXT_CLR)
    _apply_grid(ax, H, W, show_ticks)

    plt.tight_layout(pad=0.4)
    fig.subplots_adjust(**adjust_kw)

    out = os.path.join(output_dir, f'{prefix}_{global_idx:04d}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=_FIG_BG)
    plt.close()
    return out


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description='Comparação visual: A* vs dois modelos CNN vs heatmap embarcado.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Dados de entrada
    p.add_argument('--npz', '-n', required=True,
                   help='NPZ com arrays "X" (e opcionalmente "Y")')

    # Modelo 1
    p.add_argument('--model1', '-m1', required=True,
                   help='Modelo 1 (.keras / .h5 / .onnx / .tflite)')
    p.add_argument('--name1', default='Modelo 1',
                   help='Título do painel do Modelo 1 (padrão: "Modelo 1")')

    # Modelo 2
    p.add_argument('--model2', '-m2', required=True,
                   help='Modelo 2 (.keras / .h5 / .onnx / .tflite)')
    p.add_argument('--name2', default='Modelo 2',
                   help='Título do painel do Modelo 2 (padrão: "Modelo 2")')

    # Binários para o painel heatmap
    p.add_argument('--result', '-r', default=None, metavar='RESULT_BIN',
                   help='Binário com heatmap embarcado (int8, −128..127)')
    p.add_argument('--input-bin', '-i', default=None, metavar='INPUT_BIN',
                   help='Binário com mapa de entrada (valores 0-4, paredes/início/fim)')
    p.add_argument('--name-bin', default='Embarcado',
                   help='Título do painel heatmap (padrão: "Embarcado")')

    # Painel 1 (ground truth)
    p.add_argument('--name-gt', default='A* (Ground Truth)',
                   help='Título do painel de ground truth')

    # Controle de amostras
    p.add_argument('--ids', type=int, nargs='+', default=None, metavar='IDX',
                   help='Índices específicos a processar, ex: --ids 0 5 42 (substitui --count/--skip/--all)')
    p.add_argument('--count', '-c', type=int, default=5,
                   help='Número de amostras a processar (padrão: 5)')
    p.add_argument('--skip',  '-s', type=int, default=0,
                   help='Pular os primeiros N índices do NPZ (padrão: 0)')
    p.add_argument('--all',   '-a', action='store_true',
                   help='Processar todas as amostras do NPZ')

    # Opções gerais
    p.add_argument('--output', '-o', default=None,
                   help='Diretório de saída (padrão: ./comparativo/)')
    p.add_argument('--prefix', default='compare',
                   help='Prefixo dos arquivos PNG (padrão: compare)')
    p.add_argument('--layout', '-l', default='horizontal',
                   choices=['horizontal', 'vertical', 'square'],
                   help='Distribuição dos painéis: horizontal (padrão), vertical ou square (2×2)')
    p.add_argument('--no-ticks', action='store_true',
                   help='Ocultar os números dos ticks do grid')
    p.add_argument('--threshold', '-t', type=float, default=0.5)
    p.add_argument('--batch',     '-b', type=int,   default=32)
    p.add_argument('--height',    '-H', type=int,   default=64)
    p.add_argument('--width',     '-W', type=int,   default=64)
    return p.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    output_dir = args.output or str(Path.cwd() / 'comparativo')
    os.makedirs(output_dir, exist_ok=True)

    # ── NPZ ───────────────────────────────────────────────────────────────────
    print(f'[+] NPZ: {args.npz}')
    npz   = np.load(args.npz)
    x_key = next((k for k in ('X', 'x') if k in npz), None)
    y_key = next((k for k in ('Y', 'y') if k in npz), None)
    if x_key is None:
        print('[!] NPZ deve conter array "X" ou "x".')
        sys.exit(1)
    X = npz[x_key].astype(np.float32)
    Y = npz[y_key].astype(np.float32) if y_key else None
    print(f'    X: {X.shape}  |  Y: {Y.shape if Y is not None else "ausente"}')

    # ── Índices a processar ───────────────────────────────────────────────────
    n_total = len(X)
    if args.ids is not None:
        bad = [i for i in args.ids if i >= n_total]
        if bad:
            print(f'[!] Índices fora do range (NPZ tem {n_total} amostras): {bad}')
            sys.exit(1)
        indices = args.ids
    elif args.all:
        indices = list(range(args.skip, n_total))
    else:
        indices = list(range(args.skip, min(args.skip + args.count, n_total)))
    print(f'    Amostras selecionadas: {len(indices)}  {indices}')

    # ── Modelos ───────────────────────────────────────────────────────────────
    for path, label in [(args.model1, 'model1'), (args.model2, 'model2')]:
        if not os.path.isfile(path):
            print(f'[!] Arquivo não encontrado: {path}')
            sys.exit(1)

    print(f'[+] Carregando {args.model1}  →  "{args.name1}"')
    model1, type1 = load_model(args.model1)
    print(f'[+] Carregando {args.model2}  →  "{args.name2}"')
    model2, type2 = load_model(args.model2)

    print(f'[+] Inferência...')
    X_sel  = X[indices]
    preds1 = run_inference(model1, type1, X_sel, args.batch)
    preds2 = run_inference(model2, type2, X_sel, args.batch)

    # ── Binários ──────────────────────────────────────────────────────────────
    result_bytes = None
    input_bytes  = None
    H, W = args.height, args.width

    if args.result:
        if os.path.isfile(args.result):
            result_bytes = open(args.result, 'rb').read()
            print(f'[+] result-bin: {args.result}  ({len(result_bytes)} bytes)')
        else:
            print(f'[!] result-bin não encontrado: {args.result}')
    if args.input_bin:
        if os.path.isfile(args.input_bin):
            input_bytes = open(args.input_bin, 'rb').read()
            print(f'[+] input-bin : {args.input_bin}  ({len(input_bytes)} bytes)')
        else:
            print(f'[!] input-bin não encontrado: {args.input_bin}')

    # ── Gera imagens ──────────────────────────────────────────────────────────
    print(f'[+] Gerando imagens em: {output_dir}')
    for local_i, global_i in enumerate(indices):
        result_map = _read_bin_map(result_bytes, global_i, H, W) if result_bytes else None
        if input_bytes:
            input_map = _read_bin_map(input_bytes, global_i, H, W)
            if input_map is None:
                input_map = _find_input_block(input_bytes, H, W)
        else:
            input_map = None

        yt   = Y[global_i] if Y is not None else None
        path = plot_comparison(
            global_idx = global_i,
            x          = X[global_i],
            y_true     = yt,
            pred1_raw  = preds1[local_i],
            pred2_raw  = preds2[local_i],
            result_map = result_map,
            input_map  = input_map,
            title_gt   = args.name_gt,
            title_m1   = args.name1,
            title_m2   = args.name2,
            title_bin  = args.name_bin,
            threshold  = args.threshold,
            output_dir = output_dir,
            prefix     = args.prefix,
            layout     = args.layout,
            show_ticks = not args.no_ticks,
        )
        print(f'    [{local_i+1:>3}/{len(indices)}]  {path}')

    print(f'\n[OK] {len(indices)} imagens geradas em: {output_dir}')


if __name__ == '__main__':
    main()
