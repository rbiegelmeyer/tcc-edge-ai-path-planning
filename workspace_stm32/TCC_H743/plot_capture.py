#!/usr/bin/env python3
"""
plot_capture.py
---------------
Gera gráfico comparativo a partir de um JSON de captura do STM32
(produzido por stm32_debug_link.py --capture --out captura.json).

Painéis (sem --npz):
  1. Entrada     — mapa de paredes + marcadores início/fim  (stai_input[0])
  2. Saída STM32 — heatmap int8 com paredes/início/fim sobrepostos (stai_output[0])

Painéis (com --npz):
  1. Entrada     — stai_input[0] decodificado (obstáculos + início/fim)
  2. Esperado    — x[x_val_idx] com caminho A* (y[x_val_idx]) sobreposto
  3. Saída STM32 — heatmap stai_output[0] com overlay de obstáculos/início/fim

O NPZ deve ser gerado por csv_to_npz.py e conter as chaves:
  x  — (N, 64, 64, 3) float32
  y  — (N, 64, 64, 1) float32   (caminho A*)
  ids — (N,) int32               (IDs globais dos mapas)

Uso:
    python plot_capture.py captura.json
    python plot_capture.py captura.json --npz W064xH064_D03_S100000_E100100.npz
    python plot_capture.py captura.json --npz dados.npz --out resultado.png --no-ticks
"""

import argparse
import json
import numpy as np
import matplotlib
if not matplotlib.is_interactive():
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path


# ── Paleta ────────────────────────────────────────────────────────────────────
_PATH_COLOR = np.array([0.95, 0.45, 0.00], dtype=np.float32)
_OBST_COLOR = np.array([0.40, 0.40, 0.40], dtype=np.float32)
_WALL_RGBA  = (0.40, 0.40, 0.40)

_FIG_BG    = 'white'
_GRID_CLR  = '#cccccc'
_SPINE_CLR = '#aaaaaa'
_TEXT_CLR  = 'black'

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
_LEGEND_PATH = mlines.Line2D([], [], color=_PATH_COLOR, linewidth=3,
                              label='Caminho A*')


# ── Helpers visuais ───────────────────────────────────────────────────────────
def _make_img(obst, path_mask=None):
    """RGB float32: livre=branco, parede=cinza, caminho=laranja."""
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


def _overlay_map_on_heatmap(ax, obst, start_yx, end_yx):
    """Sobrepõe obstáculos (alpha sólido) e marcadores sobre um heatmap."""
    H, W = obst.shape
    wall_rgba = np.zeros((H, W, 4), dtype=np.float32)
    wall_rgba[:, :, :3] = np.array(_WALL_RGBA, dtype=np.float32)
    wall_rgba[:, :, 3]  = (obst > 0.5).astype(np.float32)
    ax.imshow(wall_rgba, interpolation='nearest', origin='upper', aspect='equal')
    _add_markers(ax, start_yx, end_yx)


# ── Decode stai_input ─────────────────────────────────────────────────────────
def decode_input(in_data: list, scale: float, zero_point: int):
    """Dequantiza stai_input[0] int8 → obst, start_yx, end_yx."""
    raw       = np.array(in_data, dtype=np.int8).reshape(64, 64, 3)
    float_map = (raw.astype(np.float32) - zero_point) * scale
    obst  = float_map[:, :, 0]
    start = float_map[:, :, 1]
    end   = float_map[:, :, 2]
    return obst, np.argwhere(start > 0.5), np.argwhere(end > 0.5)


def decode_output(out_data: list) -> np.ndarray:
    return np.array(out_data, dtype=np.int8).reshape(64, 64)


def decode_map_solved(solved_data: list):
    """Decodifica map_solved int8 (0-4) em componentes visuais."""
    raw      = np.array(solved_data, dtype=np.int8).reshape(64, 64)
    obst     = (raw == 1).astype(np.float32)
    path     = (raw == 2).astype(np.float32)
    start_yx = np.argwhere(raw == 3)
    end_yx   = np.argwhere(raw == 4)
    return obst, path, start_yx, end_yx


# ── Carregamento do NPZ ───────────────────────────────────────────────────────
def load_npz_sample(npz_path: str, x_val_idx: int):
    """Carrega amostra x_val_idx do NPZ gerado por csv_to_npz.py.

    Espera chaves: x [N,64,64,3], y [N,64,64,1], ids [N]
    """
    npz = np.load(npz_path)
    n   = len(npz['x'])

    if x_val_idx >= n:
        raise IndexError(
            f"x_val_idx={x_val_idx} fora do range do NPZ (n={n}). "
            f"IDs disponíveis: {npz['ids'].tolist() if 'ids' in npz.files else list(range(n))}"
        )

    x = npz['x'][x_val_idx].astype(np.float32)   # [64,64,3]
    obst  = x[:, :, 0]
    start = x[:, :, 1]
    end   = x[:, :, 2]

    y = npz['y'][x_val_idx].squeeze().astype(np.float32) if 'y' in npz.files else None

    global_id = int(npz['ids'][x_val_idx]) if 'ids' in npz.files else None

    return dict(
        obst      = obst,
        start_yx  = np.argwhere(start > 0.5),
        end_yx    = np.argwhere(end   > 0.5),
        y         = y,
        global_id = global_id,
    )


# ── Plot principal ────────────────────────────────────────────────────────────
def plot_capture(capture_json: str,
                 npz_path: str = None,
                 output_path: str = None,
                 layout: str = 'horizontal',
                 show_ticks: bool = True,
                 show_interactive: bool = False,
                 window_pos: str = None,
                 fig=None) -> str:

    with open(capture_json, encoding='utf-8') as f:
        cap = json.load(f)

    inp = cap['stai_input']
    out = cap['stai_output']
    x_val_idx = cap.get('x_val_idx')

    obst, start_yx, end_yx = decode_input(
        inp['data'], inp['scale'], inp['zero_point']
    )
    output_map = decode_output(out['data'])

    # Tenta carregar amostra do NPZ
    val = None
    if npz_path and x_val_idx is not None:
        try:
            val = load_npz_sample(npz_path, x_val_idx)
        except Exception as e:
            print(f'[!] NPZ ignorado: {e}')

    # Decodifica map_solved se presente no JSON
    solved = None
    if 'map_solved' in cap:
        try:
            solved_raw = cap['map_solved']
            s_obst, s_path, s_start_yx, s_end_yx = decode_map_solved(solved_raw['data'])
            solved = dict(obst=s_obst, path=s_path,
                          start_yx=s_start_yx, end_yx=s_end_yx)
        except Exception as e:
            print(f'[!] map_solved ignorado: {e}')

    # Layout de painéis:
    #   Com NPZ: [Ground Truth (NPZ) | A* STM32 (map_solved, se presente) | Saída NN STM32]
    #   Sem NPZ: [Entrada (stai_input) | Saída STM32]
    if val is not None:
        n_panels = 3 if solved is not None else 2
    else:
        n_panels = 2
    H, W = 64, 64

    if layout == 'vertical':
        adjust_kw = dict(hspace=0.10)
        if fig is None:
            fig, axes = plt.subplots(n_panels, 1,
                                     figsize=(4.5, 4.5 * n_panels))
        else:
            fig.clf()
            axes = fig.subplots(n_panels, 1)
    else:
        adjust_kw = dict(wspace=0.14)
        if fig is None:
            fig, axes = plt.subplots(1, n_panels,
                                     figsize=(4.5 * n_panels + 0.5, 4.5))
        else:
            fig.clf()
            axes = fig.subplots(1, n_panels)

    axes = list(axes)
    fig.patch.set_facecolor(_FIG_BG)

    idx_str = f'  |  x_val_idx={x_val_idx}' if x_val_idx is not None else ''
    fig.suptitle(
        f'Captura STM32H743{idx_str}',
        fontsize=10, color=_TEXT_CLR, y=0.98,
    )

    panel_idx = 0

    if val is not None:
        # ── Painel 1: Ground Truth (NPZ x + y) ───────────────────────────────
        ax = axes[panel_idx]; panel_idx += 1
        ax.set_facecolor(_FIG_BG)
        ax.imshow(_make_img(val['obst'], val['y']),
                  interpolation='nearest', aspect='equal', origin='upper')
        _add_markers(ax, val['start_yx'], val['end_yx'])
        id_info = f'  (ID: {val["global_id"]})' if val['global_id'] is not None else ''
        ax.set_title('Ground Truth' + id_info, fontsize=11, color=_TEXT_CLR)
        _apply_grid(ax, H, W, show_ticks)
        legend_items = _LEGEND_ELEMENTS + ([_LEGEND_PATH] if val['y'] is not None else [])
        ax.legend(handles=legend_items, loc='upper right', fontsize=8,
                  framealpha=0.85, facecolor='white', labelcolor=_TEXT_CLR)

        # ── Painel 2 (opcional): A* STM32 (map_solved) ───────────────────────
        if solved is not None:
            ax = axes[panel_idx]; panel_idx += 1
            ax.set_facecolor(_FIG_BG)
            ax.imshow(_make_img(val['obst'], solved['path']),
                      interpolation='nearest', aspect='equal', origin='upper')
            _add_markers(ax, val['start_yx'], val['end_yx'])
            ax.set_title('A* STM32  (map_solved)', fontsize=11, color=_TEXT_CLR)
            _apply_grid(ax, H, W, show_ticks)
            legend_items = _LEGEND_ELEMENTS + [_LEGEND_PATH]
            ax.legend(handles=legend_items, loc='upper right', fontsize=8,
                      framealpha=0.85, facecolor='white', labelcolor=_TEXT_CLR)

    else:
        # ── Painel 1 (sem NPZ): Entrada (stai_input[0]) ──────────────────────
        ax = axes[panel_idx]; panel_idx += 1
        ax.set_facecolor(_FIG_BG)
        ax.imshow(_make_img(obst), interpolation='nearest',
                  aspect='equal', origin='upper')
        _add_markers(ax, start_yx, end_yx)
        ax.set_title('Entrada  (stai_input[0])', fontsize=11, color=_TEXT_CLR)
        _apply_grid(ax, H, W, show_ticks)
        ax.legend(handles=_LEGEND_ELEMENTS, loc='upper right', fontsize=8,
                  framealpha=0.85, facecolor='white', labelcolor=_TEXT_CLR)

    # ── Último painel: Saída NN STM32 (stai_output[0]) ───────────────────────
    ax = axes[panel_idx]
    ax.set_facecolor(_FIG_BG)
    im = ax.imshow(output_map.astype(np.float32),
                   cmap=_HEATMAP_CMAP, vmin=-128, vmax=127,
                   interpolation='nearest', origin='upper', aspect='equal')

    # Overlay de mapa: usa NPZ (mais limpo) se disponível, senão stai_input
    if val is not None:
        _overlay_map_on_heatmap(ax, val['obst'], val['start_yx'], val['end_yx'])
    else:
        _overlay_map_on_heatmap(ax, obst, start_yx, end_yx)

    cax  = ax.inset_axes([1.02, 0.05, 0.04, 0.90])
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.yaxis.set_tick_params(color=_TEXT_CLR, labelsize=6)
    cbar.outline.set_edgecolor(_SPINE_CLR)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=_TEXT_CLR)
    cbar.set_label('int8', color=_TEXT_CLR, fontsize=7, labelpad=4)
    ax.set_title('Saída STM32  (stai_output[0])', fontsize=11, color=_TEXT_CLR)
    _apply_grid(ax, H, W, show_ticks)

    plt.tight_layout(pad=0.4, rect=[0, 0, 1, 0.94])
    fig.subplots_adjust(**adjust_kw)

    if output_path is None:
        output_path = str(Path(capture_json).with_suffix('.png'))

    plt.savefig(output_path, dpi=150, facecolor=_FIG_BG)
    print(f'[OK] {output_path}')

    if show_interactive:
        if window_pos:
            try:
                plt.get_current_fig_manager().window.wm_geometry(window_pos)
            except Exception:
                pass
        plt.draw()
        plt.pause(0.001)
    else:
        plt.close()

    return output_path


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description='Gráfico comparativo de captura STM32 (stai_input / stai_output).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('capture_json',
                   help='JSON gerado por stm32_debug_link.py --capture --out')
    p.add_argument('--npz', '-n', default=None, metavar='FILE.npz',
                   help='NPZ gerado por csv_to_npz.py (chaves: x, y, ids)')
    p.add_argument('--out', '-o', default=None,
                   help='PNG de saída (default: mesmo nome do JSON, extensão .png)')
    p.add_argument('--layout', '-l', default='horizontal',
                   choices=['horizontal', 'vertical'],
                   help='Distribuição dos painéis (default: horizontal)')
    p.add_argument('--no-ticks', action='store_true',
                   help='Ocultar números dos ticks')
    args = p.parse_args()

    plot_capture(
        capture_json = args.capture_json,
        npz_path     = args.npz,
        output_path  = args.out,
        layout       = args.layout,
        show_ticks   = not args.no_ticks,
    )


if __name__ == '__main__':
    main()
