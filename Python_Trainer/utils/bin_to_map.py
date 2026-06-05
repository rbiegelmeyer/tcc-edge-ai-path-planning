"""
bin_to_map.py
-------------
Converts a 64x64 int8_t binary path-planning dump from the STM32H743
into a PNG heatmap image.

Colour scale: temperature gradient (inferno)
    -128 (0x80) : darkest  — empty / background
     127 (0x7F) : brightest — path / active cells

Optionally overlays the input map to mark walls, start, and end.

Usage:
    python bin_to_map.py result.bin
    python bin_to_map.py result.bin --input input.bin
    python bin_to_map.py result.bin --input input.bin --height 32 --width 32
"""

import sys
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BIN = os.path.join(_HERE, "..", "AStar", "temp.bin")

# ── Paleta clara (mesma temática do infer.py) ────────────────────────────────
_CLR_WALL  = (0.40, 0.40, 0.40)   # cinza escuro (igual _OBST_COLOR do infer.py)
_CLR_START = "limegreen"
_CLR_END   = "red"

_FIG_BG    = "white"
_GRID_CLR  = "#cccccc"
_SPINE_CLR = "#aaaaaa"
_TEXT_CLR  = "black"


# ── Carregamento ──────────────────────────────────────────────────────────────

def load_heatmap(path: str, height: int, width: int) -> np.ndarray:
    """Lê `height*width` bytes como int8 (saída do modelo / heatmap)."""
    with open(path, "rb") as f:
        raw = f.read(height * width)
    if len(raw) < height * width:
        raise ValueError(f"File too small: {len(raw)} bytes, need {height * width}")
    return np.frombuffer(raw, dtype=np.int8).reshape(height, width)


def find_input_block(data: bytes, height: int, width: int) -> np.ndarray:
    """
    Varre `data` em busca de um bloco height*width com valores em {0,1,2,3,4}
    que contenha início (3) e fim (4). Retorna o array (H, W) int8.
    """
    size  = height * width
    valid = {0, 1, 2, 3, 4}
    for off in range(0, len(data) - size + 1, 4):
        block = data[off: off + size]
        s = set(block)
        if s <= valid and {3, 4} <= s:
            return np.frombuffer(block, dtype=np.int8).reshape(height, width)
    raise ValueError("Nenhum bloco de mapa de entrada válido encontrado no dump.")


def load_input(path: str, height: int, width: int) -> np.ndarray:
    with open(path, "rb") as f:
        data = f.read()
    return find_input_block(data, height, width)


# ── Visualização ──────────────────────────────────────────────────────────────

def _overlay_input(ax, input_grid: np.ndarray) -> list:
    """
    Sobrepõe paredes, início e fim do mapa de entrada sobre o heatmap.
    Retorna lista de handles para o legend.
    """
    h, w = input_grid.shape
    handles = []

    # Paredes — overlay semi-transparente cinza escuro
    wall_rgba = np.zeros((h, w, 4), dtype=np.float32)
    wall_rgba[:, :, 0] = _CLR_WALL[0]
    wall_rgba[:, :, 1] = _CLR_WALL[1]
    wall_rgba[:, :, 2] = _CLR_WALL[2]
    wall_rgba[:, :, 3] = np.where(input_grid == 1, 0.70, 0.0)
    ax.imshow(wall_rgba, interpolation="nearest", origin="upper", aspect="equal")
    handles.append(mpatches.Patch(facecolor=_CLR_WALL, alpha=0.70,
                                  edgecolor="none", label="Obstáculo"))

    # Início (3) — limegreen ○ com borda preta (igual ao infer.py)
    starts = np.argwhere(input_grid == 3)
    if len(starts):
        r, c = starts[0]
        ax.scatter(c, r, s=140, color=_CLR_START, marker="o",
                   edgecolors="black", linewidths=1.5, zorder=10)
        handles.append(mpatches.Patch(facecolor=_CLR_START, edgecolor="black",
                                      linewidth=1, label="Início"))

    # Fim (4) — red ✕ com borda darkred (igual ao infer.py)
    ends = np.argwhere(input_grid == 4)
    if len(ends):
        r, c = ends[0]
        ax.scatter(c, r, s=140, color=_CLR_END, marker="X",
                   edgecolors="darkred", linewidths=1.5, zorder=10)
        handles.append(mpatches.Patch(facecolor=_CLR_END, edgecolor="darkred",
                                      linewidth=1, label="Final"))

    return handles


def plot_map(grid: np.ndarray, title: str, out_png: str,
             input_grid: np.ndarray = None) -> None:
    fig, ax = plt.subplots(figsize=(9, 9))
    fig.patch.set_facecolor(_FIG_BG)
    ax.set_facecolor(_FIG_BG)

    im = ax.imshow(
        grid,
        cmap="OrRd",
        vmin=-128,
        vmax=127,
        interpolation="nearest",
        origin="upper",
        aspect="equal",
    )

    # Grade fina (por célula)
    h, w = grid.shape
    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.tick_params(which="minor", length=0)
    ax.grid(which="minor", color=_GRID_CLR, linewidth=0.25)

    # Ticks maiores a cada 8 células
    ax.set_xticks(range(0, w, 8))
    ax.set_yticks(range(0, h, 8))
    ax.tick_params(colors=_TEXT_CLR, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(_SPINE_CLR)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color=_TEXT_CLR, labelsize=8)
    cbar.outline.set_edgecolor(_SPINE_CLR)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=_TEXT_CLR)
    cbar.set_label("int8 value  (−128 = empty · 127 = path)",
                   color=_TEXT_CLR, fontsize=9, labelpad=8)

    # Overlay do mapa de entrada (paredes, início, fim)
    extra_handles = []
    if input_grid is not None:
        extra_handles = _overlay_input(ax, input_grid)

    if extra_handles:
        ax.legend(
            handles=extra_handles,
            loc="upper right",
            framealpha=0.85,
            fontsize=10,
            facecolor="white",
            labelcolor=_TEXT_CLR,
        )

    ax.set_title(title, color=_TEXT_CLR, fontsize=13, pad=10)
    fig.tight_layout()

    fig.savefig(out_png, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved : {out_png}")
    plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Converte dump int8 do STM32 em heatmap PNG.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("bin",  nargs="?", default=DEFAULT_BIN,
                   help="Arquivo .bin de saída do modelo (padrão: ../AStar/temp.bin)")
    p.add_argument("--input", "-i", default=None, metavar="INPUT_BIN",
                   help="Mapa de entrada .bin com paredes/início/fim (valores 0-4)")
    p.add_argument("--height", "-H", type=int, default=64)
    p.add_argument("--width",  "-W", type=int, default=64)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.bin):
        print(f"Error: file not found: {args.bin}")
        sys.exit(1)

    print(f"Heatmap : {args.bin}")
    grid = load_heatmap(args.bin, args.height, args.width)

    from collections import Counter
    counts = Counter(int(v) for v in grid.flatten())
    print(f"Size    : {args.height}x{args.width}")
    print("Values  :")
    for v, c in sorted(counts.items()):
        print(f"  {v:5d} (0x{v & 0xFF:02X}) : {c} cells")

    input_grid = None
    if args.input:
        if not os.path.isfile(args.input):
            print(f"Warning: input file not found ({args.input}), skipping overlay.")
        else:
            print(f"Input   : {args.input}")
            input_grid = load_input(args.input, args.height, args.width)
            starts = np.argwhere(input_grid == 3)
            ends   = np.argwhere(input_grid == 4)
            if len(starts):
                print(f"  Start : row={starts[0][0]}  col={starts[0][1]}")
            if len(ends):
                print(f"  End   : row={ends[0][0]}  col={ends[0][1]}")
            print(f"  Walls : {int(np.sum(input_grid == 1))} cells")

    suffix  = "_overlay" if input_grid is not None else ""
    title   = f"{os.path.basename(args.bin)}  —  {args.height}×{args.width}  int8"
    out_png = os.path.splitext(args.bin)[0] + f"_map{suffix}.png"
    plot_map(grid, title, out_png, input_grid=input_grid)


if __name__ == "__main__":
    main()
