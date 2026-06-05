"""
plot_result.py
--------------
Converts a STM32 memory dump (.bin) into a visual plot of the A* result map.

Usage:
    python plot_result.py                          # uses matrix_data.bin
    python plot_result.py <result.bin>             # custom result dump
    python plot_result.py <result.bin> <input.bin> # overlay walls from input dump

The script auto-detects the 64x64 map inside the dump by finding the first
4096-byte aligned block whose values are strictly in {0,1,2,3,4} and that
contains at least one path cell (2), one start (3), and one goal (4).

Cell encoding:
    0 = empty (not on path)
    1 = wall
    2 = path
    3 = start
    4 = end / goal
"""

import sys
import os
from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm

# ── Catppuccin Mocha palette (matches the SDL2 visualiser) ──────────────────
# Normalised to [0,1] for matplotlib
COLORS = {
    0: (30  / 255, 30  / 255, 46  / 255),  # empty   – dark navy
    1: (69  / 255, 71  / 255, 90  / 255),  # wall    – slate grey
    2: (137 / 255, 220 / 255, 235 / 255),  # path    – sky blue
    3: (137 / 255, 180 / 255, 250 / 255),  # start   – lavender blue
    4: (250 / 255, 179 / 255, 135 / 255),  # end     – peach
}

CMAP  = ListedColormap([COLORS[i] for i in range(5)])
NORM  = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], ncolors=5)

MAP_H = 64
MAP_W = 64
MAP_SIZE = MAP_H * MAP_W  # 4096 bytes


# ── Helpers ──────────────────────────────────────────────────────────────────

def find_map_block(data: bytes, need_walls: bool = False) -> Tuple[int, np.ndarray]:
    """
    Scan `data` for the first 4-byte-aligned block of MAP_SIZE bytes
    whose values are in {0,1,2,3,4}.

    If need_walls=False (result map): must contain 2, 3, and 4.
    If need_walls=True  (input map):  must contain 1, 3, and 4.

    Returns (offset, array shaped (MAP_H, MAP_W), dtype int8).
    Raises ValueError if no valid block is found.
    """
    valid = {0, 1, 2, 3, 4}
    for off in range(0, len(data) - MAP_SIZE + 1, 4):
        block = data[off: off + MAP_SIZE]
        s = set(block)
        if not s <= valid:
            continue
        if need_walls:
            if {1, 3, 4} <= s:
                arr = np.frombuffer(block, dtype=np.int8).reshape(MAP_H, MAP_W)
                return off, arr
        else:
            if {2, 3, 4} <= s:
                arr = np.frombuffer(block, dtype=np.int8).reshape(MAP_H, MAP_W)
                return off, arr
    raise ValueError("No valid A* map block found in the binary dump.")


def load_bin(path: str, need_walls: bool = False) -> Tuple[int, np.ndarray]:
    with open(path, "rb") as f:
        data = f.read()
    return find_map_block(data, need_walls=need_walls)


def merge_walls(result: np.ndarray, input_map: np.ndarray) -> np.ndarray:
    """Overlay walls from input_map onto result so they appear in the plot."""
    merged = result.copy()
    merged[input_map == 1] = 1
    return merged


def plot(grid: np.ndarray, title: str, output_path: Optional[str] = None) -> None:
    h, w = grid.shape

    # Count cells per category
    counts = {v: int(np.sum(grid == v)) for v in range(5)}
    path_len = counts[2]

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor(COLORS[0])
    ax.set_facecolor(COLORS[0])

    ax.imshow(grid, cmap=CMAP, norm=NORM, interpolation="nearest",
              origin="upper", aspect="equal")

    # Grid lines
    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.tick_params(which="minor", length=0)
    ax.grid(which="minor", color=(10/255, 10/255, 18/255), linewidth=0.3)
    ax.set_xticks([])
    ax.set_yticks([])

    # Legend
    labels = {0: "Empty", 1: "Wall", 2: f"Path ({path_len} cells)",
              3: "Start", 4: "End"}
    patches = [mpatches.Patch(facecolor=COLORS[v],
                              edgecolor="white", linewidth=0.5,
                              label=labels[v])
               for v in range(5) if counts[v] > 0]
    legend = ax.legend(handles=patches, loc="upper right",
                       framealpha=0.85, fontsize=11,
                       facecolor=(24/255, 24/255, 37/255),
                       labelcolor="white")

    ax.set_title(title, color="white", fontsize=14, pad=12)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved: {output_path}")

    plt.show()


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    default_bin = os.path.join(os.path.dirname(__file__), "..", "AStar", "matrix_data.bin")
    result_bin = sys.argv[1] if len(sys.argv) > 1 else default_bin
    input_bin  = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.isfile(result_bin):
        print(f"Error: file not found: {result_bin}")
        sys.exit(1)

    print(f"Reading result: {result_bin}")
    r_off, result_grid = load_bin(result_bin, need_walls=False)
    print(f"  Found result map at offset 0x{r_off:06X}")

    grid = result_grid.copy()
    title = f"A* Result  —  {result_bin}"

    if input_bin:
        if not os.path.isfile(input_bin):
            print(f"Warning: input file not found ({input_bin}), skipping wall overlay.")
        else:
            print(f"Reading input: {input_bin}")
            i_off, input_grid = load_bin(input_bin, need_walls=True)
            print(f"  Found input map at offset 0x{i_off:06X}")
            grid  = merge_walls(result_grid, input_grid)
            title = f"A* Result + Walls  —  {result_bin}"

    # Summary
    counts = {v: int(np.sum(grid == v)) for v in range(5)}
    print(f"\nCell counts: empty={counts[0]}  wall={counts[1]}  "
          f"path={counts[2]}  start={counts[3]}  end={counts[4]}")

    # Locate start and end pixel positions
    start_pos = tuple(int(x) for x in np.argwhere(grid == 3)[0]) if counts[3] else None
    end_pos   = tuple(int(x) for x in np.argwhere(grid == 4)[0]) if counts[4] else None
    if start_pos:
        print(f"Start : row={start_pos[0]}  col={start_pos[1]}")
    if end_pos:
        print(f"End   : row={end_pos[0]}  col={end_pos[1]}")

    out_png = os.path.splitext(result_bin)[0] + "_plot.png"
    plot(grid, title, output_path=out_png)


if __name__ == "__main__":
    main()
