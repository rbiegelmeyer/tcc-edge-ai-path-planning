"""
Converte CSV de mapas para arquivo .npz

Uso:
  python csv_to_npz.py <arquivo.csv> [--out <destino.npz>] [--max <N>]

Exemplos:
  python csv_to_npz.py W064xH064_D03_S100000_E100100.csv
  python csv_to_npz.py maps.csv --out dataset.npz --max 100

Arrays salvos no .npz:
  x        — (N, 64, 64, 3) float32 [0, 1]: mapas codificados em 3 canais
               canal 0 = obstáculos, canal 1 = start, canal 2 = goal
               path (2) removido (tratado como livre)
  ids      — (N,) int32: IDs dos mapas
  start_x  — (N,) int16
  start_y  — (N,) int16
  end_x    — (N,) int16
  end_y    — (N,) int16

O array 'x' é compatível com convert_npy.py para gerar x_val_data.h.
"""

import argparse
import csv
import os
import numpy as np

H, W, C = 64, 64, 3


def map_to_channels(map_str):
    """Converte string flat do mapa em array (H, W, 3) float32 [0, 1]."""
    assert len(map_str) == H * W, f"Tamanho inesperado: {len(map_str)}"
    raw = np.frombuffer(map_str.encode(), dtype=np.uint8) - ord('0')  # shape (4096,)
    raw = raw.reshape(H, W)

    out = np.zeros((H, W, C), dtype=np.float32)
    out[:, :, 0] = (raw == 1).astype(np.float32)  # obstáculos
    out[:, :, 1] = (raw == 3).astype(np.float32)  # start
    out[:, :, 2] = (raw == 4).astype(np.float32)  # goal
    # raw == 2 (path) → todos os canais = 0 (removido)
    return out


def main():
    parser = argparse.ArgumentParser(description='Converte CSV de mapas para .npz')
    parser.add_argument('csv', help='Arquivo CSV de entrada')
    parser.add_argument('--out', default=None, help='Arquivo .npz de saída (padrão: <csv>.npz)')
    parser.add_argument('--max', type=int, default=None, help='Número máximo de amostras')
    args = parser.parse_args()

    if args.out is None:
        args.out = os.path.splitext(args.csv)[0] + '.npz'

    rows = []
    with open(args.csv, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if args.max is not None:
        rows = rows[:args.max]

    N = len(rows)
    print(f"Lidos {N} mapas de '{args.csv}'")

    x       = np.zeros((N, H, W, C), dtype=np.float32)
    ids     = np.zeros(N, dtype=np.int32)
    start_x = np.zeros(N, dtype=np.int16)
    start_y = np.zeros(N, dtype=np.int16)
    end_x   = np.zeros(N, dtype=np.int16)
    end_y   = np.zeros(N, dtype=np.int16)

    for i, row in enumerate(rows):
        x[i]       = map_to_channels(row['map'])
        ids[i]     = int(row['id'])
        start_x[i] = int(row['start_x'])
        start_y[i] = int(row['start_y'])
        end_x[i]   = int(row['end_x'])
        end_y[i]   = int(row['end_y'])

    np.savez_compressed(
        args.out,
        x=x,
        ids=ids,
        start_x=start_x,
        start_y=start_y,
        end_x=end_x,
        end_y=end_y,
    )

    size_kb = os.path.getsize(args.out) / 1024
    print(f"Gerado '{args.out}': {N} amostras, {size_kb:.1f} KB")
    print(f"  x.shape={x.shape}, dtype={x.dtype}")
    print(f"  Para gerar x_val_data.h: python convert_npy.py {args.out} --key x")


if __name__ == '__main__':
    main()
