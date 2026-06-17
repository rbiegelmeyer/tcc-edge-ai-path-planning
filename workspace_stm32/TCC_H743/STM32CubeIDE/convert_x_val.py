"""
Converte um CSV de mapas para x_val_data.h

Uso:
  python convert_x_val.py <arquivo.csv> [--out <destino.h>] [--max <N>]

Exemplos:
  python convert_x_val.py W064xH064_D03_S100000_E100100.csv
  python convert_x_val.py maps.csv --out ../Core/Inc/x_val_data.h --max 40

Codificação do mapa (coluna 'map'):
  0 = livre
  1 = obstáculo  → canal 0 = 127
  2 = path       → removido (tratado como 0)
  3 = start      → canal 1 = 127
  4 = goal       → canal 2 = 127

Células sem marcador recebem -128 no canal correspondente.

Formato de saída: int8, HWC, scale=1/255, zero_point=-128
  índice = (row * W + col) * 3 + canal
"""

import argparse
import csv
import os

H, W, C   = 64, 64, 3
ITEM_SIZE = H * W * C   # 12288

ON  =  127   # int8 equivalente a float 1.0  (1.0 * 255 - 128)
OFF = -128   # int8 equivalente a float 0.0  (0.0 * 255 - 128)


def map_to_channels(map_str):
    """Converte string flat do mapa em array HWC int8 de 12288 bytes."""
    assert len(map_str) == H * W, f"Tamanho inesperado: {len(map_str)}"

    buf = bytearray(ITEM_SIZE)
    for idx, ch in enumerate(map_str):
        cell = int(ch)
        base = idx * 3
        buf[base + 0] = ON  & 0xFF if cell == 1 else OFF & 0xFF  # obstáculo
        buf[base + 1] = ON  & 0xFF if cell == 3 else OFF & 0xFF  # start
        buf[base + 2] = ON  & 0xFF if cell == 4 else OFF & 0xFF  # goal
        # cell == 2 (path) → todos os canais = OFF (removido)
    return buf


def fmt_signed(byte_val):
    """Converte byte sem sinal para representação int8 com sinal."""
    v = byte_val if byte_val < 128 else byte_val - 256
    return str(v)


def main():
    parser = argparse.ArgumentParser(description='Converte CSV de mapas para header C (x_val_data.h)')
    parser.add_argument('csv', help='Arquivo CSV de entrada')
    parser.add_argument('--out', default='../Core/Inc/x_val_data.h', help='Arquivo .h de saída (padrão: ../Core/Inc/x_val_data.h)')
    parser.add_argument('--max', type=int, default=None, help='Número máximo de amostras a converter')
    args = parser.parse_args()

    rows = []
    with open(args.csv, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if args.max is not None:
        rows = rows[:args.max]

    N = len(rows)
    print(f"Lidos {N} mapas de '{args.csv}'")
    print(f"Tamanho total estimado: {N * ITEM_SIZE / 1024:.1f} KB")

    if N * ITEM_SIZE > 1_400_000:
        print("AVISO: mais de 1,4 MB — pode não caber na Flash junto com os pesos.")
        print("       Use --max para limitar o número de amostras (ex: --max 40)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    with open(args.out, 'w') as f:
        f.write('/* Auto-gerado por convert_x_val.py — nao editar manualmente */\n')
        f.write('#ifndef X_VAL_DATA_H\n')
        f.write('#define X_VAL_DATA_H\n\n')
        f.write('#include <stdint.h>\n\n')
        f.write(f'#define X_VAL_COUNT     {N}\n')
        f.write(f'#define X_VAL_ITEM_SIZE {ITEM_SIZE}\n\n')
        f.write(f'static const int8_t x_val_data[{N}][{ITEM_SIZE}] = {{\n')

        for i, row in enumerate(rows):
            buf   = map_to_channels(row['map'])
            vals  = ', '.join(fmt_signed(b) for b in buf)
            comma = ',' if i < N - 1 else ''
            f.write(f'  /* id={row["id"]} start=({row["start_x"]},{row["start_y"]}) '
                    f'end=({row["end_x"]},{row["end_y"]}) */\n')
            f.write(f'  {{{vals}}}{comma}\n')

        f.write('};\n\n')
        f.write('#endif /* X_VAL_DATA_H */\n')

    size_kb = os.path.getsize(args.out) / 1024
    print(f"Gerado '{args.out}': {N} amostras, {size_kb:.0f} KB")


if __name__ == '__main__':
    main()
