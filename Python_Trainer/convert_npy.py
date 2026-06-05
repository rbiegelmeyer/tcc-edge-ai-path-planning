"""
Converte arquivo .npy ou .npz de mapas para x_val_data.h

Uso:
  python convert_npy.py <arquivo.npy>  [--out <destino.h>] [--max <N>]
  python convert_npy.py <arquivo.npz>  [--key <chave>] [--out <destino.h>] [--max <N>]

Exemplos:
  python convert_npy.py X_val.npy
  python convert_npy.py data.npz --key x_val --max 40
  python convert_npy.py X_val.npy --out ./x_val_data.h

Shape esperado do array: (N, 64, 64, 3)  — HWC, 3 canais

Dtype suportado:
  float32 / float64  — valores em [0, 1]: quantiza para int8 (scale=1/255, zp=-128)
  uint8              — valores em [0, 255]: converte diretamente (int8 = uint8 - 128)
  int8               — usa diretamente, sem conversão

Codificação de canais (igual ao convert_x_val.py):
  canal 0 = obstáculos  (127 = presente, -128 = ausente)
  canal 1 = start       (127 = presente, -128 = ausente)
  canal 2 = goal        (127 = presente, -128 = ausente)
"""

import argparse
import os
import sys
import numpy as np

H, W, C   = 64, 64, 3
ITEM_SIZE = H * W * C   # 12288


def load_array(path, key):
    ext = os.path.splitext(path)[1].lower()

    if ext == '.npy':
        return np.load(path)

    if ext == '.npz':
        archive = np.load(path)
        keys = list(archive.keys())
        if key is None:
            if len(keys) == 1:
                key = keys[0]
                print(f"NPZ: usando chave '{key}'")
            else:
                print(f"NPZ contém as chaves: {keys}")
                print("Use --key para especificar qual usar.")
                sys.exit(1)
        if key not in archive:
            print(f"Erro: chave '{key}' não encontrada. Disponíveis: {keys}")
            sys.exit(1)
        return archive[key]

    print(f"Erro: extensão '{ext}' não suportada. Use .npy ou .npz")
    sys.exit(1)


def to_int8(arr):
    """Converte array para int8 conforme o dtype de entrada."""
    dtype = arr.dtype

    if dtype == np.int8:
        return arr

    if dtype == np.uint8:
        return (arr.astype(np.int16) - 128).astype(np.int8)

    if np.issubdtype(dtype, np.floating):
        quantized = np.round(arr * 255.0) - 128
        return np.clip(quantized, -128, 127).astype(np.int8)

    print(f"Erro: dtype '{dtype}' não suportado. Use float32, uint8 ou int8.")
    sys.exit(1)


def fmt_row(row_int8):
    """Converte linha numpy int8 para string CSV de valores."""
    return ', '.join(str(int(v)) for v in row_int8)


def main():
    parser = argparse.ArgumentParser(description='Converte .npy/.npz de mapas para header C (x_val_data.h)')
    parser.add_argument('npy', help='Arquivo .npy ou .npz de entrada')
    parser.add_argument('--key', default=None, help='Chave do array dentro de um .npz')
    parser.add_argument('--out', default='./x_val_data.h', help='Arquivo .h de saída (padrão: ./x_val_data.h)')
    parser.add_argument('--max', type=int, default=None, help='Número máximo de amostras a converter')
    args = parser.parse_args()

    arr = load_array(args.npy, args.key)

    if arr.ndim != 4 or arr.shape[1:] != (H, W, C):
        print(f"Erro: shape esperado (N, {H}, {W}, {C}), obtido {arr.shape}")
        sys.exit(1)

    print(f"Array carregado: shape={arr.shape}, dtype={arr.dtype}")

    if args.max is not None:
        arr = arr[:args.max]

    arr = to_int8(arr)
    N   = arr.shape[0]

    print(f"Convertendo {N} amostras...")
    print(f"Tamanho total estimado: {N * ITEM_SIZE / 1024:.1f} KB")

    if N * ITEM_SIZE > 1_400_000:
        print("AVISO: mais de 1,4 MB — pode não caber na Flash junto com os pesos.")
        print("       Use --max para limitar o número de amostras (ex: --max 40)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    flat = arr.reshape(N, -1)   # (N, 12288)

    with open(args.out, 'w') as f:
        f.write('/* Auto-gerado por convert_npy.py — nao editar manualmente */\n')
        f.write('#ifndef X_VAL_DATA_H\n')
        f.write('#define X_VAL_DATA_H\n\n')
        f.write('#include <stdint.h>\n\n')
        f.write(f'#define X_VAL_COUNT     {N}\n')
        f.write(f'#define X_VAL_ITEM_SIZE {ITEM_SIZE}\n\n')
        f.write(f'static const int8_t x_val_data[{N}][{ITEM_SIZE}] = {{\n')

        for i in range(N):
            vals  = fmt_row(flat[i])
            comma = ',' if i < N - 1 else ''
            f.write(f'  {{{vals}}}{comma}\n')

        f.write('};\n\n')
        f.write('#endif /* X_VAL_DATA_H */\n')

    size_kb = os.path.getsize(args.out) / 1024
    print(f"Gerado '{args.out}': {N} amostras, {size_kb:.0f} KB")


if __name__ == '__main__':
    main()
