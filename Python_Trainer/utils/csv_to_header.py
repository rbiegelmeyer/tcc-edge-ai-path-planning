#!/usr/bin/env python3
"""
csv_to_header.py — Converte um dataset CSV de mapas para um único .h com dados para o A*.

O .h gerado contém os arrays static const diretamente, sem necessidade de um .c separado.
Inclua-o em somente UM arquivo .c para evitar duplicação de dados em flash.

Uso:
    python csv_to_header.py W064xH064_D03_S100000_E100100.csv
    python csv_to_header.py dataset.csv -o test_maps.h -n 50 --skip 10
    python csv_to_header.py dataset.csv --ids 0 5 12 37
"""

import argparse
import csv
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description='Converte CSV de mapas para header C para o A*.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('csv',
                   help='Arquivo CSV de entrada (colunas: id,difficulty,start_x,start_y,end_x,end_y,height,width,map)')
    p.add_argument('--output', '-o', default=None,
                   help='Arquivo .h de saída (padrão: <stem>_maps.h)')
    p.add_argument('--count', '-n', type=int, default=None,
                   help='Número de mapas a exportar (padrão: todos)')
    p.add_argument('--skip', type=int, default=0,
                   help='Pular os primeiros N mapas do CSV (padrão: 0)')
    p.add_argument('--ids', type=int, nargs='+', default=None,
                   help='Exportar somente os mapas com estes índices de linha (0-based, ignora --skip/--count)')
    p.add_argument('--guard', default=None,
                   help='Nome do include guard (padrão: derivado do nome do .h)')
    return p.parse_args()


def read_records(csv_path, skip, count, ids):
    """Lê registros do CSV e aplica os filtros de seleção."""
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    if ids is not None:
        bad = [i for i in ids if i >= len(all_rows)]
        if bad:
            print(f'[!] Índices fora do range (CSV tem {len(all_rows)} linhas): {bad}')
            sys.exit(1)
        return [all_rows[i] for i in ids], ids
    else:
        sliced = all_rows[skip:]
        if count is not None:
            sliced = sliced[:count]
        original_indices = list(range(skip, skip + len(sliced)))
        return sliced, original_indices


def format_array(values, width, indent='    '):
    """Uma linha C por linha do mapa, alinhada."""
    rows = []
    for r in range(0, len(values), width):
        chunk = values[r:r + width]
        rows.append(indent + ','.join(str(v) for v in chunk))
    return ',\n'.join(rows)


def build_header(records, orig_indices, csv_name, out_name, guard):
    out = []

    def w(*args):
        out.append(' '.join(str(a) for a in args))

    def blank():
        out.append('')

    first = records[0]
    height = int(first['height'])
    width  = int(first['width'])
    count  = len(records)

    # ── Cabeçalho do arquivo ──────────────────────────────────────────────────
    w(f'/**')
    w(f' * @file {out_name}')
    w(f' * @brief Mapas para validação do A* — gerado automaticamente.')
    w(f' *')
    w(f' * Fonte  : {csv_name}')
    w(f' * Mapas  : {count}  (índices CSV: {orig_indices[0]}..{orig_indices[-1]})')
    w(f' * Grade  : {width}x{height}')
    w(f' * Células: 0=vazio  1=parede  2=caminho  3=início  4=fim')
    w(f' *')
    w(f' * ATENÇÃO: inclua este arquivo em somente UM .c para evitar')
    w(f' *          duplicação de dados na memória flash.')
    w(f' * NÃO EDITE — re-gere com csv_to_header.py se necessário.')
    w(f' */')
    blank()
    w(f'#ifndef {guard}')
    w(f'#define {guard}')
    blank()
    w(f'#include <stdint.h>')
    blank()

    # ── Constantes ────────────────────────────────────────────────────────────
    w(f'#define TEST_MAP_COUNT  {count}')
    w(f'#define TEST_MAP_HEIGHT {height}')
    w(f'#define TEST_MAP_WIDTH  {width}')
    w(f'#define TEST_MAP_SIZE   (TEST_MAP_HEIGHT * TEST_MAP_WIDTH)')
    blank()

    # ── Struct de metadados ───────────────────────────────────────────────────
    w(f'typedef struct {{')
    w(f'    uint32_t id;')
    w(f'    int16_t  start_row;   /* start_y no CSV */')
    w(f'    int16_t  start_col;   /* start_x no CSV */')
    w(f'    int16_t  end_row;     /* end_y no CSV */')
    w(f'    int16_t  end_col;     /* end_x no CSV */')
    w(f'}} TestMapMeta;')
    blank()

    # ── Array único de mapas ──────────────────────────────────────────────────
    w(f'static const int8_t test_maps[TEST_MAP_COUNT][TEST_MAP_SIZE] = {{')
    for idx, rec in enumerate(records):
        map_id    = int(rec['id'])
        h         = int(rec['height'])
        ww        = int(rec['width'])
        start_col = int(rec['start_x'])
        start_row = int(rec['start_y'])
        end_col   = int(rec['end_x'])
        end_row   = int(rec['end_y'])

        raw = rec['map'].strip()
        if len(raw) != h * ww:
            print(f'[!] id={map_id}: tamanho do mapa {len(raw)} != {h}x{ww}={h*ww}')
            sys.exit(1)
        values = [int(c) for c in raw]

        comma = ',' if idx < count - 1 else ''
        w(f'    /* [{idx}] id={map_id}  start=({start_row},{start_col})  end=({end_row},{end_col}) */')
        w(f'    {{')
        w(format_array(values, ww, indent='        '))
        w(f'    }}{comma}')
    w(f'}};')
    blank()

    # ── Tabela de metadados ───────────────────────────────────────────────────
    w(f'static const TestMapMeta test_maps_meta[TEST_MAP_COUNT] = {{')
    for idx, rec in enumerate(records):
        map_id    = int(rec['id'])
        start_col = int(rec['start_x'])
        start_row = int(rec['start_y'])
        end_col   = int(rec['end_x'])
        end_row   = int(rec['end_y'])
        comma = ',' if idx < count - 1 else ''
        w(f'    {{ {map_id:>7}, {start_row:>3}, {start_col:>3}, {end_row:>3}, {end_col:>3} }}{comma}')
    w(f'}};')
    blank()
    w(f'#endif /* {guard} */')

    return '\n'.join(out)


def main():
    args = parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f'[!] Arquivo não encontrado: {csv_path}')
        sys.exit(1)

    out_path = Path(args.output) if args.output else Path.cwd() / (csv_path.stem + '_maps.h')
    guard    = args.guard or (out_path.stem.upper().replace('-', '_') + '_H')

    print(f'[+] Lendo {csv_path.name}...')
    records, orig_indices = read_records(csv_path, args.skip, args.count, args.ids)

    if not records:
        print('[!] Nenhum mapa selecionado.')
        sys.exit(1)

    height = int(records[0]['height'])
    width  = int(records[0]['width'])
    print(f'    {len(records)} mapas  |  {width}x{height}  |  saída: {out_path.name}')

    content = build_header(records, orig_indices, csv_path.name, out_path.name, guard)

    out_path.write_text(content, encoding='utf-8')
    size_kb = out_path.stat().st_size / 1024
    print(f'[OK] {out_path}  ({size_kb:.1f} KB)')


if __name__ == '__main__':
    main()
