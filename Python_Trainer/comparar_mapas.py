"""
comparar_mapas.py
-----------------
Gera colagens horizontais comparando o mapa esperado (SDL2 BMP) com o
mapa atingido pelo MCU (matplotlib PNG).

Uso:
    python comparar_mapas.py [--esperados DIR] [--atingido DIR] [--output DIR]
                             [--id-offset N] [--gap PX]

Saída: <output>/<id>.png — colagem lado a lado (esperado | atingido)
"""

import argparse
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image

# ── Cores SDL2 / matplotlib (Catppuccin Mocha) ─────────────────────────────
_BG_MAP    = np.array([30, 30, 46],  dtype=np.int16)  # fundo do mapa
_BG_INFO   = np.array([24, 24, 37],  dtype=np.int16)  # barra inferior SDL2
_BG_THRESH = 10                                        # tolerância de cor


# ── Helpers ─────────────────────────────────────────────────────────────────

def _nonbg_mask(arr: np.ndarray, bg: np.ndarray, thresh: int) -> np.ndarray:
    """Booleano H×W: True onde o pixel difere do fundo por mais de thresh."""
    return np.max(np.abs(arr[:, :, :3].astype(np.int16) - bg), axis=2) > thresh


# ── Crop: imagem esperada (BMP SDL2) ─────────────────────────────────────────

def crop_esperado(img: Image.Image) -> Image.Image:
    """
    Remove a barra de informações SDL2 do rodapé.
    A coluna 0 é sempre a linha de grade (10,10,18) na área do mapa e muda
    para C_INFOBG (24,24,37) exatamente na primeira linha da barra.
    Escaneia de cima para baixo até encontrar essa transição.
    """
    arr = np.array(img.convert("RGB"))
    H, W = arr.shape[:2]

    # Threshold estrito: (10,10,18) → diff 19; (24,24,37) → diff 0
    for y in range(H):
        diff = np.max(np.abs(arr[y, 0, :3].astype(np.int16) - _BG_INFO))
        if diff <= 5:
            return img.crop((0, 0, W, y))

    return img  # fallback: sem barra detectada


# ── Crop: imagem atingida (PNG matplotlib) ────────────────────────────────────

def _find_map_top(nonbg_row: np.ndarray) -> int:
    """
    Detecta a primeira linha do conteúdo do mapa (abaixo do título).
    Estratégia: título → gap (linhas com 0 pixels não-fundo) → mapa.
    Retorna o índice da primeira linha do mapa.
    """
    seen_content = False
    in_gap       = False

    for i, count in enumerate(nonbg_row):
        if count > 0 and not seen_content:
            seen_content = True          # entrou no bloco do título
        elif count == 0 and seen_content:
            in_gap = True                # entrou no gap pós-título
        elif count > 0 and in_gap:
            return i                     # primeira linha do mapa

    return 0  # fallback


def crop_atingido(img: Image.Image) -> Image.Image:
    """
    Remove o título do topo e o padding lateral, retendo apenas o grid.
    """
    arr = np.array(img.convert("RGB"))
    H, W = arr.shape[:2]

    mask        = _nonbg_mask(arr, _BG_MAP, _BG_THRESH)
    nonbg_row   = mask.sum(axis=1)
    nonbg_col   = mask.sum(axis=0)

    map_top = _find_map_top(nonbg_row)

    # Última linha com conteúdo
    rows_with_content = np.where(nonbg_row > 0)[0]
    map_bottom = int(rows_with_content[-1]) + 1 if len(rows_with_content) else H

    # Colunas: extremos com qualquer pixel não-fundo
    cols_with_content = np.where(nonbg_col > 0)[0]
    left  = int(cols_with_content[0])    if len(cols_with_content) else 0
    right = int(cols_with_content[-1]) + 1 if len(cols_with_content) else W

    return img.crop((left, map_top, right, map_bottom))


# ── Colagem ──────────────────────────────────────────────────────────────────

def make_collage(
    esp: Image.Image,
    ati: Image.Image,
    gap_px: int = 8,
) -> Image.Image:
    """
    Colagem horizontal: esperado (esq) | atingido (dir).
    Ambas redimensionadas para a maior altura entre as duas.
    """
    target_h = max(esp.size[1], ati.size[1])

    def fit_h(im: Image.Image, h: int) -> Image.Image:
        ow, oh = im.size
        return im if oh == h else im.resize((int(ow * h / oh), h), Image.LANCZOS)

    esp = fit_h(esp, target_h)
    ati = fit_h(ati, target_h)

    total_w = esp.size[0] + gap_px + ati.size[0]
    out = Image.new("RGB", (total_w, target_h), tuple(_BG_MAP.tolist()))
    out.paste(esp, (0, 0))
    out.paste(ati, (esp.size[0] + gap_px, 0))
    return out


# ── Indexação de arquivos ─────────────────────────────────────────────────────

def _build_atingido_index(atingido_dir: Path) -> Dict[int, Path]:
    pattern = re.compile(r"^map_solved_(\d+)_plot\.png$", re.IGNORECASE)
    index: Dict[int, Path] = {}
    for p in atingido_dir.glob("map_solved_*_plot.png"):
        m = pattern.match(p.name)
        if m:
            index[int(m.group(1))] = p
    return index


# ── Ponto de entrada ──────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Colagem esperado vs atingido")
    ap.add_argument("--esperados",  default="esperados",   help="Pasta com BMPs SDL2")
    ap.add_argument("--atingido",   default="atingido",    help="Pasta com PNGs matplotlib")
    ap.add_argument("--output",     default="comparativo", help="Pasta de saída")
    ap.add_argument("--id-offset",  type=int, default=100000,
                    help="Offset: id_bmp - offset = id_atingido (default: 100000)")
    ap.add_argument("--gap",        type=int, default=8,
                    help="Espaço em px entre as duas imagens na colagem (default: 8)")
    args = ap.parse_args()

    esp_dir = Path(args.esperados)
    ati_dir = Path(args.atingido)
    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True)

    ati_index = _build_atingido_index(ati_dir)
    if not ati_index:
        print(f"[ERRO] Nenhum 'map_solved_*_plot.png' encontrado em: {ati_dir}")
        return

    bmps = sorted(esp_dir.glob("*.bmp"))
    if not bmps:
        print(f"[ERRO] Nenhum BMP encontrado em: {esp_dir}")
        return

    ok = skip = 0
    for bmp_path in bmps:
        try:
            esp_id = int(bmp_path.stem)
        except ValueError:
            print(f"[SKIP] Nome inesperado: {bmp_path.name}")
            skip += 1
            continue

        ati_id   = esp_id - args.id_offset
        ati_path = ati_index.get(ati_id)
        if ati_path is None:
            print(f"[SKIP] Sem par para {bmp_path.name}  (atingido id={ati_id})")
            skip += 1
            continue

        esp_img = Image.open(bmp_path).convert("RGB")
        ati_img = Image.open(ati_path).convert("RGB")

        esp_c = crop_esperado(esp_img)
        ati_c = crop_atingido(ati_img)

        combined = make_collage(esp_c, ati_c, gap_px=args.gap)
        w, h = combined.size
        combined = combined.crop((0, 0, w - 160, h))
        out_path = out_dir / f"{esp_id}.png"
        combined.save(out_path, optimize=True)
        print(f"[OK] {out_path.name}  esp={esp_c.size}  ati={ati_c.size}  "
              f"colagem={combined.size}")
        ok += 1

    print(f"\nConcluído: {ok} colagens geradas, {skip} ignorados.")


if __name__ == "__main__":
    main()
