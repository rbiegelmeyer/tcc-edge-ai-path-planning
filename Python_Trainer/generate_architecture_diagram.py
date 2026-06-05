"""
Gera o diagrama detalhado de arquitetura da U-Net mostrando cada operação.

Uso:
    python generate_architecture_diagram.py
    python generate_architecture_diagram.py --output minha_arquitetura.png
"""

import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ---------------------------------------------------------------
# Dimensões e posições
# ---------------------------------------------------------------
BH      = 0.52   # altura de cada bloco
BW_ENC  = 3.50   # largura dos blocos do encoder
BW_DEC  = 3.80   # largura dos blocos do decoder
BW_BN   = 4.00   # largura dos blocos do bottleneck
BW_IO   = 4.20   # largura dos blocos de entrada/saída
VGAP    = 0.18   # gap entre blocos do mesmo grupo
LGAP_E  = 1.30   # gap entre grupos do encoder (ajustado para alinhar skips)
LGAP_D  = 0.60   # gap entre grupos do decoder

EX  = 5.0    # centro X do encoder
DX  = 21.0   # centro X do decoder
BNX = 13.0   # centro X do bottleneck

# ---------------------------------------------------------------
# Cores  (fundo, texto)
# ---------------------------------------------------------------
C_IO    = ('#EBF5FB', '#1A252F')
C_CONV  = ('#D6EAF8', '#1A252F')
C_POOL  = ('#D5D8DC', '#1A252F')
C_BN    = ('#FDEBD0', '#1A252F')
C_UP    = ('#D5F5E3', '#1A252F')
C_CAT   = ('#F9EBEA', '#1A252F')
C_OUT   = ('#FDEDEC', '#1A252F')

# Cores das setas
AC_ENC  = '#2471A3'   # encoder flow
AC_DEC  = '#1E8449'   # decoder flow
AC_BN   = '#784212'   # bottleneck flow
AC_SKIP = '#7D3C98'   # skip connections
# ---------------------------------------------------------------


def _block(ax, x, y, label, colors, w=BW_ENC, h=BH, fontsize=8.0):
    fc, tc = colors
    rect = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle='round,pad=0.07',
        facecolor=fc, edgecolor='#95A5A6',
        linewidth=0.9, zorder=3,
    )
    ax.add_patch(rect)
    ax.text(x, y, label, ha='center', va='center',
            fontsize=fontsize, color=tc, zorder=4,
            fontfamily='DejaVu Sans')


def _arrow(ax, x1, y1, x2, y2, color, lw=1.4,
           label='', label_side='right', dashed=False):
    ls = (0, (5, 3)) if dashed else 'solid'
    ax.annotate(
        '', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle='->', color=color, lw=lw,
            linestyle=ls,
            connectionstyle='arc3,rad=0',
        ),
        zorder=2,
    )
    if label:
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        dx = 0.15 if label_side == 'right' else -0.15
        ha = 'left' if label_side == 'right' else 'right'
        ax.text(mx + dx, my, label,
                fontsize=6.5, color=color, ha=ha, va='center',
                style='italic', zorder=5)


def _skip_arrow(ax, x1, y, x2, label=''):
    """Seta horizontal tracejada para conexão skip."""
    _arrow(ax, x1, y, x2, y, AC_SKIP, lw=1.3, dashed=True,
           label=label, label_side='right')


def _curved_arrow(ax, x1, y1, x2, y2, color, lw=1.5, label='', rad=0.3):
    ax.annotate(
        '', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle='->', color=color, lw=lw,
            connectionstyle=f'arc3,rad={rad}',
        ),
        zorder=2,
    )
    if label:
        mx = (x1 + x2) / 2 - 0.4
        my = (y1 + y2) / 2
        ax.text(mx, my, label, fontsize=6.5, color=color,
                ha='right', va='center', style='italic', zorder=5)


# ---------------------------------------------------------------
# Cálculo das posições Y (computado programaticamente)
# ---------------------------------------------------------------
def _compute_positions():
    """
    Calcula todos os centros Y dos blocos.
    Encoder desce (y decresce). Decoder sobe (y cresce).
    LGAP_E é calibrado para que os skips (enc c2) fiquem
    exatamente na mesma altura que os concats do decoder.
    """
    pos = {}
    BH_, VG, LGE, LGD = BH, VGAP, LGAP_E, LGAP_D

    # --- ENCODER (de cima para baixo) ---
    TOP = 21.0
    y = TOP

    pos['input'] = y - BH_ / 2
    y -= BH_

    # quatro níveis: filters 32, 64, 128, 256
    for i in range(1, 5):
        y -= LGD             # gap após entrada / nível anterior
        pos[f'e{i}_c1'] = y - BH_ / 2
        y -= BH_ + VG
        pos[f'e{i}_c2'] = y - BH_ / 2   # ← ponto de saída do skip
        y -= BH_ + VG
        pos[f'e{i}_mp'] = y - BH_ / 2
        y -= BH_
        if i < 4:
            y -= LGE         # gap extra entre níveis do encoder

    # bottleneck (sem LGAP_E extra antes, só LGAP_D)
    y -= LGD
    pos['bn_c1'] = y - BH_ / 2
    y -= BH_ + VG
    pos['bn_c2'] = y - BH_ / 2
    pos['_bn_bottom'] = y - BH_

    # --- DECODER (de baixo para cima) ---
    # Ancoramos D4 concat ao skip de E4 para conexões horizontais exatas
    skip_e4 = pos['e4_c2']
    # D4 concat é o 2º bloco do grupo (de baixo): bottom + BH/2 + BH + VG
    # → bottom = skip_e4 - 1.5*BH - VG
    y_dec = skip_e4 - 1.5 * BH_ - VG  # bottom edge do grupo D4

    for i in range(4, 0, -1):
        pos[f'd{i}_up']  = y_dec + BH_ / 2
        y_dec           += BH_ + VG
        pos[f'd{i}_cat'] = y_dec + BH_ / 2
        y_dec           += BH_ + VG
        pos[f'd{i}_c1']  = y_dec + BH_ / 2
        y_dec           += BH_ + VG
        pos[f'd{i}_c2']  = y_dec + BH_ / 2
        y_dec           += BH_
        if i > 1:
            y_dec        += LGD

    y_dec += LGD
    pos['output'] = y_dec + BH_ / 2

    return pos


# ---------------------------------------------------------------
# Geração do diagrama
# ---------------------------------------------------------------
def generate(output_path='unet_architecture.png'):
    pos = _compute_positions()

    y_min = pos['_bn_bottom'] - 0.5
    y_max = pos['output'] + 0.8
    x_min, x_max = 0.5, 26.5
    fig_w = x_max - x_min
    fig_h = y_max - y_min

    fig, ax = plt.subplots(figsize=(fig_w * 1.05, fig_h * 1.05))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.axis('off')
    fig.patch.set_facecolor('#FAFAFA')

    # --- Título ---
    ax.text((x_min + x_max) / 2, y_max - 0.4,
            'Arquitetura U-Net — Operações Detalhadas',
            ha='center', va='center', fontsize=13,
            fontweight='bold', color='#1C2833')

    # ------------------------------------------------------------------
    # ENCODER  (esquerda, setas para BAIXO)
    # ------------------------------------------------------------------
    filter_map = {1: 32, 2: 64, 3: 128, 4: 256}

    # Entrada
    _block(ax, EX, pos['input'],
           f'Entrada  —  64×64×3\n(obstáculos | início | fim)',
           C_IO, w=BW_IO, fontsize=7.5)

    for i in range(1, 5):
        f = filter_map[i]
        _block(ax, EX, pos[f'e{i}_c1'], f'Conv 3×3  ReLU  ({f}f)',  C_CONV)
        _block(ax, EX, pos[f'e{i}_c2'], f'Conv 3×3  ReLU  ({f}f)',  C_CONV)
        _block(ax, EX, pos[f'e{i}_mp'],  'MaxPool 2×2',              C_POOL)

    # Bottleneck
    _block(ax, BNX, pos['bn_c1'], 'Conv 3×3  ReLU  (512f)', C_BN, w=BW_BN)
    _block(ax, BNX, pos['bn_c2'], 'Conv 3×3  ReLU  (512f)', C_BN, w=BW_BN)

    # ------------------------------------------------------------------
    # DECODER  (direita, setas para CIMA)
    # ------------------------------------------------------------------
    for i in range(4, 0, -1):
        f = filter_map[i]
        _block(ax, DX, pos[f'd{i}_up'],  f'UpConv 2×2  ReLU  ({f}f)',   C_UP,  w=BW_DEC)
        _block(ax, DX, pos[f'd{i}_cat'],  'Concatenar  (skip)',           C_CAT, w=BW_DEC)
        _block(ax, DX, pos[f'd{i}_c1'],  f'Conv 3×3  ReLU  ({f}f)',      C_CONV, w=BW_DEC)
        _block(ax, DX, pos[f'd{i}_c2'],  f'Conv 3×3  ReLU  ({f}f)',      C_CONV, w=BW_DEC)

    # Saída
    _block(ax, DX, pos['output'],
           'Conv 1×1  Sigmoid  (1f)\n64×64×1  —  máscara do caminho',
           C_OUT, w=BW_IO, fontsize=7.5)

    # ------------------------------------------------------------------
    # SETAS — encoder (para baixo)
    # ------------------------------------------------------------------
    _arrow(ax, EX, pos['input'] - BH/2,
               EX, pos['e1_c1'] + BH/2,  AC_ENC)

    for i in range(1, 5):
        _arrow(ax, EX, pos[f'e{i}_c1'] - BH/2,
                   EX, pos[f'e{i}_c2'] + BH/2, AC_ENC)
        _arrow(ax, EX, pos[f'e{i}_c2'] - BH/2,
                   EX, pos[f'e{i}_mp'] + BH/2, AC_ENC)
        if i < 4:
            _arrow(ax, EX, pos[f'e{i}_mp'] - BH/2,
                       EX, pos[f'e{i+1}_c1'] + BH/2, AC_ENC)

    # E4 MaxPool → BN (curva para o centro)
    _curved_arrow(ax,
                  EX,  pos['e4_mp'] - BH/2,
                  BNX, pos['bn_c1'] + BH/2,
                  AC_BN, label='MaxPool 2×2', rad=-0.25)

    # BN interno
    _arrow(ax, BNX, pos['bn_c1'] - BH/2,
               BNX, pos['bn_c2'] + BH/2, AC_BN)

    # ------------------------------------------------------------------
    # SETAS — decoder (para cima)
    # ------------------------------------------------------------------
    # BN → D4 UpConv (curva para a direita)
    _curved_arrow(ax,
                  BNX, pos['bn_c2'] + BH/2,
                  DX,  pos['d4_up'] - BH/2,
                  AC_DEC, rad=-0.25)

    for i in range(4, 0, -1):
        _arrow(ax, DX, pos[f'd{i}_up']  + BH/2,
                   DX, pos[f'd{i}_cat'] - BH/2, AC_DEC)
        _arrow(ax, DX, pos[f'd{i}_cat'] + BH/2,
                   DX, pos[f'd{i}_c1']  - BH/2, AC_DEC)
        _arrow(ax, DX, pos[f'd{i}_c1']  + BH/2,
                   DX, pos[f'd{i}_c2']  - BH/2, AC_DEC)
        if i > 1:
            _arrow(ax, DX, pos[f'd{i}_c2']    + BH/2,
                       DX, pos[f'd{i-1}_up']  - BH/2, AC_DEC)

    _arrow(ax, DX, pos['d1_c2'] + BH/2,
               DX, pos['output'] - BH/2, AC_DEC)

    # ------------------------------------------------------------------
    # SETAS SKIP (horizontais exatas)
    # ------------------------------------------------------------------
    skip_labels = {1: 'skip', 2: '', 3: '', 4: ''}
    for i in range(1, 5):
        x1 = EX + BW_ENC / 2
        x2 = DX - BW_DEC / 2
        y  = pos[f'e{i}_c2']   # mesmo que pos[f'd{i}_cat']
        _skip_arrow(ax, x1, y, x2, label=skip_labels[i])

    # ------------------------------------------------------------------
    # LABELS DE GRUPO (lado esquerdo do encoder / direito do decoder)
    # ------------------------------------------------------------------
    group_label_x_enc = EX - BW_ENC / 2 - 0.25
    group_label_x_dec = DX + BW_DEC / 2 + 0.25
    for i in range(1, 5):
        f = filter_map[i]
        y_mid = (pos[f'e{i}_c1'] + pos[f'e{i}_mp']) / 2
        ax.text(group_label_x_enc, y_mid,
                f'E{i}\n({f}f)', ha='right', va='center',
                fontsize=6.5, color='#5D6D7E', style='italic')

        y_mid = (pos[f'd{i}_up'] + pos[f'd{i}_c2']) / 2
        ax.text(group_label_x_dec, y_mid,
                f'D{i}\n({f}f)', ha='left', va='center',
                fontsize=6.5, color='#5D6D7E', style='italic')

    # ------------------------------------------------------------------
    # LEGENDA
    # ------------------------------------------------------------------
    legend_items = [
        mpatches.Patch(facecolor=C_CONV[0], edgecolor='#95A5A6',
                       label='Conv 3×3 + ReLU'),
        mpatches.Patch(facecolor=C_POOL[0], edgecolor='#95A5A6',
                       label='MaxPool 2×2  (÷2 resolução)'),
        mpatches.Patch(facecolor=C_BN[0],   edgecolor='#95A5A6',
                       label='Bottleneck Conv'),
        mpatches.Patch(facecolor=C_UP[0],   edgecolor='#95A5A6',
                       label='UpConv 2×2  (×2 resolução)'),
        mpatches.Patch(facecolor=C_CAT[0],  edgecolor='#95A5A6',
                       label='Concatenar com skip'),
        mpatches.Patch(facecolor=AC_SKIP,
                       label='Conexão skip (sem perda espacial)'),
    ]
    ax.legend(handles=legend_items,
              loc='lower center',
              bbox_to_anchor=(0.5, -0.04),
              ncol=3, fontsize=8,
              framealpha=0.95, edgecolor='#BDC3C7')

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches='tight',
                facecolor='#FAFAFA')
    plt.close()
    print(f'Diagrama salvo em: {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='unet_architecture.png')
    args = parser.parse_args()
    generate(args.output)
