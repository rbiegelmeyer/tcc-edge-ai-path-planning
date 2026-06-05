"""
Seleciona o melhor modelo (menor val_loss) de cada dataset e gera dois
gráficos comparativos mostrando o benefício de aumentar o volume de dados:
  - best_model_comparison_iou.png
  - best_model_comparison_loss.png

Uso:
    python best_model_comparison.py
    python best_model_comparison.py --csv training_log.csv
    python best_model_comparison.py --output-iou iou.png --output-loss loss.png
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

CSV_PATH = os.path.join(os.path.dirname(__file__), 'training_log.csv')


def _load_best_per_dataset(csv_path):
    df = pd.read_csv(csv_path)
    best = (df.sort_values('best_val_loss')
              .drop_duplicates(subset='dataset', keep='first')
              .sort_values('total')
              .reset_index(drop=True))
    best['bottleneck'] = best['sfilter'] * (2 ** best['depth'])
    return best


def _improvement_pct(a, b, lower_is_better=True):
    """Retorna a string de melhoria percentual entre dois pontos consecutivos."""
    if lower_is_better:
        pct = (a - b) / a * 100
        sign = '▼'
        color = '#2ecc71'
    else:
        pct = (b - a) / a * 100
        sign = '▲'
        color = '#2ecc71'
    return f'{sign} {pct:.1f}'.replace('.', ',') + '%', color


def _plot_single(best, metric, ylabel, color, dark_color, lower_is_better, output_path):
    """Gera e salva o gráfico para uma única métrica."""
    rows   = list(best.itertuples())
    n      = len(rows)
    xs     = [r.total for r in rows]
    ys     = [getattr(r, metric) for r in rows]
    y_range = max(ys) - min(ys) if max(ys) != min(ys) else 1.0

    _, ax = plt.subplots(1, 1, figsize=(10, 7))

    # Limites do eixo X com padding lateral
    x_pad = (max(xs) - min(xs)) * 0.08
    ax.set_xlim(min(xs) - x_pad, max(xs) + x_pad)

    # Limites do eixo Y: dados ocupam os 55% inferiores, topo reservado para caixas
    y_lo = min(ys) - y_range * 0.30
    ax.set_ylim(y_lo, y_lo + y_range / 0.52)

    # Posições das caixas de anotação em axes fraction
    BOX_Y_FRAC  = 0.95
    margin_frac = 0.18
    box_xf = [margin_frac + (1 - 2 * margin_frac) * i / (n - 1) for i in range(n)]

    # Altura fixa para todos os textos de porcentagem: logo abaixo do menor scatter
    pct_y = min(ys) - y_range * 0.12

    # Linha de tendência
    ax.plot(xs, ys, color=color, linewidth=2.5, linestyle='--', alpha=0.5, zorder=1,
            label='Melhor modelo por dataset')

    for idx, row in enumerate(rows):
        x  = row.total
        y  = getattr(row, metric)
        sf, dp, bn = int(row.sfilter), int(row.depth), int(row.bottleneck)

        ax.scatter(x, y, s=60, color=color, edgecolors=dark_color,
                   linewidths=1.2, zorder=4)

        config_txt = f'sfilter = {sf}\ndepth = {dp}\ngargalo = {bn}\n{y:.4f}'.replace('.', ',')
        ax.annotate(
            config_txt,
            xy=(x, y),
            xytext=(box_xf[idx], BOX_Y_FRAC),
            xycoords='data',
            textcoords='axes fraction',
            ha='center', va='top', fontsize=10,
            fontweight='bold', color='black',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='black', alpha=0.92, linewidth=1.2),
            arrowprops=dict(
                arrowstyle='->', color='black', lw=1.3,
                mutation_scale=18, shrinkB=6,
                connectionstyle='arc3,rad=0.0',
            ),
            zorder=5,
        )

        # Percentual de melhoria entre datasets consecutivos — todos na mesma altura
        if idx > 0:
            prev_x = rows[idx - 1].total
            label, imp_color = _improvement_pct(getattr(rows[idx - 1], metric), y, lower_is_better)
            ax.text((prev_x + x) / 2, pct_y, label,
                    ha='center', va='top', fontsize=10,
                    fontweight='bold', color=imp_color,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#f9f9f9',
                              edgecolor=imp_color, alpha=0.9, linewidth=1))

    ax.set_xlabel('Amostras de Treinamento', fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.tick_params(axis='both', labelsize=11)
    ax.set_xticks(xs)
    ax.set_xticklabels([f'{x:,}'.replace(',', '.') for x in xs], fontsize=11)
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f'{v:.2f}'.replace('.', ','))
    )
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=11, loc='best')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Gráfico salvo em: {output_path}')


def plot_best_comparison(csv_path=CSV_PATH, output_iou=None, output_loss=None):
    best = _load_best_per_dataset(csv_path)

    out_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(out_dir, exist_ok=True)

    if output_iou is None:
        output_iou = os.path.join(out_dir, 'best_model_comparison_iou.png')
    if output_loss is None:
        output_loss = os.path.join(out_dir, 'best_model_comparison_loss.png')

    _plot_single(best, 'best_val_iou_metric', 'IoU de Validação',
                 '#2980b9', '#1a5276', False, output_iou)
    _plot_single(best, 'best_val_loss', 'Loss de Validação (BCE+Dice)',
                 '#e74c3c', '#922b21', True, output_loss)

    # Tabela resumida no terminal
    print('\nMelhor modelo por dataset:')
    print(f'{"Dataset":<40} {"Amostras":>9} {"sfilter":>8} {"depth":>6} '
          f'{"Val Loss":>10} {"Val IoU":>9}')
    print('─' * 90)
    for _, row in best.iterrows():
        print(f'{row["dataset"]:<40} {int(row["total"]):>9,} {int(row["sfilter"]):>8} '
              f'{int(row["depth"]):>6} {row["best_val_loss"]:>10.4f} '
              f'{row["best_val_iou_metric"]:>9.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Comparativo do melhor modelo por dataset'
    )
    parser.add_argument('--csv', default=CSV_PATH,
                        help=f'Caminho do CSV (padrão: {CSV_PATH})')
    parser.add_argument('--output-iou', default=None,
                        help='Caminho de saída do gráfico IoU (opcional)')
    parser.add_argument('--output-loss', default=None,
                        help='Caminho de saída do gráfico Val Loss (opcional)')
    args = parser.parse_args()

    plot_best_comparison(csv_path=args.csv,
                         output_iou=args.output_iou,
                         output_loss=args.output_loss)
