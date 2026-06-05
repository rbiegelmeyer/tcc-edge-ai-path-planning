"""
Visualiza os resultados da busca de hiperparâmetros (sfilter × depth) para um dataset.

Modos:
  --dataset NOME   analisa um dataset específico (heatmap + linhas)
  --compare        compara a evolução do melhor resultado entre todos os datasets

Uso:
    python analyze_training_results.py --dataset W064xH064_D03_S000000_E050000
    python analyze_training_results.py --dataset W064xH064_D03_S000000_E025000 --output resultado.png
    python analyze_training_results.py --compare
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

CSV_PATH = os.path.join(os.path.dirname(__file__), 'training_log.csv')

# ---------------------------------------------------------------
# Paleta
# ---------------------------------------------------------------
LINE_COLORS = ['#1f77b4','#ff7f0e','#17becf','#e7ba52',
               '#9467bd','#8c564b','#e377c2','#7f7f7f']


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def _load(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path)
    df['bottleneck'] = df['sfilter'] * (2 ** df['depth'])
    return df


def _best_per_config(df):
    """Mantém somente a melhor execução por (sfilter, depth)."""
    return (df.sort_values('best_val_loss')
              .drop_duplicates(subset=['sfilter', 'depth'], keep='first'))




def _line_plot(ax, df, metric_col, ylabel, title, highlight_min=True, legend_loc='upper right'):
    """Linha por sfilter, eixo X = depth."""
    sfilters = sorted(df['sfilter'].unique())
    best_val = df[metric_col].min() if highlight_min else df[metric_col].max()
    best_row = df.loc[df[metric_col] == best_val].iloc[0]

    for idx, sf in enumerate(sfilters):
        sub = df[df['sfilter'] == sf].sort_values('depth')
        color = LINE_COLORS[idx % len(LINE_COLORS)]
        ax.plot(sub['depth'], sub[metric_col],
                marker='o', markersize=5, linewidth=1.8,
                color=color, label=f'sfilter={sf}')

    # Marca o melhor ponto globalmente
    mc          = 'red' if highlight_min else 'green'
    best_val_br = f'{best_val:.4f}'.replace('.', ',')
    ax.scatter(best_row['depth'], best_val,
               s=200, marker='o', facecolors='none',
               edgecolors=mc, linewidths=1.2, zorder=5,
               label=f'Melhor: {best_val_br}\n(f={int(best_row["sfilter"])}, d={int(best_row["depth"])})')

    ax.set_xlabel('Profundidade (depth)', fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', labelsize=16)
    ax.legend(fontsize=16, loc=legend_loc, ncol=2)
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f'{v:.2f}'.replace('.', ','))
    )
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xticks(sorted(df['depth'].unique()))


# ---------------------------------------------------------------
# Modo 1 — análise de dataset específico
# ---------------------------------------------------------------
def plot_dataset(dataset_name, output_path=None):
    df_all = _load()

    available = df_all['dataset'].unique().tolist()
    if dataset_name not in available:
        print(f'[ERRO] Dataset "{dataset_name}" não encontrado.')
        print('Disponíveis:')
        for d in sorted(available):
            print(f'  {d}')
        return

    df = _best_per_config(df_all[df_all['dataset'] == dataset_name].copy())
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    _line_plot(axes[0], df, 'best_val_iou_metric',
               'IoU (%)', 'IoU (%) por Profundidade', highlight_min=False, legend_loc='lower right')
    _line_plot(axes[1], df, 'best_val_loss',
               'Val. Loss (%)', 'Val. Loss (%) por Profundidade', highlight_min=True)

    plt.tight_layout()

    if output_path is None:
        out_dir = os.path.join(os.path.dirname(__file__), 'results', dataset_name)
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, 'hyperparameter_analysis.png')

    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f'Gráfico salvo em: {output_path}')


# ---------------------------------------------------------------
# Modo 2 — comparação entre datasets (evolução com mais dados)
# ---------------------------------------------------------------
def plot_compare(output_path=None):
    df_all = _load()
    best   = _best_per_config(df_all)

    # Ordena datasets pelo número de amostras
    datasets = (best[['dataset', 'total']]
                .drop_duplicates()
                .sort_values('total')['dataset']
                .tolist())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for metric, ax, ylabel, title, cmap_line, best_fn in [
        ('best_val_loss',       axes[0], 'Val. Loss (%)',
         'Val. Loss (%) × Tamanho do Dataset',  'Reds',   min),
        ('best_val_iou_metric', axes[1], 'IoU Validação (%)',
         'IoU Validação (%) × Tamanho do Dataset', 'Greens', max),
    ]:
        # Uma linha por combinação (sfilter, depth)
        configs = best[['sfilter', 'depth']].drop_duplicates()
        configs = configs.sort_values(['sfilter', 'depth'])

        plotted = 0
        for _, row in configs.iterrows():
            sf, dp = int(row['sfilter']), int(row['depth'])
            pts = []
            for ds in datasets:
                sub = best[(best['dataset'] == ds) &
                           (best['sfilter'] == sf) &
                           (best['depth']   == dp)]
                if len(sub):
                    pts.append((best[best['dataset'] == ds]['total'].iloc[0],
                                sub[metric].iloc[0]))
            if len(pts) >= 2:
                xs, ys = zip(*sorted(pts))
                ax.plot(xs, ys, marker='o', markersize=4, linewidth=1.2,
                        alpha=0.7, label=f'f={sf} d={dp}')
                plotted += 1

        # Destaca o melhor ponto global
        best_val = best_fn(best[metric])
        best_row = best.loc[best[metric] == best_val].iloc[0]
        ds_size  = best_row['total']
        best_val_br = f'{best_val:.4f}'.replace('.', ',')
        ax.scatter(ds_size, best_val, s=200, marker='o',
                   facecolors='none', edgecolors='black', linewidths=1.2, zorder=6,
                   label=f'Global: {best_val_br}\n(f={int(best_row["sfilter"])}, d={int(best_row["depth"])})')

        ax.set_xlabel('Amostras de Treino', fontsize=20)
        ax.set_ylabel(ylabel, fontsize=20)
        ax.set_title(title, fontsize=18, fontweight='bold')
        ax.tick_params(axis='both', labelsize=16)
        ax.legend(fontsize=16, loc='upper right', ncol=3)
        ax.grid(True, linestyle='--', alpha=0.5)
        xticks = sorted(best['total'].unique())
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda v, _: f'{v:.2f}'.replace('.', ','))
        )
        ax.set_xticks(xticks)
        ax.set_xticklabels([f'{x:,}'.replace(',', '.') for x in xticks], rotation=15, fontsize=16)

    plt.tight_layout()

    if output_path is None:
        out_dir = os.path.join(os.path.dirname(__file__), 'results')
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, 'dataset_comparison.png')

    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f'Gráfico salvo em: {output_path}')


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------
def main():
    global CSV_PATH

    parser = argparse.ArgumentParser(
        description='Análise visual dos resultados de treinamento'
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dataset', metavar='NOME',
                       help='Nome do dataset a analisar')
    group.add_argument('--compare', action='store_true',
                       help='Compara evolução entre todos os datasets')
    parser.add_argument('--output', default=None,
                        help='Caminho de saída da imagem (opcional)')
    parser.add_argument('--csv', default=CSV_PATH,
                        help=f'Caminho do CSV (padrão: {CSV_PATH})')
    args = parser.parse_args()

    CSV_PATH = args.csv

    if args.compare:
        plot_compare(args.output)
    else:
        plot_dataset(args.dataset, args.output)


if __name__ == '__main__':
    main()
