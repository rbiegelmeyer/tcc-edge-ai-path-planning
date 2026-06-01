"""
Pipeline de treinamento automatizado:
  1. Treina modelo professor (main.run)
  2. Destila para modelo aluno (Distiller.run)
  3. Converte para ONNX quantizado (convertKeras2ONNX.run)
"""

import traceback
from Coach import train
from Distiller import distill
from ConvertKeras2ONNX import convert

# ---------------------------------------------------------------
# Datasets a processar — edite esta lista conforme necessário
# ---------------------------------------------------------------
DATASETS = [
    # '../AStar/W064xH064_D03_S000000_E001000.csv',
    # '../AStar/W064xH064_D03_S000000_E005000.csv',
    # '../AStar/W064xH064_D03_S000000_E020000.csv',
    # '../AStar/W064xH064_D03_S000000_E005000.csv',
    # '../AStar/W064xH064_D03_S000000_E010000.csv',
    '../AStar/W064xH064_D03_S000000_E025000.csv',
    # '../AStar/W064xH064_D03_S000000_E080000.csv',
    # '../AStar/W064xH064_D03_S000000_E050000.csv',
    # '../AStar/W064xH064_D03_S000000_E025000.csv',
    # '../AStar/W064xH064_D03_S000000_E100000.csv',
    # '../AStar/W064xH064_D02_S000000_E005000.csv',
    # '../AStar/W064xH064_D03_S000000_E005000.csv',
    # '../AStar/W064xH064_D04_S000000_E005000.csv',
    # '../AStar/W064xH064_D05_S000000_E005000.csv',
    # '../AStar/W032xH032_D00_S000000_E005000.csv',
    # '../AStar/W032xH032_D01_S000000_E005000.csv',
    # '../AStar/W032xH032_D02_S000000_E005000.csv',
    # '../AStar/W032xH032_D03_S000000_E005000.csv',
    # '../AStar/W032xH032_D04_S000000_E005000.csv',
    # '../AStar/W064xH064_D00_S000000_E001000.csv',
    # '../AStar/W064xH064_D01_S000000_E001000.csv',
    # '../AStar/W064xH064_D02_S000000_E001000.csv',
    # '../AStar/W064xH064_D03_S000000_E001000.csv',
    # '../AStar/W064xH064_D04_S000000_E001000.csv',
    # '../AStar/W064xH064_D03_S000000_E050000.csv',
]

# ---------------------------------------------------------------

def run_pipeline(datasets):
    results = []

    for df_filename in datasets:
        label = df_filename.split('/')[-1]
        print(f'\n{"="*60}')
        print(f'  DATASET: {label}')
        print(f'{"="*60}')

        # for sfilter in [4, 8, 16, 32, 64, 128]:
        # for sfilter in [32, 64, 128]:
        #     for depth in [2, 3, 4, 5, 6]:
        #     for depth in [4, 5, 6, 7, 8]:
                
                # if ((sfilter * (2 ** depth)) > 1024):
                #     continue

        tasks = [
            # [32, 5, '../AStar/W064xH064_D03_S000000_E005000.csv'],
            # [64, 4, '../AStar/W064xH064_D03_S000000_E005000.csv'],
            # [32, 5, '../AStar/W064xH064_D03_S000000_E010000.csv'],
            # [64, 4, '../AStar/W064xH064_D03_S000000_E010000.csv'],
            # [32, 5, '../AStar/W064xH064_D03_S000000_E025000.csv'],
            [32, 3, '../AStar/W064xH064_D03_S000000_E075000.csv'],
            [32, 4, '../AStar/W064xH064_D03_S000000_E075000.csv'],
            [32, 5, '../AStar/W064xH064_D03_S000000_E075000.csv'],
            [64, 3, '../AStar/W064xH064_D03_S000000_E075000.csv'],
            [64, 4, '../AStar/W064xH064_D03_S000000_E075000.csv'],
            # [32, 5, '../AStar/W064xH064_D03_S000000_E080000.csv'],
            [128, 3, '../AStar/W064xH064_D03_S000000_E075000.csv'],
        ]

        for sfilter, depth, dataset in tasks:

            try:
                print('\n[1/3] Treinamento do professor...')
                results_path, teacher_ckpt = train(dataset, sfilter=sfilter, depth=depth)
                # results_path = f'./results/{label.split(".")[0]}'
                # teacher_ckpt = f'{results_path}/path_finder_Unet.keras'

                continue

                print('\n[2/3] Destilação do aluno...')
                student_ckpt = distill(results_path, teacher_ckpt)

                print('\n[3/3] Conversão para ONNX...')
                onnx_path = convert(results_path)

                results.append({
                    'dataset':      label,
                    'status':       'ok',
                    'results_path': results_path,
                    'onnx':         onnx_path,
                })
                print(f'\n  Concluído: {onnx_path}')

            except Exception:
                print(f'\n  ERRO em {label}:')
                traceback.print_exc()
                results.append({'dataset': label, 'status': 'erro'})

    # --- Resumo final ---
    print(f'\n{"="*60}')
    print('  RESUMO DA PIPELINE')
    print(f'{"="*60}')
    for r in results:
        status = 'OK' if r['status'] == 'ok' else 'ERRO'
        print(f'  [{status}] {r["dataset"]}')
    print()


if __name__ == '__main__':
    run_pipeline(DATASETS)
