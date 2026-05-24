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
    # '../AStar/W064xH064_D01_S000000_E005000.csv',
    # '../AStar/W064xH064_D02_S000000_E005000.csv',
    # '../AStar/W064xH064_D03_S000000_E005000.csv',
    # '../AStar/W064xH064_D04_S000000_E005000.csv',
    # '../AStar/W064xH064_D05_S000000_E005000.csv',
    '../AStar/W064xH064_D00_S000000_E001000.csv',
    '../AStar/W064xH064_D01_S000000_E001000.csv',
    '../AStar/W064xH064_D02_S000000_E001000.csv',
    '../AStar/W064xH064_D03_S000000_E001000.csv',
    '../AStar/W064xH064_D04_S000000_E001000.csv',
    '../AStar/W064xH064_D05_S000000_E001000.csv',
]

# ---------------------------------------------------------------

def run_pipeline(datasets):
    results = []

    for df_filename in datasets:
        label = df_filename.split('/')[-1]
        print(f'\n{"="*60}')
        print(f'  DATASET: {label}')
        print(f'{"="*60}')

        try:
            print('\n[1/3] Treinamento do professor...')
            results_path, teacher_ckpt = train(df_filename)
            # results_path = f'./results/{label.split(".")[0]}'
            # teacher_ckpt = f'{results_path}/path_finder_Unet.keras'

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
