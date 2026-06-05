"""
Validação automática de modelos ONNX no device via STM32 Cube AI Studio (stedgeai CLI).

Uso:
    python validate_device.py                     # valida todos os modelos quantizados
    python validate_device.py --host-only         # valida só no host (sem device físico)
    python validate_device.py --board stm32h743   # valida apenas num board específico
    python validate_device.py --model results/W064xH064_D01.../student_distilled_quantized_static.onnx
"""

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime
from glob import glob
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------
# CONFIGURAÇÃO — ajuste conforme sua instalação
# ---------------------------------------------------------------

# Caminho para o executável stedgeai do STM32 Cube AI Studio
# Exemplos comuns:
#   Windows: r"C:\ST\STM32CubeIDE_1.x.x\plugins\com.st.stm32cube.ide.mcu.externaltools.stedgeai.win32_x.x.x\tools\bin\stedgeai.exe"
#   Ou diretamente: r"C:\ST\STM32CubeAI\stedgeai.exe"
STEDGEAI_PATH = r"C:\ST\STEdgeAI\4.0\Utilities\windows\stedgeai.exe"  # <<< EDITE AQUI

# Diretório onde o pipeline salva os resultados
RESULTS_DIR = "./results"

# Boards suportados: nome_curto -> configuração do stedgeai v4
# --target: família do MCU (stedgeai v4 não aceita "stm32" genérico)
# --device: MCU específico dentro da família
# --board-name: placa de desenvolvimento usada no --mode target
BOARDS = {
    "stm32h743": {
        "target":      "stm32h7",
        "board_name":  "NUCLEO-H743ZI2",
        "extra_flags": [],
        # Workspace criado pelo STM32 Cube AI Studio para este board  <<< CONFIGURE
        "workspace":   r"C:\Users\rbieg\.stm32cubeaistudio\workspace\TesteH7",
        "description": "STM32H743ZI Nucleo-144 (Cortex-M7 @ 480MHz)",
    },
    "stm32n657": {
        "target":      "stm32n6",
        "board_name":  "STM32N6570-DK",
        "extra_flags": ["--st-neural-art", "--use-onnx-simplifier"],
        # Workspace criado pelo STM32 Cube AI Studio para este board  <<< CONFIGURE
        "workspace":   r"C:\Users\rbieg\.stm32cubeaistudio\workspace\TesteN6\.ai\run\run-17\.ai\st_ai_ws",
        "description": "STM32N6570 Discovery Kit (Cortex-M55 + NPU Ethos-U55)",
    },
}

# Arquivo de saída do relatório consolidado
REPORT_FILE = "./validation_report.json"

# Número máximo de amostras usadas na validação (host e device).
# Limitar reduz o custo temporal sem prejudicar a avaliação qualitativa.
MAX_VAL_SAMPLES = 100

# ---------------------------------------------------------------


def _find_stedgeai() -> str:
    """Tenta localizar o stedgeai automaticamente se o path padrão não existir."""
    if os.path.isfile(STEDGEAI_PATH):
        return STEDGEAI_PATH

    # Busca em paths comuns do Windows (funciona em WSL via /mnt/c)
    drive = "/mnt/c" if platform.system() == "Linux" else "C:"
    candidates = glob(f"{drive}/ST/**/stedgeai.exe", recursive=True)
    candidates += glob(f"{drive}/Program Files/STMicroelectronics/**/stedgeai.exe", recursive=True)

    if candidates:
        found = candidates[0]
        print(f"[INFO] stedgeai encontrado em: {found}")
        return found

    print("[ERRO] stedgeai não encontrado. Configure STEDGEAI_PATH no topo do script.")
    sys.exit(1)


def _to_windows_path(path: str) -> str:
    """
    Converte qualquer path para formato Windows absoluto.
    Necessário porque o stedgeai.exe é um executável Windows e não
    resolve paths WSL relativos como './results/...'.
    """
    abs_path = os.path.abspath(path)

    if platform.system() == "Linux":
        # Usa wslpath para conversão confiável (disponível em qualquer WSL)
        try:
            result = subprocess.run(
                ["wslpath", "-w", abs_path],
                capture_output=True, text=True, check=True,
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback manual: /mnt/c/foo -> C:\foo
            if abs_path.startswith("/mnt/"):
                drive = abs_path[5].upper()
                return drive + ":\\" + abs_path[7:].replace("/", "\\")

    return abs_path


def find_models(results_dir: str, quantized_only: bool = True) -> list[str]:
    """Descobre todos os modelos ONNX nos resultados do pipeline."""
    pattern = "**/*_quantized_static.onnx" if quantized_only else "**/*.onnx"
    models = sorted(glob(os.path.join(results_dir, pattern), recursive=True))
    return models


def run_stedgeai(stedgeai: str, args: list[str], timeout: int = 300) -> dict:
    """Executa stedgeai e retorna stdout, stderr e código de saída."""
    stedgeai_win = _to_windows_path(stedgeai) if os.path.isfile(stedgeai) else stedgeai

    # No WSL, executáveis .exe do Windows precisam ser chamados via cmd.exe
    # O path absoluto do cmd.exe é necessário pois o PATH do WSL pode não incluí-lo
    if platform.system() == "Linux":
        cmd = ["/mnt/c/Windows/System32/cmd.exe", "/c", stedgeai_win] + args
    else:
        cmd = [stedgeai_win] + args

    print(f"\n  $ stedgeai {' '.join(args)}")

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        if proc.returncode != 0:
            print(f"  [WARN] returncode={proc.returncode}")
            if proc.stderr:
                print(f"  stderr: {proc.stderr[:500]}")
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except subprocess.TimeoutExpired:
        print("  [ERRO] Timeout expirado")
        return {"returncode": -1, "stdout": "", "stderr": "Timeout expirado"}
    except FileNotFoundError as e:
        print(f"  [ERRO] Executável não encontrado: {e}")
        return {"returncode": -1, "stdout": "", "stderr": str(e)}


def analyze_model(stedgeai: str, model_path: str, target: str, extra_flags: list, output_dir: str) -> dict:
    """Analisa o modelo: conta MACCs, memória RAM/Flash e complexidade."""
    print("\n  [1/3] Analisando modelo...")
    return run_stedgeai(stedgeai, [
        "analyze",
        "--target",    target,
        *extra_flags,
        "--name",      "network",
        "--model",     _to_windows_path(model_path),
        "--output",    _to_windows_path(output_dir),
        "--verbosity", "1",
    ])


def validate_host(
    stedgeai: str, model_path: str, target: str, extra_flags: list,
    val_input: str, val_output: str, output_dir: str,
) -> dict:
    """Valida o modelo simulado no host (PC) para o target especificado."""
    print("\n  [2/3] Validação no host...")
    return run_stedgeai(stedgeai, [
        "validate",
        "--target",    target,
        *extra_flags,
        "--name",      "network",
        "--model",     _to_windows_path(model_path),
        "--valinput",  _to_windows_path(val_input),
        "--valoutput", _to_windows_path(val_output),
        "--mode",      "host",
        "--c-api",     "st-ai",
        "--output",    _to_windows_path(output_dir),
        "--verbosity", "1",
    ])


def validate_target(
    stedgeai: str, model_path: str, target: str, extra_flags: list,
    val_input: str, val_output: str, workspace: str, board_name: str, output_dir: str,
) -> dict:
    """Valida o modelo no device físico via ST-Link."""
    print(f"\n  [3/3] Validação no device ({board_name})...")
    return run_stedgeai(stedgeai, [
        "validate",
        "--target",    target,
        *extra_flags,
        "--name",      "network",
        "--model",     _to_windows_path(model_path),
        "--valinput",  _to_windows_path(val_input),
        "--valoutput", _to_windows_path(val_output),
        "--mode",      "target",
        "--c-api",     "st-ai",
        "--workspace", workspace,
        "--output",    _to_windows_path(output_dir),
        "--verbosity", "1",
    ])


def _parse_metrics(stdout: str) -> dict:
    """Extrai métricas numéricas do output do stedgeai."""
    metrics = {}

    # Padrões comuns no output do stedgeai
    patterns = {
        "macc":        r"(\d[\d,\.]+)\s+MACC",
        "ram_bytes":   r"RAM\s*[:\-]\s*([\d,\.]+)\s*bytes",
        "flash_bytes": r"Flash\s*[:\-]\s*([\d,\.]+)\s*bytes",
        "accuracy":    r"[Aa]ccuracy\s*[:\-]\s*([\d\.]+)%?",
        "iou":         r"[Ii]o[Uu]\s*[:\-]\s*([\d\.]+)",
        "latency_ms":  r"[Ll]atency\s*[:\-]\s*([\d\.]+)\s*ms",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, stdout)
        if match:
            metrics[key] = match.group(1).replace(",", "")

    return metrics


def _prepare_val_data(val_input: str, val_output: str, output_dir: str) -> tuple[str, str]:
    """Limita os dados de validação a MAX_VAL_SAMPLES amostras.

    Se o conjunto possuir mais amostras do que o limite, emite um aviso e salva
    versões podadas no output_dir para não modificar os arquivos originais.
    """
    X = np.load(val_input)
    Y = np.load(val_output)

    n = X.shape[0]
    if n > MAX_VAL_SAMPLES:
        print(
            f"\n  [AVISO] O conjunto de validação possui {n} amostras. "
            f"As entradas e saídas serão podadas para {MAX_VAL_SAMPLES} amostras "
            f"a fim de reduzir o custo temporal da validação."
        )
        X = X[:MAX_VAL_SAMPLES]
        Y = Y[:MAX_VAL_SAMPLES]
        pruned_input  = str(Path(output_dir) / "X_val_pruned.npy")
        pruned_output = str(Path(output_dir) / "Y_val_pruned.npy")
        np.save(pruned_input, X)
        np.save(pruned_output, Y)
        return pruned_input, pruned_output

    return val_input, val_output


def run_validation(
    boards: list[str],
    models: list[str],
    host_only: bool = False,
) -> list[dict]:
    """Executa o pipeline completo de validação para cada modelo e board."""
    stedgeai = _find_stedgeai()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []

    for model_path in models:
        model_label = Path(model_path).parent.name
        print(f"\n{'='*60}")
        print(f"  MODELO: {model_label}")
        print(f"  Arquivo: {model_path}")
        print(f"{'='*60}")

        for board_key in boards:
            board_info = BOARDS[board_key]
            target      = board_info["target"]
            board_name  = board_info["board_name"]
            extra_flags = board_info["extra_flags"]
            workspace   = board_info["workspace"]
            results_dir = Path(model_path).parent
            val_input   = str(results_dir / "X_val.npy")
            val_output  = str(results_dir / "Y_val.npy")
            output_dir  = str(results_dir / f"stedgeai_{board_key}_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)

            val_input, val_output = _prepare_val_data(val_input, val_output, output_dir)

            print(f"\n  Board: {board_info['description']} (target={target})")
            record = {
                "timestamp":   timestamp,
                "model":       model_label,
                "model_path":  model_path,
                "board":       board_key,
                "target":      target,
                "board_name":  board_name,
                "extra_flags": extra_flags,
                "output_dir":  output_dir,
                "steps": {},
            }

            def _step(r):
                return {
                    "ok":      r["returncode"] == 0,
                    "metrics": _parse_metrics(r["stdout"]),
                    "stdout":  r["stdout"][-2000:],
                    "stderr":  r["stderr"][-500:],
                }

            # Análise estática (sempre)
            record["steps"]["analyze"] = _step(
                analyze_model(stedgeai, model_path, target, extra_flags, output_dir)
            )

            # Validação no host (sempre)
            record["steps"]["validate_host"] = _step(
                validate_host(stedgeai, model_path, target, extra_flags,
                              val_input, val_output, output_dir)
            )

            # Validação no device (opcional)
            if not host_only:
                record["steps"]["validate_target"] = _step(
                    validate_target(stedgeai, model_path, target, extra_flags,
                                    val_input, val_output, workspace, board_name, output_dir)
                )

            all_results.append(record)

    return all_results


def print_summary(results: list[dict]) -> None:
    print(f"\n{'='*60}")
    print("  RESUMO DA VALIDAÇÃO")
    print(f"{'='*60}")

    for r in results:
        host_ok = r["steps"].get("validate_host", {}).get("ok", False)
        target_ok = r["steps"].get("validate_target", {}).get("ok", None)
        analyze_metrics = r["steps"].get("analyze", {}).get("metrics", {})

        host_status   = "OK" if host_ok else "FALHOU"
        target_status = "OK" if target_ok else ("FALHOU" if target_ok is False else "N/A")

        print(f"\n  [{r['board'].upper()} / {r['target']}] {r['model']}")
        print(f"    Host:   {host_status}  |  Device: {target_status}")
        if analyze_metrics:
            macc  = analyze_metrics.get("macc", "?")
            ram   = analyze_metrics.get("ram_bytes", "?")
            flash = analyze_metrics.get("flash_bytes", "?")
            print(f"    MACC: {macc}  |  RAM: {ram} bytes  |  Flash: {flash} bytes")

        # Métricas de qualidade do modelo no host
        host_metrics = r["steps"].get("validate_host", {}).get("metrics", {})
        if host_metrics.get("accuracy"):
            print(f"    Acurácia (host): {host_metrics['accuracy']}%")

    print()


def save_report(results: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Relatório salvo em: {path}")


def main():
    parser = argparse.ArgumentParser(description="Validação de modelos ONNX no STM32")
    parser.add_argument(
        "--board",
        choices=list(BOARDS.keys()),
        nargs="+",
        default=list(BOARDS.keys()),
        help="Board(s) alvo (padrão: todos)",
    )
    parser.add_argument(
        "--model",
        nargs="+",
        help="Caminho(s) específico(s) para modelo(s) ONNX",
    )
    parser.add_argument(
        "--all-onnx",
        action="store_true",
        help="Valida todos os ONNX (padrão: apenas quantizados)",
    )
    parser.add_argument(
        "--host-only",
        action="store_true",
        help="Executa apenas validação no host, sem device físico",
    )
    parser.add_argument(
        "--report",
        default=REPORT_FILE,
        help=f"Caminho do relatório JSON (padrão: {REPORT_FILE})",
    )
    args = parser.parse_args()

    models = args.model if args.model else find_models(RESULTS_DIR, quantized_only=not args.all_onnx)

    if not models:
        print(f"[ERRO] Nenhum modelo encontrado em '{RESULTS_DIR}'.")
        sys.exit(1)

    print(f"\nModelos encontrados: {len(models)}")
    for m in models:
        print(f"  {m}")

    results = run_validation(
        boards=args.board,
        models=models,
        host_only=args.host_only,
    )

    print_summary(results)
    save_report(results, args.report)


if __name__ == "__main__":
    main()
