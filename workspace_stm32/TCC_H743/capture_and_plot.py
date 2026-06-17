#!/usr/bin/env python3
"""
capture_and_plot.py
-------------------
Captura tensores do STM32 via GDB/OpenOCD e gera gráfico comparativo.

Ajuste as constantes abaixo e execute:
    python capture_and_plot.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from stm32_debug_link import capture_tensors
from plot_capture import plot_capture


# ---------------------------------------------------------------------------
# Configuração — editar conforme necessário
# ---------------------------------------------------------------------------
ELF_PATH  = r"STM32CubeIDE\Debug\TCC_H743.elf"
NPZ_PATH  = r"STM32CubeIDE\W064xH064_D03_S100000_E100100.npz"
JSON_OUT  = r"capture.json"
PNG_OUT   = r"capture.png"

GDB_PORT  = 3333
TIMEOUT_S = 30.0
LAYOUT    = "horizontal"   # "horizontal" ou "vertical"
SHOW_TICKS = True
# ---------------------------------------------------------------------------


def main():
    elf  = str(Path(__file__).parent / ELF_PATH)
    npz  = str(Path(__file__).parent / NPZ_PATH) if NPZ_PATH else None
    json_path = str(Path(__file__).parent / JSON_OUT)
    png_path  = str(Path(__file__).parent / PNG_OUT)

    print(f"[1/2] Capturando tensores (timeout={TIMEOUT_S}s) ...")
    capture_tensors(
        elf_path   = elf,
        gdb_port   = GDB_PORT,
        timeout_s  = TIMEOUT_S,
        output_json= json_path,
    )

    print(f"[2/2] Gerando gráfico ...")
    out = plot_capture(
        capture_json = json_path,
        npz_path     = npz,
        output_path  = png_path,
        layout       = LAYOUT,
        show_ticks   = SHOW_TICKS,
    )

    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
