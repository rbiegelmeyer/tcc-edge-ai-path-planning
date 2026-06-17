#!/usr/bin/env python3
"""
capture_loop.py
---------------
Loop contínuo de captura e visualização:
  1. Aguarda o breakpoint via TCL (sem GDB)
  2. Lê os tensores diretamente via TCL read_memory
  3. Exibe o gráfico em janela interativa
  4. Retoma o firmware e volta ao passo 1

Configuração: ajuste as constantes abaixo.
Encerrar: Ctrl+C
"""

import sys
import json
from pathlib import Path

# Backend interativo ANTES de qualquer import do matplotlib
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()

sys.path.insert(0, str(Path(__file__).parent))

from stm32_debug_link import (
    OpenOCDTCL,
    resolve_symbol,
    resolve_line_address,
    _configure_gdb_events,
    CAPTURE_SOURCE_FILE, CAPTURE_LINE,
    INPUT_BYTES,  OUTPUT_BYTES,  SOLVED_BYTES,
    INPUT_SHAPE,  OUTPUT_SHAPE,  SOLVED_SHAPE,
    INPUT_SCALE,  INPUT_ZP,
    OUTPUT_SCALE, OUTPUT_ZP,
)
from plot_capture import plot_capture


# ---------------------------------------------------------------------------
# Configuração — editar conforme necessário
# ---------------------------------------------------------------------------
ELF_PATH   = r"STM32CubeIDE\Debug\TCC_H743.elf"
NPZ_PATH   = r"STM32CubeIDE\W064xH064_D03_S100000_E100100.npz"
JSON_OUT   = r"capture.json"
PNG_OUT    = r"capture.png"
TCL_PORT   = 6666
TIMEOUT_S  = 120.0
LAYOUT     = "horizontal"   # "horizontal" ou "vertical"
SHOW_TICKS = True
WINDOW_POS = "+200+500"     # posição da janela: "+x+y" em pixels a partir do canto superior esquerdo
# ---------------------------------------------------------------------------


def _dump_image(tcl: OpenOCDTCL, addr: int, n_bytes: int, tmp_file: Path) -> list:
    """Despeja n_bytes a partir de addr em binário via dump_image do OpenOCD."""
    posix_path = tmp_file.as_posix()   # TCL exige barras /
    old = tcl.sock.gettimeout()
    tcl.sock.settimeout(max(old, n_bytes / 10000.0 + 10.0))
    try:
        resp = tcl.send(f"dump_image {posix_path} 0x{addr:08x} {n_bytes}")
    finally:
        tcl.sock.settimeout(old)
    if not tmp_file.exists():
        raise RuntimeError(
            f"dump_image falhou @ 0x{addr:08x} [{n_bytes}B]: {resp!r}"
        )
    raw = tmp_file.read_bytes()
    return [b if b < 128 else b - 256 for b in raw]


def _capture(tcl: OpenOCDTCL, symbols: dict, tmp_dir: Path) -> dict:
    """Lê tensores do target halted e retorna o dict de captura."""
    in_ptr    = tcl.read_memory(symbols['stai_input'],  1, width=32)[0]
    out_ptr   = tcl.read_memory(symbols['stai_output'], 1, width=32)[0]
    x_val_idx = int(tcl.read_memory(symbols['x_val_idx'], 1, width=32)[0])

    print(f"[i] x_val_idx={x_val_idx}  "
          f"in=0x{in_ptr:08x}  out=0x{out_ptr:08x}")

    tmp_in     = tmp_dir / "_cap_input.bin"
    tmp_out    = tmp_dir / "_cap_output.bin"
    tmp_solved = tmp_dir / "_cap_solved.bin"

    in_data     = _dump_image(tcl, in_ptr,                INPUT_BYTES,  tmp_in)
    out_data    = _dump_image(tcl, out_ptr,               OUTPUT_BYTES, tmp_out)
    solved_data = _dump_image(tcl, symbols['map_solved'], SOLVED_BYTES, tmp_solved)

    for f in (tmp_in, tmp_out, tmp_solved):
        f.unlink(missing_ok=True)

    n_path = sum(1 for v in solved_data if v == 2)
    print(f"[i] map_solved: {n_path} células de caminho A*")

    return {
        "capture_point": f"{CAPTURE_SOURCE_FILE}:{CAPTURE_LINE}",
        "x_val_idx":     x_val_idx,
        "stai_input": {
            "shape":      INPUT_SHAPE,
            "format":     "S8",
            "scale":      INPUT_SCALE,
            "zero_point": INPUT_ZP,
            "n_bytes":    INPUT_BYTES,
            "data":       in_data,
        },
        "stai_output": {
            "shape":      OUTPUT_SHAPE,
            "format":     "S8",
            "scale":      OUTPUT_SCALE,
            "zero_point": OUTPUT_ZP,
            "n_bytes":    OUTPUT_BYTES,
            "data":       out_data,
        },
        "map_solved": {
            "shape":   SOLVED_SHAPE,
            "format":  "S8",
            "n_bytes": SOLVED_BYTES,
            "data":    solved_data,
        },
    }


def main():
    base      = Path(__file__).parent
    elf       = str(base / ELF_PATH)
    npz       = str(base / NPZ_PATH) if NPZ_PATH else None
    json_path = str(base / JSON_OUT)
    png_path  = str(base / PNG_OUT)

    print("[i] Resolvendo símbolos do ELF...")
    symbols = {
        'stai_input':  resolve_symbol(elf, "stai_input"),
        'stai_output': resolve_symbol(elf, "stai_output"),
        'x_val_idx':   resolve_symbol(elf, "x_val_idx"),
        'map_solved':  resolve_symbol(elf, "map_solved"),
    }
    bp_addr = resolve_line_address(elf, CAPTURE_SOURCE_FILE, CAPTURE_LINE)

    print(f"[i] &stai_input  @ 0x{symbols['stai_input']:08x}")
    print(f"[i] &stai_output @ 0x{symbols['stai_output']:08x}")
    print(f"[i] &x_val_idx   @ 0x{symbols['x_val_idx']:08x}")
    print(f"[i] &map_solved  @ 0x{symbols['map_solved']:08x}")
    print(f"[i] breakpoint   @ 0x{bp_addr:08x}  "
          f"({CAPTURE_SOURCE_FILE}:{CAPTURE_LINE})")

    _configure_gdb_events()

    iteration = 0
    fig = None
    print("[i] Iniciando loop de captura — Ctrl+C para parar\n")

    tmp_dir = Path(elf).parent

    with OpenOCDTCL(port=TCL_PORT) as tcl:
        while True:
            iteration += 1
            print(f"── iteração {iteration} "
                  f"{'─' * max(0, 50 - len(str(iteration)))}")
            print(f"[i] Aguardando breakpoint (timeout={TIMEOUT_S:.0f}s)...")

            tcl.set_bp(bp_addr, 2)
            tcl.send("catch {resume}")

            try:
                tcl.wait_halt(int(TIMEOUT_S * 1000))
            except Exception as exc:
                try:
                    tcl.remove_bp(bp_addr)
                    tcl.send("catch {resume}")
                except Exception:
                    pass
                print(f"[!] Timeout: {exc}")
                continue

            tcl.remove_bp(bp_addr)
            print("[i] Breakpoint atingido. Lendo tensores via TCL...")

            result = _capture(tcl, symbols, tmp_dir)

            Path(json_path).write_text(json.dumps(result, indent=2))

            plot_capture(
                capture_json     = json_path,
                npz_path         = npz,
                output_path      = png_path,
                layout           = LAYOUT,
                show_ticks       = SHOW_TICKS,
                show_interactive = True,
                window_pos       = WINDOW_POS,
                fig              = fig,
            )
            fig = plt.gcf()
            plt.pause(1.0)

            print("[i] Retomando firmware...")
            tcl.send("catch {resume}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[i] Encerrado pelo usuário.")
        plt.close('all')
