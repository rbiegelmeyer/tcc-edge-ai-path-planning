#!/usr/bin/env python3
"""
stm32_debug_link.py
===================
Comunicação host <-> STM32 em modo debug (SWD) usando OpenOCD.

Em vez de GDB, este script fala DIRETO com a interface TCL RPC do OpenOCD
(porta 6666). Para ler/escrever memória isso é mais simples e mais rápido
do que passar por GDB.

    seu_script.py --(TCL/6666)--> OpenOCD --(USB)--> ST-Link --(SWD)--> STM32

Requisitos:
    - OpenOCD >= 0.11  (precisa dos comandos read_memory / write_memory)
    - Python 3.8+      (só biblioteca padrão)
    - OpenOCD rodando com o tcl_port habilitado (ver openocd_stm32.cfg)

Modos de uso:
    1) Modo buffer compartilhado (protocolo original):
       python stm32_debug_link.py --elf firmware.elf --values 1 2 3 4

    2) Modo captura por breakpoint:
       python stm32_debug_link.py --elf firmware.elf --capture
       python stm32_debug_link.py --elf firmware.elf --capture --out resultado.json

Autor: exemplo de referência — ajuste os pontos marcados com  # >>> EDITAR
"""

import argparse
import json
import re
import socket
import struct
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path


# ---------------------------------------------------------------------------
# Caminhos das ferramentas ARM (toolchain STM32CubeIDE)
# ---------------------------------------------------------------------------
_TOOLCHAIN = (
    r"C:\ST\STM32CubeIDE_1.15.0\STM32CubeIDE\plugins"
    r"\com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32"
    r".14.3.rel1.win32_1.0.100.202602081740\tools\bin"
)
NM_EXE  = str(Path(_TOOLCHAIN) / "arm-none-eabi-nm.exe")
GDB_EXE = str(Path(_TOOLCHAIN) / "arm-none-eabi-gdb.exe")

# Nome do target OpenOCD (definido em stm32h7x.cfg como <CHIPNAME>.cpu0)
OPENOCD_TARGET = "stm32h7x"

# ---------------------------------------------------------------------------
# Parâmetros do breakpoint de captura (app_x-cube-ai.c:251)
# ---------------------------------------------------------------------------
# CAPTURE_SOURCE_FILE = "main.c"
# CAPTURE_LINE        = 169
CAPTURE_SOURCE_FILE = "app_x-cube-ai.c"
CAPTURE_LINE        = 255

# Tamanhos e metadados dos tensores (gerados em network.h pelo X-CUBE-AI)
# Input:  STAI_NETWORK_IN_1_SIZE  = 12288 bytes, formato S8, shape [1,64,64,3]
# Output: STAI_NETWORK_OUT_1_SIZE = 4096  bytes, formato S8, shape [1,64,1,64]
INPUT_BYTES   = 12288
OUTPUT_BYTES  = 4096
INPUT_SHAPE   = [1, 64, 64, 3]
OUTPUT_SHAPE  = [1, 64, 1, 64]
INPUT_SCALE   = 0.003921568859368563
INPUT_ZP      = -128
OUTPUT_SCALE  = 0.00392156932502985
OUTPUT_ZP     = -128

# map_solved: resultado do A* no STM32 — int8_t[64][64], valor 2 = caminho
SOLVED_BYTES  = 4096   # 64 * 64 * sizeof(int8_t)
SOLVED_SHAPE  = [64, 64]


# ---------------------------------------------------------------------------
# Camada baixa: cliente TCL RPC do OpenOCD
# ---------------------------------------------------------------------------
class OpenOCDTCL:
    """Cliente mínimo para o servidor TCL RPC do OpenOCD (padrão: porta 6666).

    O protocolo é trivial: você envia um comando TCL terminado pelo byte 0x1a
    e lê a resposta até receber outro 0x1a.
    """

    COMMAND_TOKEN = b"\x1a"

    def __init__(self, host="127.0.0.1", port=6666, timeout=5.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock = None

    # --- ciclo de vida -----------------------------------------------------
    def connect(self):
        self.sock = socket.create_connection((self.host, self.port), self.timeout)
        self.sock.settimeout(self.timeout)
        return self

    def close(self):
        if self.sock:
            try:
                self.sock.close()
            finally:
                self.sock = None

    def __enter__(self):
        return self.connect()

    def __exit__(self, *exc):
        self.close()

    # --- comando cru -------------------------------------------------------
    def send(self, command: str) -> str:
        """Envia um comando TCL e devolve a resposta como string."""
        if self.sock is None:
            raise ConnectionError("Nao conectado ao OpenOCD (chame connect()).")
        self.sock.sendall(command.encode("utf-8") + self.COMMAND_TOKEN)
        return self._recv()

    def _recv(self) -> str:
        buf = bytearray()
        while True:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("OpenOCD fechou a conexao.")
            buf.extend(chunk)
            if buf.endswith(self.COMMAND_TOKEN):
                return buf[:-1].decode("utf-8", errors="replace")

    # --- controle do alvo --------------------------------------------------
    def halt(self):        return self.send("halt")
    def resume(self):      return self.send("resume")
    def reset_halt(self):  return self.send("reset halt")
    def reset_run(self):   return self.send("reset run")

    # --- breakpoints de hardware ------------------------------------------
    def set_bp(self, address: int, size: int = 2) -> str:
        """Define um breakpoint de hardware no endereço Flash informado.

        size=2 para instruções Thumb (16-bit), size=4 para ARM/Thumb-32.
        """
        return self.send(f"bp 0x{address:08x} {size} hw")

    def remove_bp(self, address: int) -> str:
        """Remove o breakpoint de hardware no endereço informado."""
        return self.send(f"rbp 0x{address:08x}")

    def wait_halt(self, timeout_ms: int = 30000) -> str:
        """Aguarda o alvo entrar em halt. timeout_ms em milissegundos."""
        # Aumenta o timeout do socket para não expirar antes do OpenOCD
        old = self.sock.gettimeout()
        self.sock.settimeout(timeout_ms / 1000.0 + 2.0)
        try:
            return self.send(f"wait_halt {timeout_ms}")
        finally:
            self.sock.settimeout(old)

    # --- acesso a memoria --------------------------------------------------
    def read_word(self, address: int) -> int:
        """Le uma palavra de 32 bits via mrw (retorna valor pelo TCL RPC)."""
        return self._mrw(address, 1)[0]

    def write_word(self, address: int, value: int):
        """Escreve uma palavra de 32 bits."""
        self.send(f"mww 0x{address:08x} 0x{value & 0xFFFFFFFF:x}")

    def _mrw(self, address: int, count: int) -> list:
        """Le `count` palavras de 32 bits via mrw (retorna valor pelo TCL RPC).

        `mdw` apenas imprime no console do OpenOCD; `mrw` retorna o valor
        como string TCL — único modo confiável de ler memória via RPC.

        Para count > 1 envia um único script TCL com for-loop + mrw para
        minimizar round-trips, mas ainda faz uma leitura SWD por palavra.
        """
        if count == 1:
            resp = self.send(f"mrw 0x{address:08x}").strip()
            if not resp:
                raise RuntimeError(
                    f"mrw vazio @ 0x{address:08x} — "
                    f"target halted? verifique openocd.cfg"
                )
            try:
                return [int(resp, 0)]
            except ValueError:
                raise RuntimeError(
                    f"mrw falhou @ 0x{address:08x}: {resp!r}"
                )

        # Script TCL único: lê `count` palavras e devolve separadas por espaço
        tcl = (
            f"set _r {{}}; "
            f"for {{set _i 0}} {{$_i < {count}}} {{incr _i}} {{ "
            f"lappend _r [mrw [expr 0x{address:08x} + $_i * 4]] }}; "
            f"join $_r {{ }}"
        )
        # Script pode demorar — aumenta timeout proporcional ao count
        old = self.sock.gettimeout()
        self.sock.settimeout(max(old, count * 0.005 + 5.0))
        try:
            resp = self.send(tcl)
        finally:
            self.sock.settimeout(old)

        tokens = resp.split()
        if not tokens:
            raise RuntimeError(
                f"mrw loop vazio @ 0x{address:08x} [{count} words]: {resp[:200]!r}"
            )
        try:
            return [int(t, 0) for t in tokens]
        except ValueError:
            raise RuntimeError(
                f"mrw loop parse error @ 0x{address:08x}: {resp[:200]!r}"
            )

    def read_memory(self, address: int, count: int, width: int = 32):
        """Le `count` valores de `width` bits a partir de `address`.

        Usa o comando read_memory (OpenOCD >= 0.11), que devolve uma lista.
        Funciona com o nucleo rodando em Cortex-M0/M3/M4 (acesso via AHB-AP).
        """
        resp = self.send(f"read_memory 0x{address:08x} {width} {count}")
        tokens = resp.split()
        try:
            return [int(tok, 0) for tok in tokens]
        except ValueError:
            raise RuntimeError(
                f"read_memory falhou @ 0x{address:08x} "
                f"[{count}x{width}bit]: {resp[:200]!r}"
            )

    def write_memory(self, address: int, values, width: int = 32):
        """Escreve uma sequencia de valores a partir de `address`."""
        data = " ".join(str(int(v) & 0xFFFFFFFF) for v in values)
        self.send(f"write_memory 0x{address:08x} {width} {{{data}}}")


# ---------------------------------------------------------------------------
# Utilitario: resolver o endereco de um simbolo a partir do ELF
# ---------------------------------------------------------------------------
def resolve_symbol(elf_path: str, symbol: str, nm=NM_EXE) -> int:
    """Descobre o endereco de um simbolo (ex.: g_comm) lendo o ELF com `nm`.

    Evita hardcodar enderecos de RAM, que mudam a cada build.
    Requer arm-none-eabi-nm no PATH (vem com o toolchain GCC ARM).
    """
    out = subprocess.check_output([nm, elf_path], text=True)
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2] == symbol:
            return int(parts[0], 16)
    raise KeyError(f"Simbolo '{symbol}' nao encontrado em {elf_path}")


def resolve_line_address(elf_path: str,
                         source_file: str,
                         line: int,
                         gdb: str = GDB_EXE) -> int:
    """Retorna o endereço Flash de uma linha de código-fonte via arm-none-eabi-gdb.

    Usa 'info line <file>:<line>' no modo batch para obter o endereço
    sem precisar conectar ao alvo.
    """
    cmd = [gdb, "-batch", "-nx",
           "-ex", f"info line {source_file}:{line}",
           elf_path]
    try:
        out = subprocess.check_output(cmd, text=True,
                                      stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        out = e.output  # GDB pode sair com código != 0, mas ainda imprime o resultado

    m = re.search(r"starts at address (0x[0-9a-fA-F]+)", out)
    if not m:
        raise ValueError(
            f"Nao foi possivel resolver {source_file}:{line}.\n"
            f"Saida GDB:\n{out}"
        )
    return int(m.group(1), 16)


# ---------------------------------------------------------------------------
# Helper TCL: configura eventos GDB no OpenOCD antes de conectar o GDB
# ---------------------------------------------------------------------------
def _configure_gdb_events(tcl_host: str = "127.0.0.1",
                           tcl_port: int = 6666) -> None:
    """Sobrescreve os eventos gdb-attach e gdb-detach do target via TCL RPC.

    Por padrão, stm32h7x.cfg configura gdb-attach como 'reset init', o que
    reseta o firmware toda vez que o GDB conecta. Aqui substituímos por:
      gdb-attach  → halt   (apenas para, sem reset)
      gdb-detach  → resume (retoma automaticamente ao desconectar)

    O nome do target é auto-detectado via 'target names' para evitar erros
    de configuração quando o config do OpenOCD for diferente do esperado.
    """
    try:
        with OpenOCDTCL(host=tcl_host, port=tcl_port) as tcl:
            # Auto-detecta o(s) target(s) disponível(is)
            resp = tcl.send("target names").strip()
            targets = resp.split() if resp else []
            if not targets:
                print("[!] _configure_gdb_events: nenhum target encontrado via 'target names'")
                print(f"    resposta bruta: {resp!r}")
                return
            print(f"[i] Targets OpenOCD detectados: {targets}")
            for t in targets:
                tcl.send(f"{t} configure -event gdb-attach  {{ halt }}")
                tcl.send(f"{t} configure -event gdb-detach  {{ resume }}")
            print(f"[i] Eventos GDB configurados: attach=halt, detach=resume")
    except Exception as e:
        print(f"[!] Nao foi possivel configurar eventos GDB via TCL: {e}")
        print("[!] O firmware pode ser resetado ao conectar o GDB.")


# ---------------------------------------------------------------------------
# Helper GDB: executa um script em modo batch e retorna stdout+stderr
# ---------------------------------------------------------------------------
def _run_gdb(elf_path: str, script: str,
             timeout: float = 60.0, gdb: str = GDB_EXE) -> str:
    """Grava o script em arquivo temporário e executa arm-none-eabi-gdb -batch.

    Retorna a saída combinada (stdout + stderr).
    Lança TimeoutError se o processo não terminar dentro de `timeout` segundos.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".gdb",
                                     delete=False, encoding="utf-8") as f:
        f.write(script)
        script_path = f.name
    try:
        result = subprocess.run(
            [gdb, "-batch", "-n", "-x", script_path, elf_path],
            text=True, capture_output=True, timeout=timeout,
        )
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        raise TimeoutError(
            f"GDB nao terminou em {timeout:.0f}s — "
            f"breakpoint atingido? firmware rodando?"
        )
    finally:
        Path(script_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Status instantâneo do target via GDB
# ---------------------------------------------------------------------------
def print_status(elf_path: str,
                 gdb_port: int = 3333) -> None:
    """Conecta via GDB ao OpenOCD (porta gdb_port), interrompe o target
    brevemente e imprime: PC/SP/LR/xPSR, x_val_idx e ponteiros dos tensores.

    NOTA: feche qualquer sessão de debug do STM32CubeIDE antes de usar,
    pois apenas um cliente GDB pode conectar ao OpenOCD por vez.
    """
    print("[i] Resolvendo simbolos via ELF...")
    in_sym_addr  = resolve_symbol(elf_path, "stai_input")
    out_sym_addr = resolve_symbol(elf_path, "stai_output")
    idx_addr     = resolve_symbol(elf_path, "x_val_idx")

    # Garante que GDB não resetará o target ao conectar/desconectar
    _configure_gdb_events()

    script = f"""\
set pagination off
set confirm off
target remote :{gdb_port}
interrupt
info registers pc sp lr xpsr
printf "STATUS x_val_idx=%d\\n",   *(int*)          0x{idx_addr:08x}
printf "STATUS in_ptr=0x%08x\\n",  *(unsigned int*) 0x{in_sym_addr:08x}
printf "STATUS out_ptr=0x%08x\\n", *(unsigned int*) 0x{out_sym_addr:08x}
quit
"""
    print(f"[i] Conectando ao GDB server (OpenOCD porta {gdb_port})...")
    output = _run_gdb(elf_path, script, timeout=15.0)

    # --- parse registradores ---
    def _hex(pattern):
        m = re.search(pattern, output, re.IGNORECASE)
        return int(m.group(1), 16) if m else None

    pc   = _hex(r"\bpc\b\s+(0x[0-9a-f]+)")
    sp   = _hex(r"\bsp\b\s+(0x[0-9a-f]+)")
    lr   = _hex(r"\blr\b\s+(0x[0-9a-f]+)")
    xpsr = _hex(r"\bxpsr\b\s+(0x[0-9a-f]+)")

    m_idx = re.search(r"STATUS x_val_idx=(\d+)", output)
    m_in  = re.search(r"STATUS in_ptr=(0x[0-9a-f]+)", output, re.IGNORECASE)
    m_out = re.search(r"STATUS out_ptr=(0x[0-9a-f]+)", output, re.IGNORECASE)

    x_val_idx_val = int(m_idx.group(1))        if m_idx else None
    input_ptr     = int(m_in.group(1),  16)    if m_in  else None
    output_ptr    = int(m_out.group(1), 16)    if m_out else None

    print()
    print("=" * 52)
    print("  ESTADO DO TARGET  STM32H743")
    print("=" * 52)
    fmt = lambda v: f"0x{v:08x}" if v is not None else "n/a"
    print(f"  PC               : {fmt(pc)}")
    print(f"  SP               : {fmt(sp)}")
    print(f"  LR               : {fmt(lr)}")
    print(f"  xPSR             : {fmt(xpsr)}")
    print("-" * 52)
    idx_str = str(x_val_idx_val) if x_val_idx_val is not None else "n/a"
    print(f"  x_val_idx        : {idx_str}  (@ 0x{idx_addr:08x})")
    print(f"  stai_input[0] ptr: {fmt(input_ptr)}  (array @ 0x{in_sym_addr:08x})")
    print(f"  stai_output[0]ptr: {fmt(output_ptr)}  (array @ 0x{out_sym_addr:08x})")
    print("=" * 52)
    print()

    if not any([pc, x_val_idx_val is not None, input_ptr]):
        print("[!] Saida GDB bruta (para debug):")
        print(output[:800])


# ---------------------------------------------------------------------------
# Captura por breakpoint via GDB: aguarda hit, faz dump dos tensores
# ---------------------------------------------------------------------------
def capture_tensors(elf_path: str,
                    gdb_port: int = 3333,
                    timeout_s: float = 30.0,
                    output_json: str = None) -> dict:
    """Conecta via GDB, define breakpoint em CAPTURE_SOURCE_FILE:CAPTURE_LINE,
    aguarda o hit e captura stai_input[0] e stai_output[0] como listas de int8.

    Usa 'dump binary memory' do GDB — o protocolo GDB Remote lida corretamente
    com a seleção de AP para cada região de memória (AXI SRAM, DTCM etc.).

    NOTA: feche qualquer sessão de debug do STM32CubeIDE antes de usar.
    """
    print("[i] Resolvendo simbolos via ELF...")
    in_sym_addr      = resolve_symbol(elf_path, "stai_input")
    out_sym_addr     = resolve_symbol(elf_path, "stai_output")
    idx_addr         = resolve_symbol(elf_path, "x_val_idx")
    map_solved_addr  = resolve_symbol(elf_path, "map_solved")
    bp_addr          = resolve_line_address(elf_path, CAPTURE_SOURCE_FILE, CAPTURE_LINE)

    print(f"[i] &stai_input[0]  @ 0x{in_sym_addr:08x}")
    print(f"[i] &stai_output[0] @ 0x{out_sym_addr:08x}")
    print(f"[i] &x_val_idx      @ 0x{idx_addr:08x}")
    print(f"[i] &map_solved     @ 0x{map_solved_addr:08x}")
    print(f"[i] breakpoint      @ 0x{bp_addr:08x}  ({CAPTURE_SOURCE_FILE}:{CAPTURE_LINE})")

    # Configura eventos GDB (gdb-attach=halt, gdb-detach=resume)
    _configure_gdb_events()

    # ── Fase 1: aguarda breakpoint via TCL ────────────────────────────────────
    # TCL gerencia o BP; GDB só lê memória do target já parado.
    # Isso evita o problema de 'continue' travar no state-machine do GDB
    # quando o gdb-attach halt chega no meio do handshake.
    print(f"[i] Inserindo breakpoint hw @ 0x{bp_addr:08x} via TCL...")
    print(f"[i] Aguardando hit (timeout={timeout_s:.0f}s)...")

    with OpenOCDTCL() as tcl:
        tcl.set_bp(bp_addr, 2)           # hw breakpoint (Thumb = 2 bytes)
        tcl.send("catch {resume}")       # resume se estiver halted; no-op se já rodando
        try:
            tcl.wait_halt(int(timeout_s * 1000))
        except Exception as exc:
            try:
                tcl.remove_bp(bp_addr)
                tcl.resume()
            except Exception:
                pass
            raise TimeoutError(
                f"Timeout aguardando breakpoint @ 0x{bp_addr:08x} "
                f"({CAPTURE_SOURCE_FILE}:{CAPTURE_LINE})"
            ) from exc
        tcl.remove_bp(bp_addr)   # remove BP antes que o GDB conecte

    print(f"[i] Breakpoint atingido. Conectando GDB para leitura dos tensores...")

    # ── Fase 2: lê memória via GDB (target já parado no BP) ──────────────────
    elf_dir    = Path(elf_path).parent
    tmp_in     = (elf_dir / "_cap_input.bin").as_posix()
    tmp_out    = (elf_dir / "_cap_output.bin").as_posix()
    tmp_solved = (elf_dir / "_cap_solved.bin").as_posix()

    # map_solved é um array global (não ponteiro): endereço = &map_solved[0][0]
    map_solved_end = map_solved_addr + SOLVED_BYTES

    script = f"""\
set pagination off
set confirm off
target remote :{gdb_port}
set $in_ptr  = *(unsigned int*) 0x{in_sym_addr:08x}
set $out_ptr = *(unsigned int*) 0x{out_sym_addr:08x}
printf "CAPTURE x_val_idx=%d\\n",              *(int*)          0x{idx_addr:08x}
printf "CAPTURE in_ptr=0x%08x out_ptr=0x%08x\\n", $in_ptr, $out_ptr
dump binary memory {tmp_in}     $in_ptr  $in_ptr+{INPUT_BYTES}
dump binary memory {tmp_out}    $out_ptr $out_ptr+{OUTPUT_BYTES}
dump binary memory {tmp_solved} 0x{map_solved_addr:08x} 0x{map_solved_end:08x}
printf "CAPTURE done\\n"
monitor resume
quit
"""

    output = _run_gdb(elf_path, script, timeout=60.0)

    if "CAPTURE done" not in output:
        print("[!] Saida GDB bruta:")
        print(output)
        raise RuntimeError(
            "GDB nao completou a captura — breakpoint atingido? "
            "Outro cliente GDB conectado?"
        )

    # Lê os dumps binários
    path_in     = Path(elf_path).parent / "_cap_input.bin"
    path_out    = Path(elf_path).parent / "_cap_output.bin"
    path_solved = Path(elf_path).parent / "_cap_solved.bin"

    in_bytes     = path_in.read_bytes()
    out_bytes    = path_out.read_bytes()
    solved_bytes = path_solved.read_bytes()

    path_in.unlink(missing_ok=True)
    path_out.unlink(missing_ok=True)
    path_solved.unlink(missing_ok=True)

    if len(in_bytes) != INPUT_BYTES:
        raise RuntimeError(
            f"Dump input com tamanho errado: {len(in_bytes)} != {INPUT_BYTES}"
        )
    if len(out_bytes) != OUTPUT_BYTES:
        raise RuntimeError(
            f"Dump output com tamanho errado: {len(out_bytes)} != {OUTPUT_BYTES}"
        )
    if len(solved_bytes) != SOLVED_BYTES:
        raise RuntimeError(
            f"Dump map_solved com tamanho errado: {len(solved_bytes)} != {SOLVED_BYTES}"
        )

    in_data     = list(struct.unpack(f"{INPUT_BYTES}b",  in_bytes))
    out_data    = list(struct.unpack(f"{OUTPUT_BYTES}b", out_bytes))
    solved_data = list(struct.unpack(f"{SOLVED_BYTES}b", solved_bytes))

    # Extrai x_val_idx e ponteiros da saída GDB para log
    m_idx = re.search(r"CAPTURE x_val_idx=(\d+)", output)
    m_ptr = re.search(r"CAPTURE in_ptr=(0x[0-9a-f]+) out_ptr=(0x[0-9a-f]+)",
                      output, re.IGNORECASE)

    x_val_idx_val = int(m_idx.group(1)) if m_idx else None

    print(f"[i] x_val_idx      = {x_val_idx_val if x_val_idx_val is not None else 'n/a'}")
    if m_ptr:
        print(f"[i] stai_input[0]  -> dados @ {m_ptr.group(1)}")
        print(f"[i] stai_output[0] -> dados @ {m_ptr.group(2)}")
    n_path = sum(1 for v in solved_data if v == 2)
    print(f"[i] map_solved     @ 0x{map_solved_addr:08x}  ({n_path} células de caminho A*)")

    result = {
        "capture_point": f"{CAPTURE_SOURCE_FILE}:{CAPTURE_LINE}",
        "x_val_idx": x_val_idx_val,
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

    if output_json:
        Path(output_json).write_text(json.dumps(result, indent=2))
        print(f"[i] Resultado salvo em: {output_json}")
    else:
        print(f"\n--- stai_input[0]  (primeiros 32 valores) ---")
        print(in_data[:32])
        print(f"\n--- stai_output[0] (primeiros 32 valores) ---")
        print(out_data[:32])

    return result


# ---------------------------------------------------------------------------
# Camada alta: protocolo de buffer compartilhado (modo original)
# ---------------------------------------------------------------------------
class STM32DataLink:
    """Implementa o handshake host<->firmware sobre um buffer em RAM.

    Layout do buffer (deve casar com firmware_comm_example.c):

        offset  campo          dir.        significado
        0x00    input_ready    host->fw    host=1 quando a entrada esta pronta
        0x04    output_ready   fw->host    fw=1 quando a saida esta pronta
        0x08    input_len      host->fw    nº de palavras validas em input[]
        0x0C    output_len     fw->host    nº de palavras validas em output[]
        0x10    input[CAP]     host->fw
        ...     output[CAP]    fw->host
    """

    OFF_INPUT_READY = 0x00
    OFF_OUTPUT_READY = 0x04
    OFF_INPUT_LEN = 0x08
    OFF_OUTPUT_LEN = 0x0C
    OFF_INPUT = 0x10

    def __init__(self, ocd: OpenOCDTCL, base_addr: int, capacity: int = 64):
        self.ocd = ocd
        self.base = base_addr
        self.capacity = capacity
        self.off_output = self.OFF_INPUT + capacity * 4

    def send_input(self, values):
        """Escreve a entrada e sinaliza o firmware para processar."""
        if len(values) > self.capacity:
            raise ValueError(f"Entrada excede a capacidade ({self.capacity}).")
        self.ocd.write_word(self.base + self.OFF_OUTPUT_READY, 0)
        self.ocd.write_memory(self.base + self.OFF_INPUT, values)
        self.ocd.write_word(self.base + self.OFF_INPUT_LEN, len(values))
        self.ocd.write_word(self.base + self.OFF_INPUT_READY, 1)

    def wait_output(self, timeout=2.0, poll_interval=0.005) -> bool:
        """Espera o firmware sinalizar output_ready. True se chegou a tempo."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.ocd.read_word(self.base + self.OFF_OUTPUT_READY) == 1:
                return True
            time.sleep(poll_interval)
        return False

    def read_output(self):
        """Le a saida processada (output_len palavras)."""
        n = self.ocd.read_word(self.base + self.OFF_OUTPUT_LEN)
        n = min(n, self.capacity)
        return self.ocd.read_memory(self.base + self.off_output, n)

    def process(self, values, timeout=2.0):
        """Ciclo completo: envia entrada, espera e devolve a saida."""
        self.send_input(values)
        if not self.wait_output(timeout):
            raise TimeoutError("Firmware nao respondeu (output_ready).")
        return self.read_output()


# ---------------------------------------------------------------------------
# Demonstracao / CLI
# ---------------------------------------------------------------------------
@contextmanager
def link(base_addr, host, port, capacity):
    ocd = OpenOCDTCL(host=host, port=port).connect()
    try:
        yield STM32DataLink(ocd, base_addr, capacity)
    finally:
        ocd.close()


def main():
    p = argparse.ArgumentParser(description="Troca de dados com STM32 via OpenOCD.")
    p.add_argument("--host", default="127.0.0.1", help="host do OpenOCD")
    p.add_argument("--port", type=int, default=6666, help="porta TCL do OpenOCD")
    p.add_argument("--elf", help="ELF do firmware")

    # --- modo status instantâneo ---
    p.add_argument("--status", action="store_true",
                   help="Imprime estado atual do target (PC, SP, x_val_idx, ponteiros dos tensores)")

    # --- modo captura por breakpoint ---
    p.add_argument("--capture", action="store_true",
                   help=f"Define breakpoint em {CAPTURE_SOURCE_FILE}:{CAPTURE_LINE}, "
                        f"aguarda e captura stai_input[0] / stai_output[0]")
    p.add_argument("--gdb-port", type=int, default=3333,
                   help="Porta GDB do OpenOCD (default: 3333)")
    p.add_argument("--timeout", type=float, default=30.0,
                   help="Timeout em segundos para aguardar o breakpoint (default: 30)")
    p.add_argument("--out", metavar="FILE.json",
                   help="Salva resultado da captura em JSON (opcional)")

    # --- modo buffer compartilhado (original) ---
    p.add_argument("--base", type=lambda x: int(x, 0),
                   help="endereco do buffer g_comm (ex.: 0x20000000)")
    p.add_argument("--symbol", default="g_comm",
                   help="nome do simbolo do buffer compartilhado")
    p.add_argument("--capacity", type=int, default=64,
                   help="palavras por buffer (modo compartilhado)")
    p.add_argument("--values", type=int, nargs="+", default=[1, 2, 3, 4],
                   help="valores de entrada a enviar (modo compartilhado)")

    args = p.parse_args()

    try:
        # ------------------------------------------------------------------
        # Modo status instantâneo
        # ------------------------------------------------------------------
        if args.status:
            if not args.elf:
                p.error("--status requer --elf firmware.elf")
            print_status(elf_path=args.elf, gdb_port=args.gdb_port)
            return

        # ------------------------------------------------------------------
        # Modo captura por breakpoint
        # ------------------------------------------------------------------
        if args.capture:
            if not args.elf:
                p.error("--capture requer --elf firmware.elf")
            capture_tensors(
                elf_path=args.elf,
                gdb_port=args.gdb_port,
                timeout_s=args.timeout,
                output_json=args.out,
            )
            return

        # ------------------------------------------------------------------
        # Modo buffer compartilhado (original)
        # ------------------------------------------------------------------
        if args.base is None and args.elf:
            args.base = resolve_symbol(args.elf, args.symbol)
            print(f"[i] {args.symbol} @ 0x{args.base:08x} (resolvido do ELF)")
        if args.base is None:
            p.error("informe --base 0x... ou --elf firmware.elf")

        with link(args.base, args.host, args.port, args.capacity) as dl:
            print(f"[i] enviando entrada: {args.values}")
            out = dl.process(args.values)
            print(f"[i] saida processada: {out}")

    except ConnectionRefusedError:
        print("[x] OpenOCD nao esta escutando. Rode primeiro:\n"
              "    openocd -f openocd_stm32.cfg")
    except (TimeoutError, ConnectionError, KeyError, ValueError) as e:
        print(f"[x] erro: {e}")


if __name__ == "__main__":
    main()
