"""
Gera imagens comparativas entre o modelo professor (teacher) e o modelo aluno (student).

Modos disponíveis:

  vertical (padrão)
    Linha superior : teacher  (ex: test/Unet/)
    Linha inferior : student  (ex: test/distilled_Student_Unet/)
    Saída          : test/comparacao_teacher_student/comparacao_<i>.png

  4painel
    4 mapas lado a lado igualmente distribuídos:
      Input | Alvo Real | Previsão UNet | Previsão CNN Student
    Saída  : test/comparacao_4paineis/comparacao_4paineis_<i>.png

Uso:
    python compare_results.py --results ./results/result_W064xH064_D01_S000000_E005000
    python compare_results.py --results ./results/... --teacher Unet --student distilled_Student_Unet
    python compare_results.py --results ./results/... --modo 4painel
"""

import argparse
import os
import sys
from glob import glob
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------
_SEP_PX    = 6          # espessura do separador entre as linhas
_SEP_COLOR = (180, 180, 180)   # cinza claro
_LABEL_W   = 90         # largura da coluna de label lateral
_LABEL_BG  = {
    'teacher': (40,  80, 160),   # azul escuro
    'student': (160,  70,  20),  # laranja escuro
}
_LABEL_FG  = (255, 255, 255)    # branco

# constantes exclusivas do modo 4-painel
_HEADER_H      = 30              # altura do cabeçalho de cada painel
_VSEP_PX       = 4               # espessura do separador vertical
_HEADER_COLORS = {
    'shared':  ( 70,  70,  70),  # cinza escuro – painéis compartilhados
    'teacher': ( 40,  80, 160),  # azul
    'student': (160,  70,  20),  # laranja
}
# ---------------------------------------------------------------


def _load_font(size=13):
    """Tenta carregar uma fonte TTF; cai para a fonte padrão do PIL."""
    for name in ['DejaVuSans-Bold.ttf', 'Arial.ttf', 'LiberationSans-Bold.ttf']:
        try:
            return ImageFont.truetype(name, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def _label_strip(height, text, role):
    """Cria uma faixa vertical colorida com o texto rotacionado."""
    strip = Image.new('RGB', (_LABEL_W, height), _LABEL_BG[role])
    draw  = ImageDraw.Draw(strip)
    font  = _load_font(13)

    # Desenha o texto em imagem temporária e rotaciona 90°
    tmp = Image.new('RGBA', (height, _LABEL_W), (0, 0, 0, 0))
    ImageDraw.Draw(tmp).text((10, 8), text, font=font, fill=_LABEL_FG)
    rotated = tmp.rotate(90, expand=True)
    strip.paste(rotated, (0, 0), mask=rotated.split()[3])
    return strip


def _separator(width):
    return Image.new('RGB', (width, _SEP_PX), _SEP_COLOR)


def _split_3panels(img):
    """Divide um mapa_predicao (1×3 subplots) nos três painéis individuais."""
    w, h = img.size
    pw = w // 3
    return [
        img.crop((0,      0, pw,    h)),
        img.crop((pw,     0, 2*pw,  h)),
        img.crop((2*pw,   0, w,     h)),
    ]


def _panel_with_header(panel, label, role, font):
    """Retorna o painel com uma faixa colorida de identificação no topo."""
    pw, ph = panel.size
    header = Image.new('RGB', (pw, _HEADER_H), _HEADER_COLORS[role])
    draw   = ImageDraw.Draw(header)
    try:
        bb = draw.textbbox((0, 0), label, font=font)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
    except AttributeError:
        tw, th = draw.textsize(label, font=font)
    draw.text(((pw - tw) // 2, (_HEADER_H - th) // 2), label, font=font, fill=_LABEL_FG)

    strip = Image.new('RGB', (pw, _HEADER_H + ph), (255, 255, 255))
    strip.paste(header, (0, 0))
    strip.paste(panel,  (0, _HEADER_H))
    return strip


def combine_images_4panel(teacher_dir, student_dir, output_dir,
                          teacher_label='UNet', student_label='CNN Student'):
    """
    Cria comparações com 4 painéis igualmente distribuídos lado a lado:
      Input | Alvo Real | Previsão <teacher> | Previsão <student>
    """
    os.makedirs(output_dir, exist_ok=True)

    def _sorted_imgs(folder):
        imgs = glob(os.path.join(folder, 'mapa_predicao_*.png'))
        return {int(Path(p).stem.split('_')[-1]): p for p in imgs}

    teacher_imgs = _sorted_imgs(teacher_dir)
    student_imgs = _sorted_imgs(student_dir)
    common = sorted(set(teacher_imgs) & set(student_imgs))

    if not common:
        print('[AVISO] Nenhum par de imagens encontrado.')
        return 0

    font = _load_font(12)

    for idx in common:
        t = Image.open(teacher_imgs[idx]).convert('RGB')
        s = Image.open(student_imgs[idx]).convert('RGB')

        t_panels = _split_3panels(t)
        s_panels = _split_3panels(s)

        # Dimensões uniformes: largura do painel do teacher, altura máxima
        pw = t_panels[0].width
        ph = max(p.height for p in t_panels + s_panels)

        panels_data = [
            (t_panels[0], 'Input',                   'shared'),
            (t_panels[1], 'Alvo Real',               'shared'),
            (t_panels[2], f'Pred. {teacher_label}',  'teacher'),
            (s_panels[2], f'Pred. {student_label}',  'student'),
        ]

        strips = []
        for panel, label, role in panels_data:
            panel = panel.resize((pw, ph), Image.LANCZOS)
            strips.append(_panel_with_header(panel, label, role, font))

        vsep      = Image.new('RGB', (_VSEP_PX, strips[0].height), _SEP_COLOR)
        total_w   = sum(s.width for s in strips) + _VSEP_PX * (len(strips) - 1)
        combined  = Image.new('RGB', (total_w, strips[0].height), (255, 255, 255))

        x = 0
        for i, strip in enumerate(strips):
            combined.paste(strip, (x, 0))
            x += strip.width
            if i < len(strips) - 1:
                combined.paste(vsep, (x, 0))
                x += _VSEP_PX

        out = os.path.join(output_dir, f'comparacao_4paineis_{idx}.png')
        combined.save(out)
        print(f'  Salvo: {out}')

    return len(common)


def combine_images(teacher_dir, student_dir, output_dir,
                   teacher_label='Professor', student_label='Aluno'):
    os.makedirs(output_dir, exist_ok=True)

    def _sorted_imgs(folder):
        imgs = glob(os.path.join(folder, 'mapa_predicao_*.png'))
        return {int(Path(p).stem.split('_')[-1]): p for p in imgs}

    teacher_imgs = _sorted_imgs(teacher_dir)
    student_imgs = _sorted_imgs(student_dir)
    common = sorted(set(teacher_imgs) & set(student_imgs))

    if not common:
        print('[AVISO] Nenhum par de imagens encontrado.')
        return 0

    for idx in common:
        t = Image.open(teacher_imgs[idx]).convert('RGB')
        s = Image.open(student_imgs[idx]).convert('RGB')

        # Garante mesma largura (redimensiona o menor)
        w = max(t.width, s.width)
        if t.width != w:
            t = t.resize((w, t.height), Image.LANCZOS)
        if s.width != w:
            s = s.resize((w, s.height), Image.LANCZOS)

        # Faixas laterais de identificação
        t_label = _label_strip(t.height, teacher_label, 'teacher')
        s_label = _label_strip(s.height, student_label, 'student')

        # Linha superior (label + imagem do teacher)
        top = Image.new('RGB', (_LABEL_W + w, t.height))
        top.paste(t_label, (0, 0))
        top.paste(t, (_LABEL_W, 0))

        # Linha inferior (label + imagem do student)
        bot = Image.new('RGB', (_LABEL_W + w, s.height))
        bot.paste(s_label, (0, 0))
        bot.paste(s, (_LABEL_W, 0))

        sep = _separator(_LABEL_W + w)

        # Empilha verticalmente
        total_h = top.height + _SEP_PX + bot.height
        combined = Image.new('RGB', (_LABEL_W + w, total_h), (255, 255, 255))
        combined.paste(top, (0, 0))
        combined.paste(sep, (0, top.height))
        combined.paste(bot, (0, top.height + _SEP_PX))

        out = os.path.join(output_dir, f'comparacao_{idx}.png')
        combined.save(out)
        print(f'  Salvo: {out}')

    return len(common)


def _auto_detect_student(test_dir):
    candidates = [
        d for d in os.listdir(test_dir)
        if d.startswith('distilled_') and os.path.isdir(os.path.join(test_dir, d))
    ]
    return candidates[0] if candidates else None


def main():
    parser = argparse.ArgumentParser(
        description='Comparação teacher vs student'
    )
    parser.add_argument('--results', required=True,
                        help='Pasta de resultados (ex: ./results/result_W064xH064_D01_...)')
    parser.add_argument('--teacher', default='Unet',
                        help='Subpasta do teacher em test/ (padrão: Unet)')
    parser.add_argument('--student', default=None,
                        help='Subpasta do student em test/ (padrão: auto-detecta distilled_*)')
    parser.add_argument('--modo', default='vertical', choices=['vertical', '4painel'],
                        help='vertical: empilha professor/aluno  |  4painel: 4 mapas lado a lado (padrão: vertical)')
    args = parser.parse_args()

    test_dir = os.path.join(args.results, 'test')
    if not os.path.isdir(test_dir):
        print(f'[ERRO] Pasta não encontrada: {test_dir}')
        sys.exit(1)

    teacher_dir = os.path.join(test_dir, args.teacher)
    if not os.path.isdir(teacher_dir):
        print(f'[ERRO] Pasta do teacher não encontrada: {teacher_dir}')
        sys.exit(1)

    student_name = args.student or _auto_detect_student(test_dir)
    if not student_name:
        print('[ERRO] Nenhuma pasta de student encontrada. Use --student para especificar.')
        sys.exit(1)

    student_dir = os.path.join(test_dir, student_name)
    if not os.path.isdir(student_dir):
        print(f'[ERRO] Pasta do student não encontrada: {student_dir}')
        sys.exit(1)

    print(f'Teacher : {teacher_dir}')
    print(f'Student : {student_dir}')
    print(f'Modo    : {args.modo}')
    print()

    if args.modo == '4painel':
        output_dir = os.path.join(test_dir, 'comparacao_4paineis')
        print(f'Saída   : {output_dir}\n')
        n = combine_images_4panel(teacher_dir, student_dir, output_dir,
                                  teacher_label=args.teacher,
                                  student_label=student_name)
    else:
        output_dir = os.path.join(test_dir, 'comparacao_teacher_student')
        print(f'Saída   : {output_dir}\n')
        n = combine_images(teacher_dir, student_dir, output_dir,
                           teacher_label=f'Professor\n({args.teacher})',
                           student_label=f'Aluno\n({student_name})')

    print(f'\n{n} imagens geradas em: {output_dir}')


if __name__ == '__main__':
    main()
