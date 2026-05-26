"""
Gera imagens comparativas empilhando verticalmente os resultados do
modelo professor (teacher) e do modelo aluno (student).

  Linha superior : teacher  (ex: test/Unet/)
  Linha inferior : student  (ex: test/distilled_Student_Unet/)
  Saída          : test/comparacao_teacher_student/comparacao_<i>.png

Uso:
    python compare_results.py --results ./results/result_W064xH064_D01_S000000_E005000
    python compare_results.py --results ./results/... --teacher Unet --student distilled_Student_Unet
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
        description='Comparação teacher vs student — empilhamento vertical de imagens'
    )
    parser.add_argument('--results', required=True,
                        help='Pasta de resultados (ex: ./results/result_W064xH064_D01_...)')
    parser.add_argument('--teacher', default='Unet',
                        help='Subpasta do teacher em test/ (padrão: Unet)')
    parser.add_argument('--student', default=None,
                        help='Subpasta do student em test/ (padrão: auto-detecta distilled_*)')
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

    output_dir = os.path.join(test_dir, 'comparacao_teacher_student')

    print(f'Teacher : {teacher_dir}')
    print(f'Student : {student_dir}')
    print(f'Saída   : {output_dir}')
    print()

    n = combine_images(teacher_dir, student_dir, output_dir,
                       teacher_label=f'Professor\n({args.teacher})',
                       student_label=f'Aluno\n({student_name})')
    print(f'\n{n} imagens geradas em: {output_dir}')


if __name__ == '__main__':
    main()
