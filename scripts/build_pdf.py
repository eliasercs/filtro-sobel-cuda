"""
Convierte docs/informe.md a docs/informe.pdf usando reportlab platypus.

Parser markdown simplificado que soporta: headings, tablas, listas,
parrafos, code blocks, blockquotes, hr, negrita, cursiva, codigo inline.

Uso:
    py -3.10 scripts\build_pdf.py
"""

import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether
)


def parse_inline(text):
    parts = re.split(r'(`[^`]+`)', text)
    out = []
    for p in parts:
        if p.startswith('`') and p.endswith('`'):
            out.append('<font face="Courier">' + p[1:-1] + '</font>')
        else:
            p = re.sub(r'\$\$([^$]+)\$\$', r'<i>\1</i>', p)
            p = re.sub(r'\$([^$]+)\$', r'<i>\1</i>', p)
            p = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', p)
            p = re.sub(r'(?<![*\w])\*([^*\n]+)\*(?![*\w])', r'<i>\1</i>', p)
            out.append(p)
    return ''.join(out)


def md_to_flowables(md_path, styles):
    md = md_path.read_text(encoding='utf-8')
    lines = md.split('\n')
    flow = []
    i = 0
    while i < len(lines):
        line = lines[i]

        if not line.strip():
            i += 1
            continue

        if line.startswith('# '):
            flow.append(Paragraph(parse_inline(line[2:].strip()), styles['H1']))
            flow.append(Spacer(1, 0.3 * cm))
            i += 1
            continue
        if line.startswith('## '):
            flow.append(Paragraph(parse_inline(line[3:].strip()), styles['H2']))
            flow.append(Spacer(1, 0.2 * cm))
            i += 1
            continue
        if line.startswith('### '):
            flow.append(Paragraph(parse_inline(line[4:].strip()), styles['H3']))
            i += 1
            continue
        if line.startswith('#### '):
            flow.append(Paragraph(parse_inline(line[5:].strip()), styles['H4']))
            i += 1
            continue

        if line.startswith('---'):
            flow.append(Spacer(1, 0.3 * cm))
            i += 1
            continue

        if line.startswith('> '):
            blockquote_lines = []
            while i < len(lines) and lines[i].startswith('> '):
                blockquote_lines.append(lines[i][2:])
                i += 1
            text = ' '.join(blockquote_lines).strip()
            flow.append(Paragraph(parse_inline(text), styles['Blockquote']))
            flow.append(Spacer(1, 0.1 * cm))
            continue

        if line.strip().startswith('|') and line.strip().endswith('|'):
            table_rows = []
            while i < len(lines) and lines[i].strip().startswith('|') and lines[i].strip().endswith('|'):
                cells = [c.strip() for c in lines[i].strip().strip('|').split('|')]
                table_rows.append(cells)
                i += 1
            if len(table_rows) >= 2 and all(set(c) <= set('-:') for c in table_rows[1]):
                table_rows.pop(1)
            data = []
            for row in table_rows:
                data.append([Paragraph(parse_inline(c), styles['TableCell']) for c in row])
            t = Table(data, repeatRows=1)
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#003366')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8.5),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('ALIGN', (0, 1), (-1, -1), 'LEFT'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f6f8fa')]),
                ('LEFTPADDING', (0, 0), (-1, -1), 4),
                ('RIGHTPADDING', (0, 0), (-1, -1), 4),
                ('TOPPADDING', (0, 0), (-1, -1), 3),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ]))
            flow.append(t)
            flow.append(Spacer(1, 0.3 * cm))
            continue

        if line.strip().startswith('- ') or line.strip().startswith('* '):
            list_items = []
            while i < len(lines) and (lines[i].strip().startswith('- ') or lines[i].strip().startswith('* ')):
                list_items.append(lines[i].strip()[2:])
                i += 1
            for item in list_items:
                flow.append(Paragraph('&bull;&nbsp; ' + parse_inline(item), styles['BodyText']))
            flow.append(Spacer(1, 0.1 * cm))
            continue

        if line.strip().startswith('```'):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code_lines.append(lines[i])
                i += 1
            i += 1
            code_text = '<br/>'.join(parse_inline(l) for l in code_lines)
            flow.append(Paragraph(code_text, styles['Code']))
            flow.append(Spacer(1, 0.2 * cm))
            continue

        paragraph_lines = []
        while i < len(lines) and lines[i].strip() and not (
            lines[i].startswith('#') or lines[i].startswith('---') or
            lines[i].startswith('> ') or lines[i].strip().startswith('|') or
            lines[i].strip().startswith('- ') or lines[i].strip().startswith('* ') or
            lines[i].strip().startswith('```')
        ):
            paragraph_lines.append(lines[i].strip())
            i += 1
        if paragraph_lines:
            text = ' '.join(paragraph_lines)
            flow.append(Paragraph(parse_inline(text), styles['BodyText']))
            flow.append(Spacer(1, 0.15 * cm))

    return flow


def make_styles():
    base = getSampleStyleSheet()
    styles = {
        'H1': ParagraphStyle('H1', parent=base['Heading1'],
                             fontSize=18, textColor=colors.HexColor('#003366'),
                             spaceAfter=10, spaceBefore=4),
        'H2': ParagraphStyle('H2', parent=base['Heading2'],
                             fontSize=14, textColor=colors.HexColor('#003366'),
                             spaceAfter=8, spaceBefore=10,
                             borderWidth=0, borderColor=colors.HexColor('#003366'),
                             borderPadding=2),
        'H3': ParagraphStyle('H3', parent=base['Heading3'],
                             fontSize=12, textColor=colors.HexColor('#003366'),
                             spaceAfter=4, spaceBefore=8),
        'H4': ParagraphStyle('H4', parent=base['Heading4'],
                             fontSize=10.5, textColor=colors.HexColor('#555555'),
                             spaceAfter=2, spaceBefore=4),
        'BodyText': ParagraphStyle('BodyText', parent=base['BodyText'],
                                   fontSize=10, leading=13,
                                   alignment=0),
        'Blockquote': ParagraphStyle('Blockquote', parent=base['BodyText'],
                                     fontSize=9.5, leading=12,
                                     leftIndent=20, textColor=colors.HexColor('#555555')),
        'Code': ParagraphStyle('Code', parent=base['Code'],
                               fontName='Courier', fontSize=8, leading=10,
                               leftIndent=10, backColor=colors.HexColor('#f6f8fa'),
                               borderColor=colors.HexColor('#003366'),
                               borderWidth=0, borderPadding=4),
        'TableCell': ParagraphStyle('TableCell', parent=base['BodyText'],
                                    fontSize=8.5, leading=10),
    }
    return styles


def main():
    base = Path(__file__).resolve().parent.parent
    md_path = base / 'docs' / 'informe.md'
    pdf_path = base / 'docs' / 'informe.pdf'

    if not md_path.exists():
        print(f"Error: no se encuentra {md_path}")
        return 1

    styles = make_styles()
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title='Informe técnico — Filtro Sobel CUDA',
        author='INFO1195 Actividad 4',
    )

    flow = md_to_flowables(md_path, styles)
    doc.build(flow)
    print(f"PDF generado: {pdf_path} ({pdf_path.stat().st_size // 1024} KB)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
