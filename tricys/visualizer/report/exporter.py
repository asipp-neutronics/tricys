"""Report Exporter Module for TRICYS 2.0.

Provides publication-grade multi-format export capabilities for engineering reports:
- Markdown (.md): Clean plain UTF-8 formatted academic text.
- Word (.docx): Professional Word document with Navy/Slate corporate palette,
  LaTeX tokenized math typography, zebra-striped styled tables, and embedded high-res images.
- PDF (.pdf): Vector-rendered PDF with CJK typography, tables, and embedded figures via PyMuPDF.
"""

from __future__ import annotations

import base64
import io
import os
import re
from typing import List

# Color Palette matching CFEDR 2000MW Study Standards
NAVY_PRIMARY = (30, 58, 138)  # #1E3A8A
SLATE_DARK = (31, 41, 55)  # #1F2937
SLATE_MUTED = (100, 116, 139)  # #64748B
BORDER_HEX = "CBD5E1"  # #CBD5E1
HEADER_BG_HEX = "1E3A8A"  # #1E3A8A
ZEBRA_BG_HEX = "F8FAFC"  # #F8FAFC
CALLOUT_BG_HEX = "F0F7FF"  # #F0F7FF
CALLOUT_BORDER_HEX = "2563EB"  # #2563EB


class ReportExporter:
    """Exports structured markdown reports into MD, DOCX, and PDF formats."""

    @classmethod
    def export_markdown(cls, markdown_text: str) -> str:
        """Returns clean markdown string."""
        return markdown_text.strip() + "\n"

    # ==========================================================================
    # Microsoft Word (.docx) Exporter
    # ==========================================================================
    @classmethod
    def export_docx(
        cls,
        markdown_text: str,
        title: str = "TRICYS Engineering Report",
    ) -> bytes:
        """Converts Markdown text into a publication-grade styled Microsoft Word (.docx) document."""
        import docx
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.oxml import OxmlElement, parse_xml
        from docx.oxml.ns import nsdecls, qn
        from docx.shared import Inches, Pt, RGBColor

        doc = docx.Document()

        # Set page margins (1 inch)
        for section in doc.sections:
            section.top_margin = Inches(1.0)
            section.bottom_margin = Inches(1.0)
            section.left_margin = Inches(1.0)
            section.right_margin = Inches(1.0)

        # Base style settings
        normal_style = doc.styles["Normal"]
        normal_style.font.name = "Calibri"
        normal_style.font.size = Pt(10.5)
        try:
            normal_style.element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
        except Exception:
            pass

        def set_run_font(
            run,
            font_name="Calibri",
            east_asia="Microsoft YaHei",
            size=Pt(10.5),
            color=None,
            bold=False,
            italic=False,
            subscript=False,
            superscript=False,
        ):
            run.font.name = font_name
            rPr = run._r.get_or_add_rPr()
            rFonts = rPr.find(qn("w:rFonts"))
            if rFonts is None:
                rFonts = OxmlElement("w:rFonts")
                rPr.append(rFonts)
            rFonts.set(qn("w:eastAsia"), east_asia)
            rFonts.set(qn("w:ascii"), font_name)
            rFonts.set(qn("w:hAnsi"), font_name)
            run.font.size = size
            run.font.bold = bold
            run.font.italic = italic
            run.font.subscript = subscript
            run.font.superscript = superscript
            if color:
                run.font.color.rgb = color

        def parse_math_tokens(math_text: str):
            s = math_text.strip("$").strip()
            s = re.sub(r"\\text\{([^}]+)\}", r"\1", s)
            s = s.replace(r"\dot{m}", "ṁ")
            s = s.replace(r"\Delta", "Δ")
            s = s.replace(r"\eta_f", "η_f")
            s = s.replace(r"\tau", "τ")
            s = s.replace(r"\approx", "≈")
            s = s.replace(r"\sim", "~")
            s = s.replace(r"\times", "×")
            s = s.replace(r"\cdot", "·")
            s = s.replace(r"\%", "%")
            s = s.replace(r"10^{-3}", "10⁻³")

            pattern = re.compile(r"([A-Za-zΔṁη_]+)_(\{([^}]+)\}|([A-Za-z0-9%]+))")
            runs = []
            last_idx = 0
            for m in pattern.finditer(s):
                pre = s[last_idx : m.start()]
                if pre:
                    runs.append({"text": pre, "sub": False, "sup": False})
                base = m.group(1)
                sub_text = m.group(3) if m.group(3) is not None else m.group(4)
                runs.append(
                    {"text": base, "sub": False, "sup": False, "math_var": True}
                )
                runs.append({"text": sub_text, "sub": True, "sup": False})
                last_idx = m.end()
            rest = s[last_idx:]
            if rest:
                runs.append({"text": rest, "sub": False, "sup": False})
            return runs

        def add_formatted_runs(
            p,
            text: str,
            base_size=Pt(10.5),
            base_color=RGBColor(*SLATE_DARK),
            is_bold=False,
        ):
            tokens = re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|\$[^$]+\$)", text)
            for token in tokens:
                if not token:
                    continue
                if token.startswith("**") and token.endswith("**"):
                    r = p.add_run(token[2:-2])
                    set_run_font(r, size=base_size, color=base_color, bold=True)
                elif token.startswith("*") and token.endswith("*"):
                    r = p.add_run(token[1:-1])
                    set_run_font(r, size=base_size, color=base_color, italic=True)
                elif token.startswith("`") and token.endswith("`"):
                    r = p.add_run(token[1:-1])
                    set_run_font(
                        r,
                        font_name="Consolas",
                        east_asia="Consolas",
                        size=Pt(base_size.pt * 0.92),
                        color=RGBColor(30, 41, 59),
                        bold=True,
                    )
                elif token.startswith("$") and token.endswith("$"):
                    math_runs = parse_math_tokens(token)
                    for mr in math_runs:
                        r = p.add_run(mr["text"])
                        set_run_font(
                            r,
                            font_name="Cambria Math",
                            east_asia="Microsoft YaHei",
                            size=base_size,
                            color=base_color,
                            italic=mr.get("math_var", False),
                            subscript=mr.get("sub", False),
                            superscript=mr.get("sup", False),
                        )
                else:
                    r = p.add_run(token)
                    set_run_font(r, size=base_size, color=base_color, bold=is_bold)

        # Parse lines
        lines = markdown_text.splitlines()
        i = 0
        n = len(lines)

        while i < n:
            line = lines[i].rstrip()
            if not line:
                i += 1
                continue

            # Heading 1
            if line.startswith("# "):
                h = doc.add_paragraph()
                h.paragraph_format.space_before = Pt(16)
                h.paragraph_format.space_after = Pt(6)
                r = h.add_run(line[2:].strip())
                set_run_font(
                    r,
                    font_name="Calibri",
                    east_asia="Microsoft YaHei",
                    size=Pt(18),
                    color=RGBColor(*NAVY_PRIMARY),
                    bold=True,
                )
                i += 1
                continue

            # Heading 2
            if line.startswith("## "):
                h = doc.add_paragraph()
                h.paragraph_format.space_before = Pt(14)
                h.paragraph_format.space_after = Pt(4)
                r = h.add_run(line[3:].strip())
                set_run_font(
                    r,
                    font_name="Calibri",
                    east_asia="Microsoft YaHei",
                    size=Pt(14),
                    color=RGBColor(*NAVY_PRIMARY),
                    bold=True,
                )
                i += 1
                continue

            # Heading 3
            if line.startswith("### "):
                h = doc.add_paragraph()
                h.paragraph_format.space_before = Pt(10)
                h.paragraph_format.space_after = Pt(3)
                r = h.add_run(line[4:].strip())
                set_run_font(
                    r,
                    font_name="Calibri",
                    east_asia="Microsoft YaHei",
                    size=Pt(12),
                    color=RGBColor(40, 70, 150),
                    bold=True,
                )
                i += 1
                continue

            # Heading 4
            if line.startswith("#### "):
                h = doc.add_paragraph()
                h.paragraph_format.space_before = Pt(8)
                h.paragraph_format.space_after = Pt(2)
                r = h.add_run(line[5:].strip())
                set_run_font(
                    r,
                    font_name="Calibri",
                    east_asia="Microsoft YaHei",
                    size=Pt(11),
                    color=RGBColor(*SLATE_DARK),
                    bold=True,
                )
                i += 1
                continue

            # Horizontal Rule
            if line.strip() in ("---", "***", "___"):
                p = doc.add_paragraph()
                p.paragraph_format.space_before = Pt(4)
                p.paragraph_format.space_after = Pt(4)
                p_border = parse_xml(
                    rf'<w:pBdr {nsdecls("w")}><w:bottom w:val="single" w:sz="6" w:space="1" w:color="{BORDER_HEX}"/></w:pBdr>'
                )
                p._p.get_or_add_pPr().append(p_border)
                i += 1
                continue

            # Blockquote / Callout
            if line.startswith("> "):
                callout_text = line[2:].strip()
                p = doc.add_paragraph()
                p.paragraph_format.space_before = Pt(6)
                p.paragraph_format.space_after = Pt(6)
                p.paragraph_format.left_indent = Inches(0.2)
                p.paragraph_format.right_indent = Inches(0.2)

                pPr = p._p.get_or_add_pPr()
                pBdr = parse_xml(
                    rf'<w:pBdr {nsdecls("w")}><w:left w:val="single" w:sz="24" w:space="6" w:color="{CALLOUT_BORDER_HEX}"/></w:pBdr>'
                )
                shd = parse_xml(
                    rf'<w:shd {nsdecls("w")}><w:val w:color="auto" w:fill="{CALLOUT_BG_HEX}"/></w:shd>'
                )
                pPr.append(pBdr)
                pPr.append(shd)

                add_formatted_runs(
                    p, callout_text, base_size=Pt(9.5), base_color=RGBColor(30, 58, 138)
                )
                i += 1
                continue

            # Embedded HTML / Markdown Image with data URI
            img_match = re.search(
                r'<img[^>]+src=["\'](data:image/[^;]+;base64,[^"\']+)["\']', line
            )
            if img_match:
                b64_src = img_match.group(1)
                try:
                    header, data = b64_src.split(",", 1)
                    img_bytes = base64.b64decode(data)
                    img_stream = io.BytesIO(img_bytes)

                    p = doc.add_paragraph()
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    p.paragraph_format.space_before = Pt(8)
                    p.paragraph_format.space_after = Pt(3)
                    p.add_run().add_picture(img_stream, width=Inches(6.2))

                    # Caption if next line has <em>...</em>
                    if i + 1 < n and "<em>" in lines[i + 1]:
                        cap_match = re.search(r"<em>([^<]+)</em>", lines[i + 1])
                        if cap_match:
                            p_cap = doc.add_paragraph()
                            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                            p_cap.paragraph_format.space_after = Pt(8)
                            r_cap = p_cap.add_run(cap_match.group(1))
                            set_run_font(
                                r_cap,
                                size=Pt(8.5),
                                color=RGBColor(*SLATE_MUTED),
                                italic=True,
                            )
                            i += 1
                except Exception:
                    pass
                i += 1
                continue

            # Table detection
            if line.startswith("|") and line.endswith("|"):
                table_lines: List[str] = []
                while (
                    i < n
                    and lines[i].strip().startswith("|")
                    and lines[i].strip().endswith("|")
                ):
                    table_lines.append(lines[i].strip())
                    i += 1

                cls._render_styled_docx_table(
                    doc, table_lines, set_run_font, add_formatted_runs
                )
                continue

            # Bullet item
            if line.startswith("- ") or line.startswith("* "):
                bullet_text = line[2:].strip()
                p = doc.add_paragraph(style="List Bullet")
                p.paragraph_format.space_before = Pt(1)
                p.paragraph_format.space_after = Pt(1)
                add_formatted_runs(p, bullet_text)
                i += 1
                continue

            # Standard paragraph
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.line_spacing = 1.15
            add_formatted_runs(p, line)
            i += 1

        output = io.BytesIO()
        doc.save(output)
        return output.getvalue()

    @classmethod
    def _render_styled_docx_table(
        cls, doc, table_lines: List[str], set_run_font, add_formatted_runs
    ) -> None:
        """Renders high-grade zebra-striped table with Navy header and light gray borders."""
        from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.oxml import parse_xml
        from docx.oxml.ns import nsdecls
        from docx.shared import Pt, RGBColor

        if len(table_lines) < 2:
            return

        header_row = [c.strip() for c in table_lines[0].strip("|").split("|")]
        data_rows = []
        for line in table_lines[1:]:
            cells = [c.strip() for c in line.strip("|").split("|")]
            # Skip separator line (e.g. | :--- | :--- |)
            if all(set(c).issubset({"-", ":", " "}) for c in cells if c):
                continue
            data_rows.append(cells)

        num_cols = len(header_row)
        num_rows = 1 + len(data_rows)
        table = doc.add_table(rows=num_rows, cols=num_cols)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = True

        # Header Row Styling
        hdr_cells = table.rows[0].cells
        for col_idx, text in enumerate(header_row):
            if col_idx < len(hdr_cells):
                cell = hdr_cells[col_idx]
                cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                shd = parse_xml(
                    rf'<w:shd {nsdecls("w")}><w:val w:color="auto" w:fill="{HEADER_BG_HEX}"/></w:shd>'
                )
                cell._tc.get_or_add_tcPr().append(shd)

                p = cell.paragraphs[0]
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(4)
                p.paragraph_format.space_after = Pt(4)
                r = p.add_run(text)
                set_run_font(r, size=Pt(9.0), color=RGBColor(255, 255, 255), bold=True)

        # Data Rows Styling with Zebra Stripes
        for row_idx, r_data in enumerate(data_rows):
            row_cells = table.rows[row_idx + 1].cells
            bg_color = ZEBRA_BG_HEX if (row_idx % 2 == 1) else "FFFFFF"

            for col_idx, text in enumerate(r_data):
                if col_idx < len(row_cells):
                    cell = row_cells[col_idx]
                    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                    if bg_color != "FFFFFF":
                        shd = parse_xml(
                            rf'<w:shd {nsdecls("w")}><w:val w:color="auto" w:fill="{bg_color}"/></w:shd>'
                        )
                        cell._tc.get_or_add_tcPr().append(shd)

                    p = cell.paragraphs[0]
                    p.paragraph_format.space_before = Pt(3)
                    p.paragraph_format.space_after = Pt(3)
                    # Text alignment: left for first col, center for others
                    p.alignment = (
                        WD_ALIGN_PARAGRAPH.LEFT
                        if col_idx == 0
                        else WD_ALIGN_PARAGRAPH.CENTER
                    )
                    add_formatted_runs(
                        p, text, base_size=Pt(8.8), base_color=RGBColor(*SLATE_DARK)
                    )

        # Thin border formatting
        tblPr = table._tbl.tblPr
        borders = parse_xml(
            rf'<w:tblBorders {nsdecls("w")}>'
            rf'<w:top w:val="single" w:sz="4" w:space="0" w:color="{BORDER_HEX}"/>'
            rf'<w:bottom w:val="single" w:sz="6" w:space="0" w:color="{BORDER_HEX}"/>'
            rf'<w:insideH w:val="single" w:sz="4" w:space="0" w:color="{BORDER_HEX}"/>'
            rf'<w:insideV w:val="none"/>'
            rf"</w:tblBorders>"
        )
        tblPr.append(borders)

        # Space after table
        after_p = doc.add_paragraph()
        after_p.paragraph_format.space_before = Pt(2)
        after_p.paragraph_format.space_after = Pt(4)

    # ==========================================================================
    # PDF (.pdf) Exporter
    # ==========================================================================
    @classmethod
    def export_pdf(
        cls,
        markdown_text: str,
        title: str = "TRICYS Engineering Report",
    ) -> bytes:
        """Converts Markdown text into a publication-grade PDF document.

        Uses Headless Chrome or Edge printing for publication-grade layout with
        A4 CSS, mathematical typography, running headers/footers, and booktabs tables.
        Seamlessly falls back to PyMuPDF if no browser is found.
        """

        # 1. Attempt Headless Chrome / Edge vector printing
        chrome_candidates = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        ]
        browser_exe = next(
            (exe for exe in chrome_candidates if os.path.exists(exe)), None
        )

        if browser_exe:
            try:
                import subprocess
                import tempfile

                import markdown

                def latex_to_html(tex):
                    s = tex.strip("$").strip()
                    s = re.sub(
                        r"\\text\{([^}]+)\}", r'<span class="tex-text">\1</span>', s
                    )
                    s = (
                        s.replace(r"\Delta", "&Delta;")
                        .replace(r"\tau", "&tau;")
                        .replace(r"\eta", "&eta;")
                    )
                    s = s.replace(
                        r"\dot{m}", '<span class="dot-m"><i>m&#775;</i></span>'
                    )
                    s = s.replace(r"\%", "%")
                    s = re.sub(r"_\{([^}]+)\}", r"<sub>\1</sub>", s)
                    s = re.sub(r"\^\{([^}]+)\}", r"<sup>\1</sup>", s)
                    s = re.sub(r"_([a-zA-Z0-9])", r"<sub>\1</sub>", s)
                    s = re.sub(r"\^([a-zA-Z0-9])", r"<sup>\1</sup>", s)
                    return f'<span class="math-expr">{s}</span>'

                md_proc = re.sub(
                    r"\$([^\$]+)\$", lambda m: latex_to_html(m.group(0)), markdown_text
                )
                html_body = markdown.markdown(
                    md_proc, extensions=["tables", "fenced_code", "sane_lists"]
                )

                full_html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>{title}</title><style>
@page {{
    size: A4 portrait;
    margin: 20mm 15mm 20mm 15mm;
    @top-right {{
        content: "{title}";
        font-size: 8.5pt;
        color: #64748b;
        font-family: -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    }}
    @bottom-center {{
        content: "— " counter(page) " —";
        font-size: 9pt;
        color: #64748b;
        font-family: -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    }}
}}
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "Helvetica Neue", Arial, sans-serif;
    font-size: 9.5pt;
    line-height: 1.6;
    color: #1e293b;
    background-color: #ffffff;
    margin: 0;
    padding: 0;
}}
h1 {{
    font-size: 18pt;
    font-weight: 700;
    color: #1e3a8a;
    text-align: center;
    margin-top: 10px;
    margin-bottom: 14px;
    padding-bottom: 8px;
    border-bottom: 2px solid #2563eb;
    page-break-after: avoid;
    line-height: 1.35;
}}
h2 {{
    font-size: 13pt;
    font-weight: 700;
    color: #1e3a8a;
    margin-top: 20px;
    margin-bottom: 8px;
    padding-left: 8px;
    border-left: 4px solid #2563eb;
    page-break-after: avoid;
}}
h3 {{
    font-size: 11pt;
    font-weight: 600;
    color: #1e3a8a;
    margin-top: 14px;
    margin-bottom: 6px;
    page-break-after: avoid;
}}
h4 {{
    font-size: 10pt;
    font-weight: 600;
    color: #334155;
    margin-top: 10px;
    margin-bottom: 4px;
    page-break-after: avoid;
}}
p {{
    margin-top: 0;
    margin-bottom: 8px;
    text-align: justify;
}}
.math-expr {{
    font-family: "Cambria Math", "TeX Gyre Termes", "Times New Roman", Times, serif;
    font-size: 1.02em;
    letter-spacing: 0.01em;
}}
.math-expr i {{
    font-style: italic;
    font-family: "Cambria Math", "Times New Roman", Times, serif;
}}
.dot-m {{
    font-style: italic;
    font-family: "Cambria Math", "Times New Roman", Times, serif;
}}
sub, sup {{
    font-size: 0.74em;
    line-height: 0;
    position: relative;
}}
sup {{ top: -0.4em; }}
sub {{ bottom: -0.2em; }}
table {{
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0 14px 0;
    font-size: 8.5pt;
    line-height: 1.45;
    page-break-inside: avoid;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
}}
th, td {{
    border: 1px solid #cbd5e1;
    padding: 6px 8px;
    vertical-align: middle;
}}
th {{
    background-color: #1e3a8a;
    color: #ffffff;
    font-weight: 700;
    text-align: center;
    border-bottom: 2px solid #0f172a;
}}
tr:nth-child(even) {{
    background-color: #f8fafc;
}}
blockquote {{
    margin: 10px 0;
    padding: 8px 12px;
    background-color: #f0f7ff;
    border-left: 4px solid #2563eb;
    border-radius: 4px;
    color: #1e3a8a;
    font-size: 9pt;
    page-break-inside: avoid;
}}
blockquote p {{
    margin-bottom: 4px;
}}
blockquote p:last-child {{
    margin-bottom: 0;
}}
img {{
    max-width: 98%;
    height: auto;
    display: block;
    margin: 0 auto;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.06);
}}
hr {{
    border: none;
    height: 1px;
    background-color: #cbd5e1;
    margin: 16px 0;
}}
</style></head><body>{html_body}</body></html>"""

                with tempfile.NamedTemporaryFile(
                    "w", suffix=".html", delete=False, encoding="utf-8"
                ) as f_html:
                    f_html.write(full_html)
                    tmp_html = f_html.name

                tmp_pdf = tmp_html.replace(".html", ".pdf")
                subprocess.run(
                    [
                        browser_exe,
                        "--headless=new",
                        "--disable-gpu",
                        "--run-all-compositor-stages-before-draw",
                        "--no-pdf-header-footer",
                        f"--print-to-pdf={tmp_pdf}",
                        tmp_html,
                    ],
                    capture_output=True,
                    timeout=30,
                )

                if os.path.exists(tmp_pdf) and os.path.getsize(tmp_pdf) > 0:
                    with open(tmp_pdf, "rb") as f_pdf:
                        pdf_data = f_pdf.read()
                    try:
                        os.remove(tmp_html)
                        os.remove(tmp_pdf)
                    except Exception:
                        pass
                    return pdf_data
            except Exception:
                pass

        # 2. Fallback to PyMuPDF
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz

        doc = fitz.open()

        # Register robust CJK font
        cjk_font_name = "china-s"
        for font_candidate in [
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/simsun.ttc",
        ]:
            if os.path.exists(font_candidate):
                try:
                    doc.insert_font(fontname="cjk_custom", fontfile=font_candidate)
                    cjk_font_name = "cjk_custom"
                    break
                except Exception:
                    pass

        PAGE_WIDTH = 595.0
        PAGE_HEIGHT = 842.0
        MARGIN_X = 50.0
        MARGIN_TOP = 50.0
        MARGIN_BOTTOM = 50.0

        page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
        y = MARGIN_TOP

        def _check_page_break(needed_height: float) -> fitz.Page:
            nonlocal page, y
            if y + needed_height > PAGE_HEIGHT - MARGIN_BOTTOM:
                page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
                y = MARGIN_TOP
            return page

        lines = markdown_text.splitlines()

        for line in lines:
            trimmed = line.strip()
            if not trimmed:
                y += 6
                continue

            # Image check
            img_match = re.search(
                r'<img[^>]+src=["\'](data:image/[^;]+;base64,[^"\']+)["\']', trimmed
            )
            if img_match:
                try:
                    data = img_match.group(1).split(",", 1)[1]
                    img_bytes = base64.b64decode(data)
                    _check_page_break(220)
                    rect = fitz.Rect(MARGIN_X, y, PAGE_WIDTH - MARGIN_X, y + 210)
                    page.insert_image(rect, stream=img_bytes)
                    y += 220
                except Exception:
                    pass
                continue

            # Heading 1
            if trimmed.startswith("# "):
                _check_page_break(32)
                page.insert_text(
                    (MARGIN_X, y + 18),
                    trimmed[2:].strip(),
                    fontname=cjk_font_name,
                    fontsize=16,
                    color=(0.12, 0.23, 0.54),
                )
                y += 26
                page.draw_line(
                    (MARGIN_X, y),
                    (PAGE_WIDTH - MARGIN_X, y),
                    color=(0.7, 0.8, 0.9),
                    width=1.0,
                )
                y += 10
                continue

            # Heading 2
            if trimmed.startswith("## "):
                _check_page_break(24)
                page.insert_text(
                    (MARGIN_X, y + 14),
                    trimmed[3:].strip(),
                    fontname=cjk_font_name,
                    fontsize=13,
                    color=(0.12, 0.23, 0.54),
                )
                y += 22
                continue

            # Heading 3
            if trimmed.startswith("### "):
                _check_page_break(20)
                page.insert_text(
                    (MARGIN_X, y + 12),
                    trimmed[4:].strip(),
                    fontname=cjk_font_name,
                    fontsize=11.5,
                    color=(0.15, 0.3, 0.5),
                )
                y += 18
                continue

            # Horizontal Rule
            if trimmed in ("---", "***", "___"):
                _check_page_break(12)
                page.draw_line(
                    (MARGIN_X, y + 4),
                    (PAGE_WIDTH - MARGIN_X, y + 4),
                    color=(0.8, 0.85, 0.9),
                    width=0.8,
                )
                y += 12
                continue

            # Clean markdown markdown tokens for PDF plain text
            clean_text = trimmed
            clean_text = re.sub(r"(\*\*|\*|`|\$)", "", clean_text)
            if clean_text.startswith("> "):
                clean_text = clean_text[2:]

            _check_page_break(16)
            page.insert_text(
                (MARGIN_X, y + 10),
                clean_text[:95],
                fontname=cjk_font_name,
                fontsize=9.5,
                color=(0.15, 0.18, 0.22),
            )
            y += 14

        pdf_bytes = doc.write()
        doc.close()
        return pdf_bytes
