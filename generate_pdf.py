"""Generate a well-formatted PDF from approach.md using fpdf2."""

import re
from fpdf import FPDF
from fpdf.enums import XPos, YPos


def sanitize(text):
    """Replace unicode chars that Helvetica can't render."""
    return (text
        .replace("\u2014", "-").replace("\u2013", "-")
        .replace("\u2018", "'").replace("\u2019", "'")
        .replace("\u201c", '"').replace("\u201d", '"')
        .replace("\u2022", "-").replace("\u2026", "...")
        .replace("`", ""))


class ApproachPDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(150, 150, 150)
            self.cell(0, 5, "SHL Assessment Recommendation Engine - Approach Document",
                      align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def write_rich(self, line_h, text, base_size=9, base_color=(50, 50, 50)):
        """Write text with inline **bold** support."""
        text = sanitize(text)
        parts = re.split(r"(\*\*.+?\*\*)", text)
        for part in parts:
            if part.startswith("**") and part.endswith("**"):
                self.set_font("Helvetica", "B", base_size)
                self.set_text_color(*base_color)
                self.write(line_h, part[2:-2])
            else:
                self.set_font("Helvetica", "", base_size)
                self.set_text_color(*base_color)
                self.write(line_h, part)

    def add_table(self, rows):
        """Render a markdown table with auto-sized columns."""
        headers = [sanitize(c.strip()) for c in rows[0].strip("|").split("|")]
        data = []
        for row in rows[2:]:  # skip separator
            data.append([sanitize(c.strip()) for c in row.strip("|").split("|")])

        n = len(headers)
        page_w = self.w - self.l_margin - self.r_margin

        # Measure max content width per column
        max_widths = []
        for col in range(n):
            self.set_font("Helvetica", "B", 8)
            hw = self.get_string_width(headers[col]) + 4
            max_w = hw
            self.set_font("Helvetica", "", 7.5)
            for row in data:
                if col < len(row):
                    cw = self.get_string_width(row[col]) + 4
                    max_w = max(max_w, cw)
            max_widths.append(max_w)

        total = sum(max_widths)
        if total > page_w:
            # Scale proportionally
            col_widths = [w * page_w / total for w in max_widths]
        else:
            # Distribute extra space proportionally
            col_widths = [w * page_w / total for w in max_widths]

        # Header row
        self.set_font("Helvetica", "B", 8)
        self.set_fill_color(30, 30, 60)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, fill=True, align="C")
        self.ln()

        # Data rows
        self.set_font("Helvetica", "", 7.5)
        for ri, row in enumerate(data):
            fill = ri % 2 == 0
            if fill:
                self.set_fill_color(245, 247, 250)
            self.set_text_color(50, 50, 50)
            for i in range(n):
                cell_text = row[i] if i < len(row) else ""
                self.cell(col_widths[i], 6, cell_text, border=1, fill=fill,
                          align="C" if i > 0 else "L")
            self.ln()
        self.ln(2)


def generate():
    pdf = ApproachPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    with open("output/approach.md", "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    table_buffer = []
    in_table = False

    while i < len(lines):
        line = lines[i].rstrip("\n")

        # --- Table detection ---
        if line.startswith("|") and not in_table:
            in_table = True
            table_buffer = [line]
            i += 1
            continue
        elif in_table:
            if line.startswith("|"):
                table_buffer.append(line)
                i += 1
                continue
            else:
                pdf.add_table(table_buffer)
                table_buffer = []
                in_table = False
                # fall through

        # Horizontal rule
        if line.strip() == "---":
            pdf.ln(1)
            i += 1
            continue

        # Main title (# )
        if line.startswith("# ") and not line.startswith("## "):
            pdf.set_font("Helvetica", "B", 20)
            pdf.set_text_color(26, 26, 46)
            pdf.cell(0, 12, sanitize(line[2:]),
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
            i += 1
            continue

        # Subtitle (## )
        if line.startswith("## "):
            pdf.set_font("Helvetica", "I", 11)
            pdf.set_text_color(100, 100, 120)
            pdf.cell(0, 7, sanitize(line[3:]),
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
            pdf.ln(2)
            i += 1
            continue

        # Section header (### )
        if line.startswith("### "):
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(15, 52, 96)
            pdf.cell(0, 7, sanitize(line[4:]),
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_draw_color(15, 52, 96)
            pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
            pdf.ln(2)
            i += 1
            continue

        # Standalone bold line (**...** on its own)
        bold_match = re.match(r"^\*\*(.+?)\*\*$", line.strip())
        if bold_match:
            pdf.ln(1)
            pdf.set_font("Helvetica", "B", 9.5)
            pdf.set_text_color(30, 30, 60)
            pdf.cell(0, 6, sanitize(bold_match.group(1)),
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(1)
            i += 1
            continue

        # Sub-bullet (  - ...)
        sub_bullet = re.match(r"^  - (.+)$", line)
        if sub_bullet:
            pdf.set_x(pdf.l_margin + 12)
            pdf.set_font("Helvetica", "", 8.5)
            pdf.set_text_color(80, 80, 80)
            pdf.cell(4, 5, "-")
            pdf.write_rich(5, sub_bullet.group(1), base_size=8.5, base_color=(80, 80, 80))
            pdf.ln(5)
            i += 1
            continue

        # Bullet (- ...)
        bullet_match = re.match(r"^- (.+)$", line.strip())
        if bullet_match:
            pdf.set_x(pdf.l_margin + 5)
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(50, 50, 50)
            pdf.cell(4, 5, "-")
            pdf.write_rich(5, bullet_match.group(1))
            pdf.ln(5)
            i += 1
            continue

        # Empty line
        if line.strip() == "":
            pdf.ln(1)
            i += 1
            continue

        # Regular paragraph with inline bold
        text = line.strip()
        if text:
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(50, 50, 50)
            pdf.write_rich(5, text)
            pdf.ln(5)

        i += 1

    # Flush remaining table
    if in_table and table_buffer:
        pdf.add_table(table_buffer)

    pdf.output("output/approach.pdf")
    print(f"PDF generated: {pdf.page_no()} pages")


if __name__ == "__main__":
    generate()
