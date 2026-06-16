# filename: scripts/generate_report_pdf.py
# purpose:  Convert PROJECT_REPORT.md to PROJECT_REPORT.pdf
# version:  1.0

import os
import sys
import markdown2
from xhtml2pdf import pisa

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD_FILE  = os.path.join(ROOT, "PROJECT_REPORT.md")
PDF_FILE = os.path.join(ROOT, "PROJECT_REPORT.pdf")

CSS = """
@page {
    size: A4;
    margin: 2cm 2.2cm 2cm 2.2cm;
}
body {
    font-family: Arial, Helvetica, sans-serif;
    font-size: 10.5pt;
    color: #2c3e50;
    line-height: 1.55;
}
h1 {
    font-size: 18pt;
    color: #1b3a5c;
    border-bottom: 2px solid #1b3a5c;
    padding-bottom: 6px;
    margin-top: 22px;
}
h2 {
    font-size: 14pt;
    color: #1b3a5c;
    border-bottom: 1px solid #aed6f1;
    margin-top: 18px;
}
h3 {
    font-size: 11.5pt;
    color: #177e89;
    margin-top: 14px;
}
h4 {
    font-size: 10.5pt;
    color: #2c3e50;
    font-style: italic;
}
table {
    width: 100%;
    border-collapse: collapse;
    margin: 10px 0;
    font-size: 9.5pt;
}
th {
    background-color: #1b3a5c;
    color: white;
    padding: 5px 7px;
    text-align: left;
}
td {
    border: 1px solid #d5d8dc;
    padding: 4px 7px;
}
tr:nth-child(even) td {
    background-color: #f4f6f8;
}
code {
    background-color: #f0f3f4;
    padding: 1px 4px;
    font-family: Courier New, monospace;
    font-size: 9pt;
    border-radius: 2px;
}
pre {
    background-color: #f0f3f4;
    padding: 8px 10px;
    font-family: Courier New, monospace;
    font-size: 8.5pt;
    border-left: 3px solid #1b3a5c;
    overflow: hidden;
}
blockquote {
    border-left: 4px solid #177e89;
    margin-left: 0;
    padding-left: 12px;
    color: #555;
    font-style: italic;
}
ul, ol {
    margin: 6px 0;
    padding-left: 20px;
}
li {
    margin: 2px 0;
}
a {
    color: #177e89;
}
"""

def convert():
    with open(MD_FILE, encoding="utf-8") as f:
        md_text = f.read()

    # markdown2 with tables + fenced code blocks
    html_body = markdown2.markdown(
        md_text,
        extras=["tables", "fenced-code-blocks", "strike", "break-on-newline"]
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>{CSS}</style>
</head>
<body>
{html_body}
</body>
</html>"""

    with open(PDF_FILE, "wb") as pdf_out:
        result = pisa.CreatePDF(html, dest=pdf_out)

    if result.err:
        print(f"ERROR: PDF conversion had {result.err} error(s)")
        sys.exit(1)
    else:
        size_kb = os.path.getsize(PDF_FILE) // 1024
        print(f"Saved -> {PDF_FILE}  ({size_kb} KB)")


if __name__ == "__main__":
    convert()
