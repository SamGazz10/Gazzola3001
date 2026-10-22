"""Render a Markdown report to an A4 PDF (pandoc -> HTML with print CSS -> Chromium).

Usage: python render.py input.md output.pdf
Relative image paths are resolved against the Markdown file's folder.
Prints the page count, and the page on which a heading containing "References" starts.
"""
import subprocess
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

CSS = """
@page { size: A4; margin: 20mm 18mm 20mm 18mm; }
html { font-family: "Liberation Serif", "Times New Roman", serif; font-size: 11pt; line-height: 1.3; color: #111; }
body { margin: 0; }
h1 { font-size: 17pt; margin: 0 0 4pt; line-height: 1.2; }
h2 { font-size: 13pt; margin: 14pt 0 4pt; break-after: avoid; }
h3 { font-size: 11.5pt; margin: 10pt 0 3pt; break-after: avoid; }
p { margin: 0 0 6pt; text-align: justify; hyphens: auto; }
ul, ol { margin: 0 0 6pt; padding-left: 18pt; }
li { margin: 0 0 2pt; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0 8pt; font-size: 8.5pt; line-height: 1.2; break-inside: auto; }
th, td { border-top: 0.5pt solid #999; border-bottom: 0.5pt solid #999; padding: 1.5pt 3pt; vertical-align: top; text-align: left; }
td code, th code { font-size: 7.5pt; }
th { background: #eee; }
tr { break-inside: avoid; }
img { max-width: 100%; display: block; margin: 4pt auto; }
figure { margin: 6pt 0 8pt; break-inside: avoid; }
figcaption { font-size: 9pt; color: #333; text-align: center; margin-top: 2pt; }
code { font-family: "Liberation Mono", monospace; font-size: 9pt; }
pre { font-size: 8.5pt; line-height: 1.2; background: #f6f6f6; padding: 4pt 6pt; break-inside: avoid; white-space: pre-wrap; }
hr { border: 0; border-top: 0.5pt solid #bbb; margin: 8pt 0; }
a { color: #1a4f9c; text-decoration: none; }
.pagebreak { break-before: page; }
h2#references, h2[id^='appendix-a'] { break-before: page; }
p:has(> img) { margin: 4pt 0 0; break-after: avoid; }
p:has(> img) + p { font-size: 9pt; text-align: center; margin: 0 0 8pt; }
p:has(> em:only-child):has(+ table) { font-size: 9pt; margin: 6pt 0 2pt; }
math { font-size: 105%; }
"""


def main(src, out):
    src, out = Path(src).resolve(), Path(out).resolve()
    html = subprocess.run(
        ["pandoc", str(src), "--from", "gfm+tex_math_dollars+raw_html", "--to", "html5", "--standalone",
         "--mathml", "--metadata", "title= ", "--resource-path", str(src.parent)],
        check=True, capture_output=True, text=True).stdout
    html = html.replace("</head>", f"<style>{CSS}</style><base href=\"file://{src.parent}/\"></head>")
    html = html.replace('<header id="title-block-header">\n<h1 class="title"> </h1>\n</header>', "")
    tmp = out.with_suffix(".html")
    tmp.write_text(html)
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/opt/pw-browsers/chromium")
        page = browser.new_page()
        page.goto(f"file://{tmp}")
        page.wait_for_load_state("networkidle")
        page.pdf(path=str(out), format="A4", print_background=True,
                 margin={"top": "18mm", "bottom": "18mm", "left": "18mm", "right": "18mm"},
                 display_header_footer=True, header_template="<span></span>",
                 footer_template='<div style="font-size:8pt;width:100%;text-align:center;color:#555">'
                                 '<span class="pageNumber"></span></div>')
        browser.close()
    info = subprocess.run(["pdfinfo", str(out)], capture_output=True, text=True).stdout
    pages = int(next(l for l in info.splitlines() if l.startswith("Pages:")).split()[1])
    print(f"{out.name}: {pages} pages")
    for n in range(1, pages + 1):
        text = subprocess.run(["pdftotext", "-f", str(n), "-l", str(n), str(out), "-"],
                              capture_output=True, text=True).stdout
        if any(line.strip() in ("References", "Appendices") or line.strip().startswith(("References", "Appendix A"))
               for line in text.splitlines()):
            print(f"  'References'/'Appendix' heading found on page {n}")
            break


if __name__ == "__main__":
    main(*sys.argv[1:3])
