import markdown2
from weasyprint import HTML
import os

DOCS_DIR = "/home/claude/eko-prototype/sample-docs"

# Convert these 6 to real PDFs to prove the pipeline handles actual PDF extraction
TO_CONVERT = [
    "POL-COC-001-code-of-conduct.md",
    "POL-PAY-014-payroll-compensation.md",
    "ONB-NJ-002-day1-week1-checklist.md",
    "POL-SEC-003-it-security-acceptable-use.md",
    "FAQ-GEN-001-general-faqs.md",
    "TRN-MAND-009-mandatory-trainings.md",
]

CSS = """
<style>
  body { font-family: 'DejaVu Sans', sans-serif; font-size: 11pt; color: #1a1a1a; line-height: 1.5; margin: 40px; }
  h1 { font-size: 20pt; color: #0F1B2D; border-bottom: 2px solid #C9A24B; padding-bottom: 8px; }
  h2 { font-size: 14pt; color: #0F1B2D; margin-top: 24px; }
  h3 { font-size: 12pt; color: #333; }
  p { margin: 8px 0; }
  ol, ul { margin: 8px 0; padding-left: 24px; }
  li { margin: 4px 0; }
  .meta { background: #f2f2f2; padding: 12px 16px; border-left: 3px solid #C9A24B; font-size: 9.5pt; color: #444; margin-bottom: 20px; }
</style>
"""

for fname in TO_CONVERT:
    path = os.path.join(DOCS_DIR, fname)
    with open(path, "r") as f:
        md_text = f.read()

    # split off the metadata header block (first 6 lines after title) into a styled box
    lines = md_text.split("\n")
    title = lines[0]
    meta_lines = []
    body_start = 1
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "":
            continue
        if line.startswith("##"):
            body_start = i
            break
        meta_lines.append(line)
        body_start = i + 1

    meta_html = "<div class='meta'>" + "<br>".join(meta_lines) + "</div>"
    body_md = "\n".join(lines[body_start:])
    body_html = markdown2.markdown(body_md, extras=["tables"])
    title_html = markdown2.markdown(title)

    full_html = f"<html><head>{CSS}</head><body>{title_html}{meta_html}{body_html}</body></html>"

    out_path = path.replace(".md", ".pdf")
    HTML(string=full_html).write_pdf(out_path)
    print(f"Created {out_path}")

print("Done.")
