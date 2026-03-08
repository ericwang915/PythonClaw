---
name: pdf_writer
description: "Create PDF documents from text, Markdown, or HTML content. Use when: user asks to create, generate, or write a PDF. Supports headings, paragraphs, bullet lists, tables, and basic formatting. NOT for: editing existing PDFs, filling forms, or adding images/charts."
dependencies: fpdf2
metadata:
  emoji: "📝"
---

# PDF Writer Skill

Create PDF documents from plain text, Markdown, or structured content using fpdf2.

## When to Use

✅ **USE this skill when:**
- "Create a PDF with this text"
- "Write a report as PDF"
- "Convert this markdown to PDF"
- "Generate a PDF document"
- User wants to produce a new PDF file from scratch

## When NOT to Use

❌ **DON'T use this skill when:**
- Editing an existing PDF → use PDF manipulation tools
- Filling PDF forms → use form-filling libraries
- Complex layouts with images/charts → use LaTeX or specialized tools
- Converting PDF to other formats → use `pdf_convert`

## Usage/Commands

```bash
python {skill_path}/write_pdf.py OUTPUT_PATH [options]
```

Options:
- `--text "content"` — plain text body
- `--file INPUT_FILE` — read body from a text/markdown file
- `--title "Title"` — document title (shown on first page)
- `--author "Name"` — PDF metadata author
- `--font-size 12` — base font size (default 12)
- `--landscape` — landscape orientation (default portrait)
- `--format json` — output result as JSON

Content passed via `--text` or `--file` supports lightweight Markdown:
- `# Heading 1`, `## Heading 2`, `### Heading 3`
- `- bullet items`
- `**bold**` and `*italic*`
- Blank lines for paragraph breaks

### Examples

- "Create a PDF with this text" → `python {skill_path}/write_pdf.py output.pdf --text "Hello World"`
- "Write report.md as PDF" → `python {skill_path}/write_pdf.py report.pdf --file report.md --title "Report"`
- "Generate a landscape PDF" → `python {skill_path}/write_pdf.py wide.pdf --text "..." --landscape`

## Notes

- Install dependency: `pip install fpdf2`
- Output is UTF-8 compatible (supports CJK characters via built-in fonts)
- Maximum recommended body size: ~100 KB of text
