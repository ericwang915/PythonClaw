---
name: pdf_split
description: "Split a PDF into separate files by page ranges. Use when: user asks to split, extract pages from, or break apart a PDF. Supports extracting single pages, ranges, or every-N-pages splitting. NOT for: merging PDFs, reading text, or converting formats."
dependencies: PyPDF2
metadata:
  emoji: "✂️"
---

# PDF Split Skill

Split a PDF file into multiple smaller PDFs by page ranges.

## When to Use

✅ **USE this skill when:**
- "Split this PDF into individual pages"
- "Extract pages 5-10 from report.pdf"
- "Break this PDF into 3 parts"
- "Get page 7 as a separate PDF"
- User wants to extract or separate pages from a PDF

## When NOT to Use

❌ **DON'T use this skill when:**
- Merging multiple PDFs → use `pdf_merge`
- Reading/extracting text → use `pdf_reader`
- Creating new PDFs → use `pdf_writer`
- Converting to images → use `pdf_convert`

## Usage/Commands

```bash
python {skill_path}/split_pdf.py INPUT_PDF [options]
```

Options:
- `--pages "1-5"` — extract a specific page range to a single output file
- `--each` — split into one PDF per page
- `--every N` — split into chunks of N pages each
- `--output DIR` — output directory (default: same as input)
- `--prefix NAME` — filename prefix for output files (default: input filename)
- `--format json` — output result as JSON

### Examples

- "Extract pages 5-10" → `python {skill_path}/split_pdf.py doc.pdf --pages 5-10`
- "Split into individual pages" → `python {skill_path}/split_pdf.py doc.pdf --each`
- "Split into chunks of 5 pages" → `python {skill_path}/split_pdf.py doc.pdf --every 5 --output ./parts/`
- "Get page 3" → `python {skill_path}/split_pdf.py doc.pdf --pages 3`

## Notes

- Install dependency: `pip install PyPDF2`
- Page numbers are 1-indexed
- Output files are named `{prefix}_p{start}-{end}.pdf`
