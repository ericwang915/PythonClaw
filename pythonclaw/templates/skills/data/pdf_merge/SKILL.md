---
name: pdf_merge
description: "Merge multiple PDF files into a single PDF. Use when: user asks to combine, join, or concatenate PDFs. Supports page selection and ordering. NOT for: splitting PDFs, editing content, or converting formats."
dependencies: PyPDF2
metadata:
  emoji: "📎"
---

# PDF Merge Skill

Merge multiple PDF files into one combined document using PyPDF2.

## When to Use

✅ **USE this skill when:**
- "Merge these PDFs together"
- "Combine report1.pdf and report2.pdf"
- "Join all PDFs in this folder"
- "Append this PDF to the end of another"
- User wants to create a single PDF from multiple source PDFs

## When NOT to Use

❌ **DON'T use this skill when:**
- Splitting a single PDF → use `pdf_split`
- Creating a PDF from text → use `pdf_writer`
- Extracting text from PDF → use `pdf_reader`
- Converting PDF to images → use `pdf_convert`

## Usage/Commands

```bash
python {skill_path}/merge_pdf.py OUTPUT_PATH INPUT1 INPUT2 [INPUT3 ...] [options]
```

Options:
- `--pages "1:1-3,2:5-10"` — select specific pages per input file (1-indexed, file_index:page_range)
- `--dir PATH` — merge all PDFs in a directory (alphabetical order)
- `--reverse` — reverse the input order
- `--format json` — output result as JSON

### Examples

- "Merge a.pdf and b.pdf" → `python {skill_path}/merge_pdf.py combined.pdf a.pdf b.pdf`
- "Merge all PDFs in reports/" → `python {skill_path}/merge_pdf.py combined.pdf --dir reports/`
- "Combine only pages 1-3 of first and 5-10 of second" → `python {skill_path}/merge_pdf.py out.pdf a.pdf b.pdf --pages "1:1-3,2:5-10"`

## Notes

- Install dependency: `pip install PyPDF2`
- Input files must be valid PDFs
- Preserves bookmarks and links when possible
