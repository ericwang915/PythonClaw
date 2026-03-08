---
name: pdf_convert
description: "Convert between PDF and image formats. Use when: user asks to convert PDF pages to images (PNG/JPEG) or combine images into a PDF. NOT for: text extraction (use pdf_reader), PDF editing, or document format conversion (DOCX/HTML)."
dependencies: PyPDF2, Pillow
metadata:
  emoji: "🔄"
---

# PDF Convert Skill

Convert PDF pages to images or combine images into a PDF.

## When to Use

✅ **USE this skill when:**
- "Convert this PDF to images"
- "Save page 3 as a PNG"
- "Make a PDF from these images"
- "Export PDF pages as JPEG"
- User wants to convert between PDF and image formats

## When NOT to Use

❌ **DON'T use this skill when:**
- Extracting text from PDF → use `pdf_reader`
- Creating PDF from text/markdown → use `pdf_writer`
- Converting to DOCX, HTML, etc. → use specialized converters
- Merging PDFs → use `pdf_merge`

## Usage/Commands

### Images to PDF

```bash
python {skill_path}/convert_pdf.py to-pdf OUTPUT_PDF IMAGE1 IMAGE2 [IMAGE3 ...] [options]
```

Options:
- `--dir PATH` — use all images from a directory
- `--fit` — fit images to page (default: actual size with margins)
- `--page-size A4|letter` — page size (default: A4)

### PDF to Images

```bash
python {skill_path}/convert_pdf.py to-images INPUT_PDF [options]
```

Options:
- `--pages "1-5"` — specific pages (default: all)
- `--output DIR` — output directory (default: same as input)
- `--img-format png|jpeg` — image format (default: png)
- `--dpi 150` — resolution (default: 150)

### Examples

- "Convert PDF to PNGs" → `python {skill_path}/convert_pdf.py to-images doc.pdf --output ./images/`
- "Make PDF from images" → `python {skill_path}/convert_pdf.py to-pdf album.pdf img1.png img2.jpg`
- "Page 3 as JPEG" → `python {skill_path}/convert_pdf.py to-images doc.pdf --pages 3 --img-format jpeg`

## Notes

- Install dependencies: `pip install PyPDF2 Pillow`
- `to-images` requires a PDF rendering library; falls back to a placeholder if pdf2image/poppler is unavailable
- Supported image inputs: PNG, JPEG, BMP, TIFF, GIF
