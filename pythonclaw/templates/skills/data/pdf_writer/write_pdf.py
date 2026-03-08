#!/usr/bin/env python3
"""Create PDF documents from text or Markdown content."""

import argparse
import json
import re
import sys

try:
    from fpdf import FPDF
except ImportError:
    print("Error: fpdf2 not installed.  Run: pip install fpdf2", file=sys.stderr)
    sys.exit(1)


class MarkdownPDF(FPDF):
    """Simple FPDF subclass that renders lightweight Markdown."""

    def __init__(self, title: str = "", author: str = "", font_size: int = 12, **kw):
        super().__init__(**kw)
        self._base_size = font_size
        self.set_auto_page_break(auto=True, margin=20)
        self.add_page()
        self.set_font("Helvetica", size=font_size)
        if title:
            self.set_font("Helvetica", "B", font_size + 6)
            self.cell(0, 12, title, new_x="LMARGIN", new_y="NEXT", align="C")
            self.ln(4)
            self.set_font("Helvetica", size=font_size)
        if author:
            self.set_author(author)
        if title:
            self.set_title(title)

    def render_markdown(self, text: str) -> None:
        for line in text.split("\n"):
            stripped = line.strip()

            if not stripped:
                self.ln(4)
                continue

            # Headings
            if stripped.startswith("### "):
                self.set_font("Helvetica", "B", self._base_size + 1)
                self.cell(0, 8, stripped[4:], new_x="LMARGIN", new_y="NEXT")
                self.set_font("Helvetica", size=self._base_size)
                continue
            if stripped.startswith("## "):
                self.set_font("Helvetica", "B", self._base_size + 2)
                self.cell(0, 9, stripped[3:], new_x="LMARGIN", new_y="NEXT")
                self.set_font("Helvetica", size=self._base_size)
                continue
            if stripped.startswith("# "):
                self.set_font("Helvetica", "B", self._base_size + 4)
                self.cell(0, 10, stripped[2:], new_x="LMARGIN", new_y="NEXT")
                self.set_font("Helvetica", size=self._base_size)
                continue

            # Bullet
            if stripped.startswith("- ") or stripped.startswith("* "):
                self.cell(8)
                self._render_inline("\u2022 " + stripped[2:])
                continue

            self._render_inline(stripped)

    def _render_inline(self, text: str) -> None:
        """Render a line with **bold** and *italic* spans."""
        parts = re.split(r"(\*\*.*?\*\*|\*.*?\*)", text)
        for part in parts:
            if part.startswith("**") and part.endswith("**"):
                self.set_font("Helvetica", "B", self._base_size)
                self.write(6, part[2:-2])
                self.set_font("Helvetica", size=self._base_size)
            elif part.startswith("*") and part.endswith("*"):
                self.set_font("Helvetica", "I", self._base_size)
                self.write(6, part[1:-1])
                self.set_font("Helvetica", size=self._base_size)
            else:
                self.write(6, part)
        self.ln(6)


def main():
    parser = argparse.ArgumentParser(description="Create a PDF document.")
    parser.add_argument("output", help="Output PDF path")
    parser.add_argument("--text", default=None, help="Body text (plain or markdown)")
    parser.add_argument("--file", default=None, help="Read body from this text/md file")
    parser.add_argument("--title", default="", help="Document title")
    parser.add_argument("--author", default="", help="Document author metadata")
    parser.add_argument("--font-size", type=int, default=12, help="Base font size")
    parser.add_argument("--landscape", action="store_true", help="Landscape orientation")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    args = parser.parse_args()

    body = args.text or ""
    if args.file:
        try:
            with open(args.file, encoding="utf-8") as f:
                body = f.read()
        except Exception as exc:
            print(f"Error reading input file: {exc}", file=sys.stderr)
            sys.exit(1)

    if not body.strip():
        print("Error: no content provided. Use --text or --file.", file=sys.stderr)
        sys.exit(1)

    orientation = "L" if args.landscape else "P"
    pdf = MarkdownPDF(
        title=args.title,
        author=args.author,
        font_size=args.font_size,
        orientation=orientation,
    )
    pdf.render_markdown(body)

    try:
        pdf.output(args.output)
    except Exception as exc:
        print(f"Error writing PDF: {exc}", file=sys.stderr)
        sys.exit(1)

    result = {
        "output": args.output,
        "pages": pdf.page,
        "title": args.title or "(untitled)",
    }

    if args.format == "json":
        print(json.dumps(result, indent=2))
    else:
        print(f"PDF created: {args.output} ({pdf.page} pages)")


if __name__ == "__main__":
    main()
