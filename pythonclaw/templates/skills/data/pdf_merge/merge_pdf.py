#!/usr/bin/env python3
"""Merge multiple PDF files into one."""

import argparse
import glob
import json
import os
import sys

try:
    from PyPDF2 import PdfReader, PdfWriter
except ImportError:
    print("Error: PyPDF2 not installed.  Run: pip install PyPDF2", file=sys.stderr)
    sys.exit(1)


def parse_page_selections(spec: str) -> dict[int, list[int]]:
    """Parse 'file_idx:page_range' selections, e.g. '1:1-3,2:5-10'."""
    selections: dict[int, list[int]] = {}
    for part in spec.split(","):
        part = part.strip()
        if ":" not in part:
            continue
        file_idx_str, page_spec = part.split(":", 1)
        file_idx = int(file_idx_str)
        pages: list[int] = []
        for seg in page_spec.split("-"):
            seg = seg.strip()
        if "-" in page_spec:
            start, end = page_spec.split("-", 1)
            pages = list(range(int(start) - 1, int(end)))
        else:
            pages = [int(page_spec) - 1]
        selections[file_idx] = pages
    return selections


def merge_pdfs(
    inputs: list[str],
    output: str,
    page_selections: dict[int, list[int]] | None = None,
) -> dict:
    writer = PdfWriter()
    total_pages = 0

    for idx, path in enumerate(inputs, 1):
        reader = PdfReader(path)
        num_pages = len(reader.pages)

        if page_selections and idx in page_selections:
            pages = [p for p in page_selections[idx] if 0 <= p < num_pages]
        else:
            pages = list(range(num_pages))

        for page_idx in pages:
            writer.add_page(reader.pages[page_idx])
            total_pages += 1

    with open(output, "wb") as f:
        writer.write(f)

    return {
        "output": output,
        "inputFiles": len(inputs),
        "totalPages": total_pages,
        "sizeBytes": os.path.getsize(output),
    }


def main():
    parser = argparse.ArgumentParser(description="Merge multiple PDFs into one.")
    parser.add_argument("output", help="Output PDF path")
    parser.add_argument("inputs", nargs="*", help="Input PDF files")
    parser.add_argument("--dir", default=None, help="Merge all PDFs from a directory")
    parser.add_argument("--pages", default=None, help="Page selections per file (e.g. '1:1-3,2:5-10')")
    parser.add_argument("--reverse", action="store_true", help="Reverse input order")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    args = parser.parse_args()

    inputs = list(args.inputs)

    if args.dir:
        dir_pdfs = sorted(glob.glob(os.path.join(args.dir, "*.pdf")))
        inputs.extend(dir_pdfs)

    if not inputs:
        print("Error: no input PDFs provided.", file=sys.stderr)
        sys.exit(1)

    for path in inputs:
        if not os.path.isfile(path):
            print(f"Error: file not found: {path}", file=sys.stderr)
            sys.exit(1)

    if args.reverse:
        inputs.reverse()

    page_selections = None
    if args.pages:
        page_selections = parse_page_selections(args.pages)

    try:
        result = merge_pdfs(inputs, args.output, page_selections)
    except Exception as exc:
        print(f"Error merging PDFs: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.format == "json":
        print(json.dumps(result, indent=2))
    else:
        print(f"Merged {result['inputFiles']} files → {args.output} ({result['totalPages']} pages, {result['sizeBytes']:,} bytes)")


if __name__ == "__main__":
    main()
