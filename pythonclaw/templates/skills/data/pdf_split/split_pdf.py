#!/usr/bin/env python3
"""Split a PDF into multiple files by page ranges."""

import argparse
import json
import os
import sys

try:
    from PyPDF2 import PdfReader, PdfWriter
except ImportError:
    print("Error: PyPDF2 not installed.  Run: pip install PyPDF2", file=sys.stderr)
    sys.exit(1)


def parse_range(spec: str, total: int) -> list[int]:
    """Parse '1-5' or '3' or '2,4,6' into 0-based indices."""
    pages: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = max(1, int(a)), min(total, int(b))
            pages.update(range(a - 1, b))
        else:
            idx = int(part) - 1
            if 0 <= idx < total:
                pages.add(idx)
    return sorted(pages)


def write_subset(reader: PdfReader, indices: list[int], output_path: str) -> int:
    writer = PdfWriter()
    for i in indices:
        writer.add_page(reader.pages[i])
    with open(output_path, "wb") as f:
        writer.write(f)
    return len(indices)


def main():
    parser = argparse.ArgumentParser(description="Split a PDF file.")
    parser.add_argument("input", help="Input PDF path")
    parser.add_argument("--pages", default=None, help="Page range to extract (e.g. '1-5' or '3')")
    parser.add_argument("--each", action="store_true", help="One PDF per page")
    parser.add_argument("--every", type=int, default=0, help="Split into chunks of N pages")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--prefix", default=None, help="Output filename prefix")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    reader = PdfReader(args.input)
    total = len(reader.pages)
    base = args.prefix or os.path.splitext(os.path.basename(args.input))[0]
    out_dir = args.output or os.path.dirname(os.path.abspath(args.input))
    os.makedirs(out_dir, exist_ok=True)

    outputs: list[dict] = []

    if args.pages:
        indices = parse_range(args.pages, total)
        if not indices:
            print("Error: no valid pages in range.", file=sys.stderr)
            sys.exit(1)
        start, end = indices[0] + 1, indices[-1] + 1
        out_path = os.path.join(out_dir, f"{base}_p{start}-{end}.pdf")
        n = write_subset(reader, indices, out_path)
        outputs.append({"path": out_path, "pages": n})

    elif args.each:
        for i in range(total):
            out_path = os.path.join(out_dir, f"{base}_p{i + 1}.pdf")
            write_subset(reader, [i], out_path)
            outputs.append({"path": out_path, "pages": 1})

    elif args.every > 0:
        chunk_size = args.every
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            indices = list(range(start, end))
            out_path = os.path.join(out_dir, f"{base}_p{start + 1}-{end}.pdf")
            write_subset(reader, indices, out_path)
            outputs.append({"path": out_path, "pages": len(indices)})

    else:
        print("Error: specify --pages, --each, or --every.", file=sys.stderr)
        sys.exit(1)

    result = {
        "input": args.input,
        "totalPages": total,
        "outputFiles": len(outputs),
        "files": outputs,
    }

    if args.format == "json":
        print(json.dumps(result, indent=2))
    else:
        print(f"Split {args.input} ({total} pages) → {len(outputs)} files:")
        for f in outputs:
            print(f"  {f['path']} ({f['pages']} pages)")


if __name__ == "__main__":
    main()
