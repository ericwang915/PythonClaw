#!/usr/bin/env python3
"""Convert between PDF and image formats."""

import argparse
import glob
import json
import os
import sys

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow not installed.  Run: pip install Pillow", file=sys.stderr)
    sys.exit(1)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".gif", ".webp"}


def images_to_pdf(
    images: list[str],
    output: str,
    fit: bool = False,
    page_size: str = "A4",
) -> dict:
    """Combine image files into a single PDF."""
    page_sizes = {
        "A4": (595.28, 841.89),
        "letter": (612, 792),
    }
    pw, ph = page_sizes.get(page_size, page_sizes["A4"])

    img_list: list[Image.Image] = []
    for path in images:
        img = Image.open(path).convert("RGB")
        if fit:
            img.thumbnail((int(pw * 2), int(ph * 2)), Image.LANCZOS)
        img_list.append(img)

    if not img_list:
        raise ValueError("No valid images provided.")

    first, rest = img_list[0], img_list[1:]
    first.save(output, "PDF", save_all=True, append_images=rest, resolution=150)

    return {
        "output": output,
        "imageCount": len(img_list),
        "sizeBytes": os.path.getsize(output),
    }


def pdf_to_images(
    input_pdf: str,
    output_dir: str,
    pages: list[int] | None = None,
    img_format: str = "png",
    dpi: int = 150,
) -> dict:
    """Convert PDF pages to images using pdf2image (poppler) if available,
    otherwise fall back to a minimal PyPDF2-based extraction."""
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_pdf))[0]
    outputs: list[dict] = []

    try:
        from pdf2image import convert_from_path

        kwargs: dict = {"dpi": dpi, "fmt": img_format}
        if pages is not None:
            for page_num in pages:
                imgs = convert_from_path(
                    input_pdf,
                    first_page=page_num,
                    last_page=page_num,
                    **kwargs,
                )
                for img in imgs:
                    out_path = os.path.join(output_dir, f"{base}_p{page_num}.{img_format}")
                    img.save(out_path)
                    outputs.append({"path": out_path, "page": page_num})
        else:
            imgs = convert_from_path(input_pdf, **kwargs)
            for i, img in enumerate(imgs, 1):
                out_path = os.path.join(output_dir, f"{base}_p{i}.{img_format}")
                img.save(out_path)
                outputs.append({"path": out_path, "page": i})

    except ImportError:
        from PyPDF2 import PdfReader

        reader = PdfReader(input_pdf)
        total = len(reader.pages)
        target_pages = pages if pages else list(range(1, total + 1))

        for page_num in target_pages:
            if page_num < 1 or page_num > total:
                continue
            page = reader.pages[page_num - 1]
            img_found = False
            for img_obj in page.images:
                out_path = os.path.join(output_dir, f"{base}_p{page_num}.{img_format}")
                with open(out_path, "wb") as f:
                    f.write(img_obj.data)
                outputs.append({"path": out_path, "page": page_num})
                img_found = True
                break
            if not img_found:
                out_path = os.path.join(output_dir, f"{base}_p{page_num}.txt")
                text = page.extract_text() or "(no extractable content)"
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(text)
                outputs.append({"path": out_path, "page": page_num, "note": "text-fallback"})

    return {
        "input": input_pdf,
        "outputDir": output_dir,
        "fileCount": len(outputs),
        "files": outputs,
    }


def parse_range(spec: str) -> list[int]:
    pages: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            pages.extend(range(int(a), int(b) + 1))
        else:
            pages.append(int(part))
    return pages


def main():
    parser = argparse.ArgumentParser(description="Convert between PDF and images.")
    sub = parser.add_subparsers(dest="command", required=True)

    # to-pdf
    p_topdf = sub.add_parser("to-pdf", help="Combine images into a PDF")
    p_topdf.add_argument("output", help="Output PDF path")
    p_topdf.add_argument("images", nargs="*", help="Input image files")
    p_topdf.add_argument("--dir", default=None, help="Directory of images")
    p_topdf.add_argument("--fit", action="store_true", help="Fit images to page")
    p_topdf.add_argument("--page-size", default="A4", choices=["A4", "letter"])
    p_topdf.add_argument("--format", choices=["text", "json"], default="text")

    # to-images
    p_toimg = sub.add_parser("to-images", help="Convert PDF pages to images")
    p_toimg.add_argument("input", help="Input PDF path")
    p_toimg.add_argument("--pages", default=None, help="Page range (e.g. '1-5')")
    p_toimg.add_argument("--output", default=None, help="Output directory")
    p_toimg.add_argument("--img-format", default="png", choices=["png", "jpeg"])
    p_toimg.add_argument("--dpi", type=int, default=150)
    p_toimg.add_argument("--format", choices=["text", "json"], default="text")

    args = parser.parse_args()

    if args.command == "to-pdf":
        images = list(args.images)
        if args.dir:
            for ext in IMAGE_EXTENSIONS:
                images.extend(sorted(glob.glob(os.path.join(args.dir, f"*{ext}"))))
        if not images:
            print("Error: no images provided.", file=sys.stderr)
            sys.exit(1)
        result = images_to_pdf(images, args.output, fit=args.fit, page_size=args.page_size)
        if args.format == "json":
            print(json.dumps(result, indent=2))
        else:
            print(f"Created {args.output} from {result['imageCount']} images ({result['sizeBytes']:,} bytes)")

    elif args.command == "to-images":
        pages = parse_range(args.pages) if args.pages else None
        out_dir = args.output or os.path.dirname(os.path.abspath(args.input))
        result = pdf_to_images(args.input, out_dir, pages, args.img_format, args.dpi)
        if args.format == "json":
            print(json.dumps(result, indent=2))
        else:
            print(f"Converted {args.input} → {result['fileCount']} files in {result['outputDir']}")
            for f in result["files"]:
                note = f" ({f['note']})" if "note" in f else ""
                print(f"  Page {f['page']}: {f['path']}{note}")


if __name__ == "__main__":
    main()
