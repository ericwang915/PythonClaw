#!/usr/bin/env python3
"""Add or remove password protection on PDF files."""

import argparse
import json
import os
import sys

try:
    from PyPDF2 import PdfReader, PdfWriter
except ImportError:
    print("Error: PyPDF2 not installed.  Run: pip install PyPDF2", file=sys.stderr)
    sys.exit(1)


def encrypt_pdf(
    input_path: str,
    output_path: str,
    user_password: str,
    owner_password: str | None = None,
    allow_print: bool = True,
    allow_copy: bool = True,
) -> dict:
    reader = PdfReader(input_path)
    writer = PdfWriter()

    for page in reader.pages:
        writer.add_page(page)

    if reader.metadata:
        writer.add_metadata(reader.metadata)

    permissions = 0
    if not allow_print:
        permissions |= 0x0004
    if not allow_copy:
        permissions |= 0x0010

    owner_pwd = owner_password or user_password
    writer.encrypt(
        user_password=user_password,
        owner_password=owner_pwd,
        permissions_flag=-1 if (allow_print and allow_copy) else ~permissions,
    )

    with open(output_path, "wb") as f:
        writer.write(f)

    return {
        "input": input_path,
        "output": output_path,
        "encrypted": True,
        "pages": len(reader.pages),
        "sizeBytes": os.path.getsize(output_path),
        "printAllowed": allow_print,
        "copyAllowed": allow_copy,
    }


def decrypt_pdf(input_path: str, output_path: str, password: str) -> dict:
    reader = PdfReader(input_path)

    if reader.is_encrypted:
        if not reader.decrypt(password):
            raise ValueError("Incorrect password — could not decrypt the PDF.")

    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)

    if reader.metadata:
        try:
            writer.add_metadata(reader.metadata)
        except Exception:
            pass

    with open(output_path, "wb") as f:
        writer.write(f)

    return {
        "input": input_path,
        "output": output_path,
        "decrypted": True,
        "pages": len(reader.pages),
        "sizeBytes": os.path.getsize(output_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Encrypt or decrypt PDF files.")
    sub = parser.add_subparsers(dest="command", required=True)

    # encrypt
    p_enc = sub.add_parser("encrypt", help="Password-protect a PDF")
    p_enc.add_argument("input", help="Input PDF path")
    p_enc.add_argument("output", help="Output PDF path")
    p_enc.add_argument("--password", required=True, help="User password")
    p_enc.add_argument("--owner-password", default=None, help="Owner password")
    p_enc.add_argument("--no-print", action="store_true", help="Disable printing")
    p_enc.add_argument("--no-copy", action="store_true", help="Disable copying")
    p_enc.add_argument("--format", choices=["text", "json"], default="text")

    # decrypt
    p_dec = sub.add_parser("decrypt", help="Remove password from a PDF")
    p_dec.add_argument("input", help="Input PDF path")
    p_dec.add_argument("output", help="Output PDF path")
    p_dec.add_argument("--password", required=True, help="Password to unlock")
    p_dec.add_argument("--format", choices=["text", "json"], default="text")

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    try:
        if args.command == "encrypt":
            result = encrypt_pdf(
                args.input,
                args.output,
                args.password,
                owner_password=args.owner_password,
                allow_print=not args.no_print,
                allow_copy=not args.no_copy,
            )
        else:
            result = decrypt_pdf(args.input, args.output, args.password)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.format == "json":
        print(json.dumps(result, indent=2))
    else:
        action = "Encrypted" if args.command == "encrypt" else "Decrypted"
        print(f"{action}: {args.input} → {args.output} ({result['pages']} pages, {result['sizeBytes']:,} bytes)")


if __name__ == "__main__":
    main()
