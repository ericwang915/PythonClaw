---
name: pdf_protect
description: "Add or remove password protection on PDF files. Use when: user asks to encrypt, password-protect, unlock, or decrypt a PDF. Supports owner and user passwords with permission controls. NOT for: digital signatures, certificate-based encryption, or DRM."
dependencies: PyPDF2
metadata:
  emoji: "🔒"
---

# PDF Protect Skill

Add or remove password protection on PDF files using PyPDF2.

## When to Use

✅ **USE this skill when:**
- "Password protect this PDF"
- "Encrypt report.pdf"
- "Remove password from this PDF"
- "Unlock this PDF — password is 1234"
- User wants to add or remove PDF encryption

## When NOT to Use

❌ **DON'T use this skill when:**
- Digital signatures → use dedicated signing tools
- Certificate-based encryption → use specialized PKI tools
- DRM protection → not supported
- Reading PDF content → use `pdf_reader`

## Usage/Commands

### Encrypt a PDF

```bash
python {skill_path}/protect_pdf.py encrypt INPUT_PDF OUTPUT_PDF --password SECRET [options]
```

Options:
- `--password SECRET` — user password (required to open the PDF)
- `--owner-password SECRET` — owner password (for permissions; defaults to user password)
- `--no-print` — disable printing
- `--no-copy` — disable text copying
- `--format json` — output result as JSON

### Decrypt a PDF

```bash
python {skill_path}/protect_pdf.py decrypt INPUT_PDF OUTPUT_PDF --password SECRET [options]
```

Options:
- `--password SECRET` — password to unlock the PDF
- `--format json` — output result as JSON

### Examples

- "Protect with password" → `python {skill_path}/protect_pdf.py encrypt doc.pdf protected.pdf --password mypass`
- "Restrict printing" → `python {skill_path}/protect_pdf.py encrypt doc.pdf out.pdf --password pass --no-print`
- "Unlock PDF" → `python {skill_path}/protect_pdf.py decrypt locked.pdf unlocked.pdf --password pass123`

## Notes

- Install dependency: `pip install PyPDF2`
- AES-128 encryption is used by default
- If owner password is not set, it defaults to the user password
