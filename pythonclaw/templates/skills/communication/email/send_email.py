#!/usr/bin/env python3
"""Generic SMTP email sender.  Reads credentials from pythonclaw.json."""

import argparse
import json
import os
import re
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def _load_email_config() -> dict:
    """Load skills.email from pythonclaw.json."""
    for path in ["pythonclaw.json", os.path.expanduser("~/.pythonclaw/pythonclaw.json")]:
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as f:
            text = f.read()
        text = re.sub(r'//.*$', '', text, flags=re.MULTILINE)
        text = re.sub(r',\s*([}\]])', r'\1', text)
        data = json.loads(text)
        return data.get("skills", {}).get("email", {})
    return {}


def send_email(
    to: list[str],
    subject: str,
    body: str,
    *,
    cc: list[str] | None = None,
    bcc: list[str] | None = None,
    html: bool = False,
) -> str:
    cfg = _load_email_config()
    server = cfg.get("smtpServer", "smtp.gmail.com")
    port = int(cfg.get("smtpPort", 587))
    sender = cfg.get("senderEmail", "")
    password = cfg.get("senderPassword", "")

    if not sender or not password:
        return "Error: Email credentials not configured.  Set skills.email in pythonclaw.json."

    msg = MIMEMultipart("alternative")
    msg["From"] = sender
    msg["To"] = ", ".join(to)
    msg["Subject"] = subject
    if cc:
        msg["Cc"] = ", ".join(cc)

    content_type = "html" if html else "plain"
    msg.attach(MIMEText(body, content_type, "utf-8"))

    all_recipients = list(to) + (cc or []) + (bcc or [])

    try:
        with smtplib.SMTP(server, port, timeout=30) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.ehlo()
            smtp.login(sender, password)
            smtp.sendmail(sender, all_recipients, msg.as_string())
        return f"Email sent to {', '.join(to)} (subject: {subject})"
    except Exception as exc:
        return f"Send failed: {exc}"


def main():
    parser = argparse.ArgumentParser(description="Send an email via SMTP.")
    parser.add_argument("--to", required=True, help="Recipient(s), comma-separated")
    parser.add_argument("--subject", required=True, help="Email subject line")
    parser.add_argument("--body", required=True, help="Email body text")
    parser.add_argument("--cc", default="", help="CC recipients, comma-separated")
    parser.add_argument("--bcc", default="", help="BCC recipients, comma-separated")
    parser.add_argument("--html", action="store_true", help="Treat body as HTML")
    args = parser.parse_args()

    to = [a.strip() for a in args.to.split(",") if a.strip()]
    cc = [a.strip() for a in args.cc.split(",") if a.strip()] or None
    bcc = [a.strip() for a in args.bcc.split(",") if a.strip()] or None

    result = send_email(to, args.subject, args.body, cc=cc, bcc=bcc, html=args.html)
    print(result)


if __name__ == "__main__":
    main()
