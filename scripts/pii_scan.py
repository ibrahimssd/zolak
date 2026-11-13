"""Block commits that contain chat data or personal data (used locally and in GitHub CI).

Fails if a tracked file contains:
  * a WhatsApp export line ("4/18/24, 14:00 - ...")
  * phone numbers / emails / IBANs / long digit runs (src/ingest/pii.py patterns)
tests/ is only checked for WhatsApp lines (it intentionally contains fake numbers).

Usage: python3 scripts/pii_scan.py $(git ls-files)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ingest.pii import EMAIL_RE, IBAN_RE, INTL_PHONE_RE, LONG_DIGITS_RE, normalize_digits  # noqa: E402

WA_LINE = re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2} - [^:]{1,60}: ", re.M)
TEXT_EXT = {".py", ".md", ".txt", ".json", ".jsonl", ".js", ".html", ".css", ".yml", ".yaml", ".sh", ".toml", ".cfg"}
# addresses that are fine to publish
OK_EMAILS = {"noreply@anthropic.com"}


def scan(path: Path) -> list[str]:
    if path.suffix not in TEXT_EXT or not path.is_file():
        return []
    text = path.read_text(encoding="utf-8", errors="ignore")
    if "pii-scan: ignore-file" in text:  # only for the pattern definitions and synthetic test fixtures
        return []
    hits = [f"whatsapp-line: {m.group(0)[:25]}…" for m in WA_LINE.finditer(text)]
    if path.parts and path.parts[0] == "tests":
        return hits
    t = normalize_digits(text)
    for name, rx in (("phone", INTL_PHONE_RE), ("long-digits", LONG_DIGITS_RE), ("iban", IBAN_RE)):
        hits += [f"{name}: {m.group(0)[:4]}…" for m in rx.finditer(t)
                 if m.group(0) not in "0123456789" * 3]  # digit-translation tables, not data
    hits += [f"email: {m.group(0)[:3]}…" for m in EMAIL_RE.finditer(t) if m.group(0).lower() not in OK_EMAILS]
    return hits


def main() -> int:
    bad = 0
    for f in sys.argv[1:]:
        for h in scan(Path(f)):
            print(f"{f}: {h}")
            bad += 1
    print(f"pii_scan: {bad} finding(s) in {len(sys.argv) - 1} files")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
