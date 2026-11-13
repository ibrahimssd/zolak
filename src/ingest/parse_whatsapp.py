"""Step 1 — parse a WhatsApp Android text export into structured messages.

Output (data/interim/parsed.jsonl, still contains PII — never commit):
    {"id", "ts", "sender", "text", "edited", "kind"}

kind: "message" | "system" | "media" | "deleted" | "contact_card" | "poll" | "file"
Only "message" records are used downstream; others are kept for statistics.

Usage:
    python -m src.ingest.parse_whatsapp --input data/raw/chat.txt --out data/interim/parsed.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

# "4/18/24, 14:00 - Rest of line"
HEADER_RE = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{2,4}), (\d{1,2}):(\d{2}) - (.*)$")

# System events have no "Sender: " prefix. Patterns are matched on the part after " - ".
SYSTEM_PATTERNS = [
    re.compile(p)
    for p in (
        r"Messages and calls are end-to-end encrypted",
        r" created group ",
        r" added ",
        r"\badded you$",
        r" was added$",
        r"was added$",
        r" removed ",
        r" left$",
        r"joined using (this|a) group('s)? (invite )?link",
        r"changed their phone number",
        r"changed (the|this) group",
        r"changed the subject",
        r"changed the group description",
        r"pinned a message",
        r"turned (on|off) disappearing messages",
        r"security code (with|changed)",
        r"is now an admin|no longer an admin",
        r"changed this group's (icon|settings)",
        r"deleted this group's icon",
        r"requested to join",
    )
]

MEDIA_RE = re.compile(r"^<Media omitted>$")
DELETED_RE = re.compile(r"^(This message was deleted|You deleted this message)$")
EDITED_TAG = "<This message was edited>"
VCF_RE = re.compile(r"\.vcf \(file attached\)$")
FILE_RE = re.compile(r"\(file attached\)$")
POLL_RE = re.compile(r"^POLL:")

# Bidi / formatting marks WhatsApp sprinkles around names and RTL text.
# U+2068/U+2069 (FSI/PDI) are KEPT here because they delimit @mentions; anonymize.py handles them.
BIDI_MARKS = re.compile("[‎‏‪-‮⁦⁧​﻿]")
NARROW_NBSP = re.compile("[  ]")


@dataclass
class Message:
    id: int
    ts: str
    sender: str | None
    text: str
    edited: bool
    kind: str


def _clean(s: str) -> str:
    s = BIDI_MARKS.sub("", s)
    s = NARROW_NBSP.sub(" ", s)
    return s


def _parse_ts(mo: re.Match) -> str:
    month, day, year, hh, mm = (int(mo.group(i)) for i in range(1, 6))
    if year < 100:
        year += 2000
    return datetime(year, month, day, hh, mm).isoformat()


def _split_sender(rest: str) -> tuple[str | None, str]:
    """Return (sender, text) or (None, rest) for system lines."""
    if any(p.search(rest) for p in SYSTEM_PATTERNS) and ": " not in rest[:60]:
        return None, rest
    if ": " in rest:
        sender, text = rest.split(": ", 1)
        # Sender names are short; a long "sender" means we split inside a system line.
        if len(sender) <= 60:
            return sender.strip(), text
    return None, rest


def _classify(text: str) -> tuple[str, str, bool]:
    edited = EDITED_TAG in text
    text = text.replace(EDITED_TAG, "").strip()
    if MEDIA_RE.match(text):
        return "media", text, edited
    if DELETED_RE.match(text):
        return "deleted", text, edited
    if VCF_RE.search(text):
        return "contact_card", text, edited
    if FILE_RE.search(text):
        return "file", text, edited
    if POLL_RE.match(text):
        return "poll", text, edited
    return "message", text, edited


def parse(lines: list[str]) -> list[Message]:
    msgs: list[Message] = []
    cur: dict | None = None

    def flush():
        if cur is None:
            return
        if cur["sender"] is None:
            kind, text, edited = "system", "\n".join(cur["lines"]).strip(), False
        else:
            kind, text, edited = _classify("\n".join(cur["lines"]))
        msgs.append(Message(len(msgs), cur["ts"], cur["sender"], text, edited, kind))

    for raw in lines:
        line = _clean(raw.rstrip("\n"))
        mo = HEADER_RE.match(line)
        if mo:
            flush()
            sender, text = _split_sender(mo.group(6))
            cur = {"ts": _parse_ts(mo), "sender": sender, "lines": [text]}
        elif cur is not None:
            cur["lines"].append(line)
    flush()
    return msgs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="data/raw/chat.txt")
    ap.add_argument("--out", default="data/interim/parsed.jsonl")
    args = ap.parse_args()

    lines = Path(args.input).read_text(encoding="utf-8").splitlines()
    msgs = parse(lines)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for m in msgs:
            f.write(json.dumps(asdict(m), ensure_ascii=False) + "\n")

    kinds = Counter(m.kind for m in msgs)
    senders = {m.sender for m in msgs if m.sender}
    print(f"parsed {len(msgs)} records from {len(lines)} lines -> {out}")
    print("kinds:", dict(kinds))
    print("distinct senders:", len(senders))
    print("date range:", min(m.ts for m in msgs), "->", max(m.ts for m in msgs))


if __name__ == "__main__":
    main()
