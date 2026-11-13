"""Step 2 — anonymize parsed messages. Runs BEFORE anything else touches the data.

Layers:
  1. speakers -> salted HMAC ids (U + 8 hex); salt kept in data/.salt (gitignored)
  2. @mentions -> "@مستخدم" (resolved speaker id kept in metadata for threading)
  3. URLs: allowlisted official domains kept, everything else -> [رابط]
  4. emails, IBANs, phone numbers, passport-like IDs, long digit runs -> [ايميل]/[رقم]
  5. person names from a lexicon (sender display names, "~ Name" in system lines) -> [اسم]
  6. optional Arabic NER (CAMeLBERT) for person names written in Arabic script -> [اسم]

Input : data/interim/parsed.jsonl   (PII)
Output: data/interim/anon.jsonl      (anonymized, kind=="message" only)
        data/interim/anon_audit.txt  (sample for manual spot-check)
        data/interim/pii_lexicon.json (PII strings for the leak test — local only)

Usage:
    python -m src.ingest.anonymize            # regex + lexicon
    python -m src.ingest.anonymize --ner      # + Arabic NER on GPU
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import random
import re
import secrets
import unicodedata
from collections import Counter
from pathlib import Path

from . import pii

SALT_PATH = Path("data/.salt")

# Tokens in contact names that are places/orgs/abbreviations, not person names.
NAME_TOKEN_STOP = {
    "thi", "uds", "uae", "neth", "dortmund", "stuttgart", "berlin", "munich", "münchen", "wuppertal",
    "education", "sunsom", "germany", "sudan", "admin", "group", "uni", "university", "the", "and",
    "for", "dr", "eng", "mr", "mrs", "ms",
    # common English/German words that also occur as nicknames
    "bad", "jay", "wiz", "emy", "ibn", "bin", "abu", "umm", "das", "der", "die", "sun", "star", "king",
    "queen", "boss", "love", "life", "hope", "angel", "doctor", "student", "engineer", "habibi",
    "man", "hem", "zee", "ytm",
}
EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF☀-➿\U0001F1E6-\U0001F1FF‍️⃣]+")
TILDE_NAME_RE = re.compile(
    r"~\s?([^~\n]+?)(?=\s+(?:added|was added|removed|left|joined|changed|pinned|and)\b|,|$)"
)


def load_salt() -> bytes:
    if not SALT_PATH.exists():
        SALT_PATH.parent.mkdir(parents=True, exist_ok=True)
        SALT_PATH.write_text(secrets.token_hex(32))
    return bytes.fromhex(SALT_PATH.read_text().strip())


def norm_key(s: str) -> str:
    """Normalize a sender / mention target so phones and names compare equal."""
    s = pii.normalize_digits(s).strip().lstrip("~").strip()
    if re.fullmatch(r"[+\d][\d \-()]+", s):
        return re.sub(r"\D", "", s)
    return EMOJI_RE.sub("", s).strip().lower()


def speaker_id(salt: bytes, sender: str) -> str:
    return "U" + hmac.new(salt, norm_key(sender).encode(), hashlib.sha256).hexdigest()[:8]


def clean_name(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = EMOJI_RE.sub("", s)
    s = re.sub(r"[^\w\s'\-.]", " ", s)          # drop symbols
    return re.sub(r"\s+", " ", s).strip(" .-'")


def is_namelike(s: str) -> bool:
    letters = sum(c.isalpha() for c in s)
    return letters >= 3 and not re.fullmatch(r"[+\d][\d \-()]+", s)


def build_name_lexicon(records: list[dict]) -> tuple[set[str], set[str]]:
    """Return (full_names, name_tokens) of group members found in senders and system lines."""
    full: set[str] = set()
    for r in records:
        s = r.get("sender")
        if s and not re.fullmatch(r"[+\d][\d \-()]+", s):
            full.add(clean_name(s))
        if r["kind"] == "system" or r["kind"] == "message":
            for m in TILDE_NAME_RE.finditer(r["text"] if r["kind"] == "system" else ""):
                full.add(clean_name(m.group(1)))
            for m in pii.MENTION_ISOLATE_RE.finditer(r["text"]):
                t = m.group(1).strip()
                if t.startswith("~"):
                    full.add(clean_name(t.lstrip("~")))
    full = {f for f in full if is_namelike(f) and f.lower() not in NAME_TOKEN_STOP}
    tokens: set[str] = set()
    for f in full:
        for t in re.split(r"[\s_.\-]+", f):
            t = t.strip()
            if is_namelike(t) and t.lower() not in NAME_TOKEN_STOP:
                tokens.add(t)
    return full, tokens


def compile_name_re(full: set[str], tokens: set[str]) -> re.Pattern | None:
    """Long names/tokens match case-insensitively; 3-letter tokens only in their exact casing."""
    items = {*full, *tokens}
    if not items:
        return None
    long_ = sorted((x for x in items if len(x) >= 4), key=len, reverse=True)
    short = sorted((x for x in items if len(x) < 4), key=len, reverse=True)
    parts = []
    if long_:
        parts.append("(?i:" + "|".join(re.escape(x) for x in long_) + ")")
    if short:
        parts.append("|".join(re.escape(x) for x in short))
    # \w already covers Arabic letters; the raw U+0600 block would also swallow Arabic punctuation (، ؛ ؟)
    return re.compile(r"(?<!\w)(?:" + "|".join(parts) + r")(?!\w)")


def scrub_mentions(text: str, sender_ids: dict[str, str], counts: dict) -> tuple[str, list[str]]:
    targets: list[str] = []

    def repl(m: re.Match) -> str:
        key = norm_key(m.group(1) if m.lastindex else m.group(0)[1:])
        if key in sender_ids:
            targets.append(sender_ids[key])
        counts["mention"] = counts.get("mention", 0) + 1
        return pii.PH_MENTION

    text = pii.MENTION_ISOLATE_RE.sub(repl, text)
    text = pii.MENTION_DIGITS_RE.sub(repl, text)
    text = pii.ISOLATES_RE.sub("", text)
    return text, targets


class ArabicNER:
    """Person-name scrubber using CAMeLBERT NER (B-PERS/I-PERS). Loaded lazily, GPU if available."""

    MODEL = "CAMeL-Lab/bert-base-arabic-camelbert-mix-ner"
    # Words the NER model commonly mislabels as PERS in chat text.
    NOT_NAMES = {"الله", "يا", "شباب", "جماعة", "اخوانا", "الشباب", "اخوي", "اخواني", "اختي", "ربنا"}

    def __init__(self, batch_size: int = 64, min_score: float = 0.80):
        import torch
        from transformers import pipeline

        self.pipe = pipeline(
            "token-classification", model=self.MODEL, aggregation_strategy="first",
            device=0 if torch.cuda.is_available() else -1,
        )
        self.batch_size = batch_size
        self.min_score = min_score

    @staticmethod
    def _chunks(text: str, max_chars: int = 600) -> list[tuple[int, str]]:
        """Split into (offset, chunk) pieces <= max_chars at newline/space boundaries."""
        pieces, start = [], 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            if end < len(text):
                cut = max(text.rfind("\n", start, end), text.rfind(" ", start, end))
                end = cut + 1 if cut > start else end
            pieces.append((start, text[start:end]))
            start = end
        return pieces

    @staticmethod
    def _word_bounds(text: str, s: int, e: int) -> tuple[int, int]:
        """Widen a (sub)word span to full word boundaries so no name fragments survive."""
        while s > 0 and (text[s - 1].isalnum() or text[s - 1] == "\u0640"):
            s -= 1
        while e < len(text) and (text[e].isalnum() or text[e] == "\u0640"):
            e += 1
        return s, e

    def scrub(self, texts: list[str], counts: dict) -> list[str]:
        out = list(texts)
        jobs = [  # (text index, offset, chunk)
            (i, off, ch) for i, t in enumerate(texts) if re.search("[؀-ۿ]", t)
            for off, ch in self._chunks(t)
        ]
        preds = self.pipe([ch for _, _, ch in jobs], batch_size=self.batch_size)
        spans_by_text: dict[int, list[tuple[int, int]]] = {}
        for (i, off, _), ents in zip(jobs, preds):
            spans_by_text.setdefault(i, []).extend(
                (off + e["start"], off + e["end"]) for e in ents
                if e["entity_group"] == "PERS" and e["score"] >= self.min_score
                and e["word"].replace(" ", "") not in self.NOT_NAMES and len(e["word"].strip()) >= 2
            )
        for i, spans in spans_by_text.items():
            t = out[i]
            for s, e in sorted({self._word_bounds(t, s, e) for s, e in spans}, reverse=True):
                t = t[:s] + pii.PH_NAME + t[e:]
                counts["ner_name"] = counts.get("ner_name", 0) + 1
            out[i] = re.sub(r"(\[اسم\]\s*){2,}", pii.PH_NAME + " ", t)
        return out


def anonymize(records: list[dict], salt: bytes, use_ner: bool = False) -> tuple[list[dict], Counter, dict]:
    sender_ids = {norm_key(r["sender"]): speaker_id(salt, r["sender"]) for r in records if r.get("sender")}
    full, tokens = build_name_lexicon(records)
    name_re = compile_name_re(full, tokens)
    totals: Counter = Counter()
    out: list[dict] = []
    for r in records:
        if r["kind"] != "message":
            continue
        counts: dict = {}
        t = pii.normalize_digits(r["text"])
        t, targets = scrub_mentions(t, sender_ids, counts)
        t = pii.scrub_urls(t, counts)
        t = pii.scrub_numbers(t, counts)
        if name_re:
            t, n = name_re.subn(pii.PH_NAME, t)
            if n:
                counts["lexicon_name"] = n
        totals.update(counts)
        out.append({
            "id": r["id"], "ts": r["ts"], "speaker": sender_ids[norm_key(r["sender"])],
            "text": t, "edited": r["edited"], "mentions": targets, "pii": counts,
        })
    if use_ner:
        ner = ArabicNER()
        counts: dict = {}
        texts = ner.scrub([m["text"] for m in out], counts)
        for m, t in zip(out, texts):
            if t != m["text"]:
                m["pii"]["ner_name"] = m["pii"].get("ner_name", 0) + t.count(pii.PH_NAME) - m["text"].count(pii.PH_NAME)
                m["text"] = t
        totals.update(counts)
    lexicon = {
        "senders": sorted({r["sender"] for r in records if r.get("sender")}),
        "full_names": sorted(full),
        "name_tokens": sorted(tokens),
    }
    return out, totals, lexicon


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="data/interim/parsed.jsonl")
    ap.add_argument("--out", default="data/interim/anon.jsonl")
    ap.add_argument("--ner", action="store_true", help="also run Arabic NER for person names")
    ap.add_argument("--audit-n", type=int, default=200)
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.input, encoding="utf-8")]
    out, totals, lexicon = anonymize(records, load_salt(), use_ner=args.ner)

    p = Path(args.out)
    with p.open("w", encoding="utf-8") as f:
        for m in out:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    (p.parent / "pii_lexicon.json").write_text(json.dumps(lexicon, ensure_ascii=False, indent=1), encoding="utf-8")

    # audit: half messages that had something scrubbed, half random
    rnd = random.Random(0)
    touched = [m for m in out if m["pii"]]
    sample = rnd.sample(touched, min(args.audit_n // 2, len(touched))) + rnd.sample(out, args.audit_n // 2)
    with (p.parent / "anon_audit.txt").open("w", encoding="utf-8") as f:
        for m in sample:
            f.write(f"--- #{m['id']} {m['ts']} {m['speaker']} {m['pii']}\n{m['text']}\n")

    leaks = [(m["id"], h) for m in out for h in pii.find_pii(m["text"])]
    print(f"anonymized {len(out)} messages -> {p}")
    print("scrub counts:", dict(totals))
    print(f"lexicon: {len(lexicon['full_names'])} full names, {len(lexicon['name_tokens'])} tokens")
    print(f"residual PII-pattern hits: {len(leaks)}", leaks[:10])


if __name__ == "__main__":
    main()
