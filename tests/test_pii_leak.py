"""Leak gate over the real processed data (skipped when data/ is absent, e.g. in CI).

Every artifact that can leave this machine (training data, retrieval KB, FAQ bank) must pass:
  * no phone / email / IBAN / long-digit / raw-mention patterns
  * no sender phone number (digits) and no member full name from the local PII lexicon
"""
import json
import re
from pathlib import Path

import pytest

from src.ingest import pii

LEX = Path("data/interim/pii_lexicon.json")
ARTIFACTS = [
    "data/processed/messages.jsonl", "data/processed/threads.jsonl", "data/processed/info_posts.jsonl",
    "data/processed/qa.jsonl", "data/processed/kb.jsonl", "data/processed/faq_bank.jsonl",
    "data/processed/kb_raw.jsonl", "data/index/kb.jsonl",
]
TEXT_FIELDS = ("text", "question", "answer", "question_sd", "answer_sd", "fact", "facts")


def _texts(path: Path):
    for line in path.open(encoding="utf-8"):
        rec = json.loads(line)
        stack = [rec]
        while stack:
            x = stack.pop()
            if isinstance(x, dict):
                stack += [v for k, v in x.items() if k in TEXT_FIELDS or k == "messages" or isinstance(v, (dict, list))]
            elif isinstance(x, list):
                stack += x
            elif isinstance(x, str):
                yield x


@pytest.mark.skipif(not LEX.exists(), reason="no local data")
@pytest.mark.parametrize("artifact", ARTIFACTS)
def test_no_pii(artifact):
    path = Path(artifact)
    if not path.exists():
        pytest.skip(f"{artifact} not built yet")
    lex = json.loads(LEX.read_text(encoding="utf-8"))
    phones = {re.sub(r"\D", "", s) for s in lex["senders"] if re.fullmatch(r"[+\d][\d \-()]+", s)}
    phones = {p[-8:] for p in phones if len(p) >= 8}  # last 8 digits are enough to identify
    names = [n for n in lex["full_names"] if len(n) >= 6 and " " in n]  # multi-word full names
    name_re = re.compile("|".join(re.escape(n) for n in names), re.I) if names else None

    leaks = []
    for t in _texts(path):
        digits = re.sub(r"\D", "", pii.normalize_digits(t))
        leaks += [f"pattern:{h}" for h in pii.find_pii(t)]
        leaks += [f"phone:{p[:3]}…" for p in phones if p in digits]
        if name_re and (m := name_re.search(t)):
            leaks.append(f"name:{m.group(0)[:2]}…")
    assert not leaks, f"{len(leaks)} leaks in {artifact}: {leaks[:10]}"
