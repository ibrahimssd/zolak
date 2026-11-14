"""Apply the Sudanese-dialect normalizer (space/dialect_fix.py) to the knowledge base in place.

The chatbot copies wording from the entries it retrieves, so Egyptian/Gulf/MSA forms in the knowledge base
(«ما فيش», «ما بتنفعش», «هذا») end up in its answers. This rewrites question, answer and facts of every record
and summary (never links or German terms) in the processed files, after saving a .bak copy of each, and prints the
share of texts with non-Sudanese markers before and after.

Usage (then rebuild the index from kb_group.jsonl):
    python -m src.rag.sudanize_kb
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "space"))
from dialect_fix import build_vocab, sudanize_text  # noqa: E402

from src.train.dialect import score  # noqa: E402

FILES = ["kb_group.jsonl", "kb_audited.jsonl", "qa_audited.jsonl", "eval_audited.jsonl", "kb_summaries.jsonl"]


def text_of(r: dict) -> str:
    return " ".join([r.get("question", ""), r.get("answer", ""), *r.get("facts", [])])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proc", default="data/processed")
    args = ap.parse_args()
    proc = Path(args.proc)
    group = [json.loads(l) for l in (proc / "kb_group.jsonl").open(encoding="utf-8")]
    vocab = build_vocab([text_of(r) for r in group])   # same vocabulary the Space builds from the index
    for name in FILES:
        path = proc / name
        if not path.exists():
            continue
        rows = [json.loads(l) for l in path.open(encoding="utf-8")]
        before = sum(score(text_of(r))["other"] > 0 for r in rows)
        counts: Counter = Counter()
        for r in rows:
            for key in ("question", "answer"):
                if isinstance(r.get(key), str):
                    r[key] = sudanize_text(r[key], vocab, counts)
            if isinstance(r.get("facts"), list):
                r["facts"] = [sudanize_text(f, vocab, counts) for f in r["facts"]]
        after = sum(score(text_of(r))["other"] > 0 for r in rows)
        shutil.copy(path, path.with_suffix(".jsonl.bak"))
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} rows, with non-Sudanese markers {before} -> {after} | fixes {dict(counts)}")


if __name__ == "__main__":
    main()
