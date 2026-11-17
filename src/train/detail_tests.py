"""Questions whose answers should be detailed: the topic-card questions of the web page and the questions students
rated 👎 in the live test (feedback dataset). Used by eval_rag (completeness) and retrieval_lab (answerable@2).

Feedback is private: it is downloaded to data/interim/feedback/ (git-ignored) and only questions without personal
data (src.ingest.pii) and mostly in Arabic script are kept.

Output: data/processed/detail_tests.jsonl  {"query", "source": "topic_card"|"feedback_down", "correction"?}
Usage:  python -m src.train.detail_tests
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from src.ingest import pii

ARABIC = re.compile(r"[؀-ۿ]")


def topic_card_questions(app_js: Path = Path("web/app.js")) -> list[str]:
    return re.findall(r'\bq: "([^"]+)"', app_js.read_text(encoding="utf-8"))


def fetch_feedback(repo: str, out_dir: Path) -> list[dict]:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    rows = []
    for f in api.list_repo_files(repo, repo_type="dataset"):
        if f.startswith("data/") and f.endswith(".jsonl"):
            p = hf_hub_download(repo, f, repo_type="dataset", local_dir=str(out_dir))
            rows += [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feedback-repo", default="ibrahimssd11/zolak-feedback")
    ap.add_argument("--out", default="data/processed/detail_tests.jsonl")
    args = ap.parse_args()

    rows = fetch_feedback(args.feedback_repo, Path("data/interim/feedback"))
    corrections: dict[str, str] = {}
    for r in rows:   # students' corrections (kept for reference; never shown to the model under test)
        q, c = re.sub(r"\s+", " ", r.get("question") or "").strip(), (r.get("correction") or "").strip()
        if q and c and not pii.find_pii(c):
            corrections.setdefault(q, c)
    tests, seen = [], set()
    for q in topic_card_questions():
        t = {"query": q, "source": "topic_card"}
        if q in corrections:
            t["correction"] = corrections[q]
        tests.append(t); seen.add(q)
    skipped = {"personal_data": 0, "not_arabic": 0}
    for r in rows:
        q = re.sub(r"\s+", " ", r.get("question") or "").strip()
        if r.get("rating") != "down" or not q or q in seen:
            continue
        if pii.find_pii(q):
            skipped["personal_data"] += 1
            continue
        if len(ARABIC.findall(q)) < 0.5 * len(re.sub(r"\W", "", q)):
            skipped["not_arabic"] += 1
            continue
        seen.add(q)
        t = {"query": q, "source": "feedback_down"}
        if q in corrections:
            t["correction"] = corrections[q]
        tests.append(t)
    Path(args.out).write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in tests), encoding="utf-8")
    print(f"{len(tests)} detail tests ({sum(t['source'] == 'topic_card' for t in tests)} topic cards, "
          f"{sum(t['source'] == 'feedback_down' for t in tests)} 👎 questions from {len(rows)} feedback rows); "
          f"skipped {skipped} -> {args.out}")


if __name__ == "__main__":
    main()
