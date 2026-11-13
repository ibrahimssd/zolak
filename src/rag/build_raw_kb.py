"""Starter knowledge base from anonymized threads — no LLM (used until the GPU teacher is available).

For each answered thread:
  question = the asker's question text
  answer   = the replies most relevant to the question (bge-m3 similarity >= --min-sim), chronological,
             capped at --max-chars; replies from the asker are dropped
For each long info post: question = its first line, answer = the post (capped).

Extra privacy pass on everything that goes into the KB (it leaves this machine as a private HF dataset and is
shown to the model as context):
  * second Arabic NER pass with a LOWER confidence threshold (higher recall)
  * placeholders ([اسم] @مستخدم ...) and leftover lexicon names are removed from the text
  * the PII gate from postprocess (patterns + member full names) must pass, else the record is dropped

Output: data/processed/kb_raw.jsonl (same record format as kb.jsonl, source = "raw_thread" | "raw_post")

Usage: source scripts/env.sh; CUDA_VISIBLE_DEVICES= python3 -m src.rag.build_raw_kb    (CPU is fine)
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from src.ingest import pii
from src.ingest.anonymize import ArabicNER
from src.ingest.postprocess import FOREIGN_SCRIPT_RE, load_name_re

PLACEHOLDER_CLEAN = re.compile(r"\s*(?:\[(?:اسم|رقم|رابط|ايميل)\]|@مستخدم)\s*")
GREETING_PREFIX = re.compile(r"^(ال?سلام عليكم[^\n،,.]*[،,.]?|شباب|يا شباب|يا جماعة|مساء الخير|صباح الخير)\s*", re.I)


def clean(text: str) -> str:
    t = PLACEHOLDER_CLEAN.sub(" ", text)
    t = GREETING_PREFIX.sub("", t.strip())
    t = re.sub(r"[ \t]+", " ", t)
    return re.sub(r"\n{2,}", "\n", t).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proc", default="data/processed")
    ap.add_argument("--out", default="data/processed/kb_raw.jsonl")
    ap.add_argument("--min-sim", type=float, default=0.45)
    ap.add_argument("--max-replies", type=int, default=6)
    ap.add_argument("--max-chars", type=int, default=900)
    ap.add_argument("--ner-threshold", type=float, default=0.55)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    proc = Path(args.proc)
    threads = [json.loads(l) for l in (proc / "threads.jsonl").open(encoding="utf-8")]
    posts = [json.loads(l) for l in (proc / "info_posts.jsonl").open(encoding="utf-8")]
    emb = SentenceTransformer("BAAI/bge-m3", device=args.device)

    # 1) relevance-select replies per thread
    drafts = []
    for t in threads:
        q = clean(t["question"])
        replies = [m for m in t["messages"] if m["part"] == "reply" and m["role"] != "السائل"]
        replies = [(m, clean(m["text"])) for m in replies]
        replies = [(m, txt) for m, txt in replies if len(txt) >= 12]
        if len(q) < 15 or not replies:
            continue
        drafts.append({"id": t["thread_id"], "source": "raw_thread", "date": t["date"], "q": q, "replies": replies})
    all_q = emb.encode([d["q"] for d in drafts], normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    flat = [(i, txt) for i, d in enumerate(drafts) for _, txt in d["replies"]]
    all_r = emb.encode([txt for _, txt in flat], normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    sims: dict[int, list[float]] = {}
    for (i, _), v in zip(flat, all_r):
        sims.setdefault(i, []).append(float(all_q[i] @ v))

    records = []
    for i, d in enumerate(drafts):
        scored = [(s, k) for k, s in enumerate(sims[i]) if s >= args.min_sim]
        keep = sorted(k for _, k in sorted(scored, reverse=True)[: args.max_replies])
        answer, total = [], 0
        for k in keep:
            txt = d["replies"][k][1]
            if total + len(txt) > args.max_chars:
                break
            answer.append(txt)
            total += len(txt)
        if not answer:
            continue
        records.append({"id": d["id"], "source": d["source"], "date": d["date"], "topic": "",
                        "question": d["q"][:500], "answer": "\n".join(f"- {a}" for a in answer),
                        "facts": [], "german_terms": [], "links": [], "confidence": "raw",
                        "time_sensitive": True, "also_seen": []})
    for p in posts:
        txt = clean(p["text"])
        first = txt.split("\n", 1)[0][:200]
        records.append({"id": p["post_id"], "source": "raw_post", "date": p["date"], "topic": "",
                        "question": first, "answer": txt[: args.max_chars * 2], "facts": [], "german_terms": [],
                        "links": [], "confidence": "raw", "time_sensitive": True, "also_seen": []})

    # 2) second, high-recall NER pass + cleanup
    ner = ArabicNER(min_score=args.ner_threshold)
    counts: dict = {}
    for field in ("question", "answer"):
        scrubbed = ner.scrub([r[field] for r in records], counts)
        for r, s in zip(records, scrubbed):
            r[field] = clean(s)

    # 3) gate
    name_re = load_name_re(Path("data/interim/pii_lexicon.json"))
    drops: Counter = Counter()
    final = []
    for r in records:
        blob = r["question"] + " " + r["answer"]
        if len(r["question"]) < 10 or len(r["answer"]) < 30:
            drops["too_short"] += 1
        elif pii.find_pii(blob):
            drops["pii_pattern"] += 1
        elif name_re and name_re.search(blob):
            drops["member_name"] += 1
        elif FOREIGN_SCRIPT_RE.search(blob):
            drops["foreign_script"] += 1
        else:
            final.append(r)

    with Path(args.out).open("w", encoding="utf-8") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"threads {len(threads)} + posts {len(posts)} -> drafts {len(records)} -> KB {len(final)}")
    print("dropped:", dict(drops), "| extra NER names removed:", counts.get("ner_name", 0))
    print("answer chars: median", sorted(len(r["answer"]) for r in final)[len(final) // 2])


if __name__ == "__main__":
    main()
