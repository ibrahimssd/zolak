"""Build official-source KB records from the teacher output, and the combined KB used for the search index.

Input : data/official/distill_official_raw.jsonl, data/processed/kb.jsonl (group knowledge)
Output: data/official/kb_official.jsonl   source="official" records (site, link, fetch date)
        data/processed/kb_all.jsonl       group + official records -> src.rag.build_index --kb ...

Usage: python3 -m src.web.build_official_kb [--device cuda]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.ingest import pii
from src.ingest.postprocess import FOREIGN_SCRIPT_RE, PLACEHOLDER_RE, build_vocab, sudanize


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/official/distill_official_raw.jsonl")
    ap.add_argument("--group-kb", default="data/processed/kb.jsonl")
    ap.add_argument("--out", default="data/official/kb_official.jsonl")
    ap.add_argument("--all", default="data/processed/kb_all.jsonl")
    ap.add_argument("--dup-threshold", type=float, default=0.92)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    raw = [json.loads(l) for l in open(args.raw, encoding="utf-8")]
    group = [json.loads(l) for l in open(args.group_kb, encoding="utf-8")]
    vocab = build_vocab([r["answer"] for r in group] + [it["answer"] for c in raw for it in c["output"].get("items", [])])
    counts, drops, recs = Counter(), Counter(), []
    for c in raw:
        for k, it in enumerate(c["output"].get("items", [])):
            q, a = sudanize(it["question"], vocab, counts).strip(), sudanize(it["answer"], vocab, counts).strip()
            blob = " ".join([q, a, *it.get("facts", [])])
            if len(q) < 10 or len(a) < 40:
                drops["too_short"] += 1; continue
            if FOREIGN_SCRIPT_RE.search(blob):
                drops["foreign_script"] += 1; continue
            if PLACEHOLDER_RE.search(blob) or pii.find_pii(blob):
                drops["pii_or_placeholder"] += 1; continue
            recs.append({"id": f"off-{c['chunk_id']}-{k}", "source": "official", "site": c["site"],
                         "date": c["fetched"], "topic": it["topic"], "question": q, "answer": a,
                         "facts": [f.strip() for f in it.get("facts", []) if f.strip()],
                         "german_terms": sorted({t.strip() for t in it.get("german_terms", []) if t.strip()}),
                         "links": [c["url"]], "confidence": "high", "time_sensitive": it.get("time_sensitive", False),
                         "chunk_id": c["chunk_id"], "also_seen": []})

    # near-duplicate questions within the official set (same fact on several pages)
    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer("BAAI/bge-m3", device=args.device).encode(
        [r["question"] for r in recs], normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    kept: list[int] = []
    for i in range(len(recs)):
        if kept and float((emb[kept] @ emb[i]).max()) >= args.dup_threshold:
            drops["duplicate"] += 1; continue
        kept.append(i)
    recs = [recs[i] for i in kept]

    with open(args.out, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.all, "w", encoding="utf-8") as f:
        for r in group + recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"official records: {len(recs)} | dropped: {dict(drops)} | dialect fixes: {dict(counts)}")
    print(f"topics: {dict(Counter(r['topic'] for r in recs).most_common())}")
    print(f"combined KB: {len(group)} group + {len(recs)} official = {len(group) + len(recs)} -> {args.all}")


if __name__ == "__main__":
    main()
