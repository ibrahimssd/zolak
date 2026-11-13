"""Step 4b — clean, gate, dedup and split teacher outputs.

Gates (a record is dropped if any fails):
  useful == True, confidence != low, answer >= 40 chars, question >= 10 chars,
  no placeholder ([اسم] [رقم] [رابط] [ايميل] @مستخدم), no PII pattern, no member full name.
Links: only allowlisted official URLs survive.
Dedup: bge-m3 cosine >= --dup-threshold between questions -> keep best (confidence, then newest);
       the dates of dropped duplicates are kept in `also_seen` (shows how often/when it was asked).
Split: --n-eval high-confidence items stratified by topic -> eval.jsonl (also in KB, NOT in qa.jsonl).

Input : data/interim/distill_raw.jsonl (+ data/interim/pii_lexicon.json)
Output: data/processed/kb.jsonl   retrieval knowledge base (all deduped items)
        data/processed/qa.jsonl   SFT source (kb minus eval)
        data/processed/eval.jsonl held-out evaluation questions

Usage: python -m src.ingest.postprocess [--input data/interim/distill_raw.jsonl]
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from . import pii

PLACEHOLDER_RE = re.compile(r"\[(?:اسم|رقم|رابط|ايميل)\]|@مستخدم")
# Qwen sometimes code-switches into CJK etc. Arabic, Latin (incl. German) and symbols are fine.
FOREIGN_SCRIPT_RE = re.compile(
    "[Ѐ-ӿ֐-׿ऀ-෿฀-࿿぀-ヿ㐀-鿿가-힯豈-﫿]"
)
CONF_RANK = {"high": 2, "medium": 1, "low": 0}


def load_name_re(lex_path: Path):
    if not lex_path.exists():
        return None
    lex = json.loads(lex_path.read_text(encoding="utf-8"))
    names = [n for n in lex["full_names"] if len(n) >= 6 and " " in n]
    return re.compile("|".join(re.escape(n) for n in names), re.I) if names else None


def gate(o: dict, name_re) -> str | None:
    """Return a drop reason or None."""
    if not o or not o.get("useful"):
        return "not_useful"
    if o.get("confidence") == "low":
        return "low_confidence"
    q, a = o.get("question", "").strip(), o.get("answer", "").strip()
    if len(q) < 10 or len(a) < 40:
        return "too_short"
    blob = " ".join([q, a, *o.get("facts", [])])
    if FOREIGN_SCRIPT_RE.search(blob):
        return "foreign_script"
    if PLACEHOLDER_RE.search(blob):
        return "placeholder"
    if pii.find_pii(blob):
        return "pii_pattern"
    if name_re and name_re.search(blob):
        return "member_name"
    return None


def dedup(items: list[dict], threshold: float, device: str) -> list[dict]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("BAAI/bge-m3", device=device)
    emb = model.encode([it["question"] for it in items], normalize_embeddings=True, batch_size=64,
                       show_progress_bar=False)
    order = sorted(range(len(items)),
                   key=lambda i: (CONF_RANK[items[i]["confidence"]], items[i]["date"]), reverse=True)
    kept_idx: list[int] = []
    for i in order:
        if kept_idx:
            sims = emb[kept_idx] @ emb[i]
            j = int(sims.argmax())
            if sims[j] >= threshold:
                items[kept_idx[j]]["also_seen"].append(items[i]["date"])
                continue
        kept_idx.append(i)
    return [items[i] for i in sorted(kept_idx, key=lambda i: items[i]["date"])]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="data/interim/distill_raw.jsonl")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--dup-threshold", type=float, default=0.90)
    ap.add_argument("--n-eval", type=int, default=150)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    name_re = load_name_re(Path("data/interim/pii_lexicon.json"))
    raw = [json.loads(l) for l in open(args.input, encoding="utf-8")]
    drops: Counter = Counter()
    items: list[dict] = []
    for r in raw:
        o = r["output"]
        reason = "json_error" if r["error"] else gate(o, name_re)
        if reason:
            drops[reason] += 1
            continue
        items.append({
            "id": r["src_id"], "source": r["kind"], "date": r["date"], "topic": o["topic"],
            "question": o["question"].strip(), "answer": o["answer"].strip(),
            "facts": [f.strip() for f in o["facts"] if f.strip()],
            "german_terms": sorted({t.strip() for t in o["german_terms"] if t.strip()}),
            "links": [u for u in o["official_links"] if pii.url_allowed(u)],
            "confidence": o["confidence"], "time_sensitive": o["time_sensitive"], "also_seen": [],
        })
    n_gated = len(items)
    items = dedup(items, args.dup_threshold, args.device)

    # stratified eval split (high confidence only)
    rnd = random.Random(0)
    by_topic = defaultdict(list)
    for it in items:
        if it["confidence"] == "high":
            by_topic[it["topic"]].append(it)
    eval_ids: set[str] = set()
    pool_total = sum(len(v) for v in by_topic.values())
    for topic, pool in by_topic.items():
        k = max(1, round(args.n_eval * len(pool) / max(pool_total, 1)))
        eval_ids.update(it["id"] for it in rnd.sample(pool, min(k, len(pool))))

    out = Path(args.out_dir)
    with (out / "kb.jsonl").open("w", encoding="utf-8") as fkb, \
            (out / "qa.jsonl").open("w", encoding="utf-8") as fqa, \
            (out / "eval.jsonl").open("w", encoding="utf-8") as fev:
        for it in items:
            line = json.dumps(it, ensure_ascii=False) + "\n"
            fkb.write(line)
            (fev if it["id"] in eval_ids else fqa).write(line)

    print(f"raw {len(raw)} -> passed gates {n_gated} -> after dedup {len(items)} "
          f"(qa {len(items) - len(eval_ids)}, eval {len(eval_ids)})")
    print("dropped:", dict(drops.most_common()))
    print("topics:", dict(Counter(it["topic"] for it in items).most_common()))
    print("confidence:", dict(Counter(it["confidence"] for it in items)),
          "time_sensitive:", sum(it["time_sensitive"] for it in items))


if __name__ == "__main__":
    main()
