"""Summarize the group's responses per question cluster (teacher LLM on local GPUs).

1. Embed the questions of the group KB (bge-m3) and cluster them (agglomerative, cosine distance <= 1 - --sim).
2. Member check (teacher): records that answer a different question than the cluster's main one are removed;
   a cluster with fewer than 2 remaining records gets no summary. This runs BEFORE the summary is written, so
   the summary's text, count and date range only cover the members that stay.
3. For each remaining cluster the teacher writes ONE consolidated answer in Sudanese dialect that uses
   ONLY those records: how many discussions and when, what most people said, where they disagreed, concrete
   numbers with their dates.
4. Output summary records (source="summary") that sit in the KB next to the individual records.

Input : data/processed/kb_audited.jsonl (group knowledge only; pass with --kb)
Output: data/processed/kb_summaries.jsonl, data/processed/kb_group.jsonl (individual + summaries, for the index),
        data/interim/summary_check.jsonl (member-check verdicts, for review)

Usage:
    source scripts/env.sh
    CUDA_VISIBLE_DEVICES=0,1,2,3 $ZOLAK_VENVS/vllm/bin/python -m src.rag.build_summaries --embed-device cuda
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

from src.ingest.distill import TEACHER, TOPICS

SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "answer": {"type": "string"},
        "facts": {"type": "array", "items": {"type": "string"}},
        "german_terms": {"type": "array", "items": {"type": "string"}},
        "topic": {"type": "string", "enum": TOPICS},
    },
    "required": ["question", "answer", "facts", "german_terms", "topic"],
}

SYSTEM = """You summarize what members of a Sudanese students' WhatsApp group in Germany said about ONE question.
You get several dated Q&A records distilled from the group. Write ONE consolidated record.

Rules:
1. Use ONLY the given records. No outside knowledge, no new numbers, offices or links.
2. "question": one general question that covers all records (Sudanese dialect).
3. "answer": 3-7 sentences in natural SUDANESE dialect (عشان، داير، زول، شنو، كدا، برضو، بتقدر، ما بتحتاج). Start by
   saying it summarizes the group, e.g. «في {n} نقاشات في القروب بين {from} و{to}، الناس قالو …». Give what most
   people agreed on, then differences or exceptions («بس في ناس قالو …»), and concrete amounts/durations WITH the
   month/year they were mentioned when they differ over time. No Egyptian/Gulf/Levantine forms (مش، مو، هذي، -ش).
   Use dialect words only where the meaning fits: «داير» means want/need and never "takes (time)"; for
   durations say «بتاخد». Keep German terms in Latin script. No names or personal details.
4. "facts": 2-6 short atomic facts in Arabic, each with its date if it may change.
5. "german_terms": search keywords. "topic": the best topic label."""

CHECK_SCHEMA = {  # property order matters: name the main question first, then judge each record against it
    "type": "object",
    "properties": {
        "main_question": {"type": "string"},
        "off_topic": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["main_question", "off_topic"],
}

CHECK_SYSTEM = """You check a cluster of dated Q&A records from a Sudanese students' WhatsApp group in Germany.
The records were grouped automatically because their questions look similar; some may not belong together.
1. "main_question": in English, the ONE question that most records answer.
2. "off_topic": the numbers k of the records [k] that answer a DIFFERENT question: another sub-question, another
   country, office, document, visa or permit type, or a different situation, so that using their answer for the
   main question would be wrong or misleading. Records that answer the main question with different details,
   numbers, dates or opinions are NOT off-topic. Return [] when all records fit."""

MONTHS = ["يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو", "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر"]


def ar_month(d: str) -> str:
    return f"{MONTHS[int(d[5:7]) - 1]} {d[:4]}"


_MON = "(?:" + "|".join(MONTHS) + r") \d{4}"
TIDY = [
    (re.compile(r"في 2 نقاشات"), "في نقاشين"),
    (re.compile(rf"بين ({_MON}) ?و ?\1"), r"في \1"),            # "بين مارس 2025 ومارس 2025" -> "في مارس 2025"
    (re.compile(rf"بين ({_MON}) ?و ?({_MON})"), r"بين \1 و\2"),
    (re.compile(r"^شلونك[،,]?\s*"), ""),
    (re.compile(r"شلون"), "كيف"),
]


def tidy(text: str) -> str:
    for pat, rep in TIDY:
        text = pat.sub(rep, text)
    return text.strip()


def member_links(records: list[dict], members: list[int], n: int = 3) -> list[str]:
    """Links the members shared (newest first, deduplicated)."""
    return list(dict.fromkeys(u for i in members for u in records[i].get("links", [])))[:n]


def cluster(records: list[dict], sim: float, device: str) -> list[list[int]]:
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import AgglomerativeClustering

    emb = SentenceTransformer("BAAI/bge-m3", device=device).encode(
        [r["question"] for r in records], normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    labels = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                     distance_threshold=1 - sim).fit_predict(emb)
    groups: dict[int, list[int]] = {}
    for i, l in enumerate(labels):
        groups.setdefault(int(l), []).append(i)
    return [g for g in groups.values() if len(g) >= 2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kb", default="data/processed/kb.jsonl")
    ap.add_argument("--out", default="data/processed/kb_summaries.jsonl")
    ap.add_argument("--combined", default="data/processed/kb_group.jsonl")
    ap.add_argument("--sim", type=float, default=0.78, help="min question similarity inside a cluster")
    ap.add_argument("--max-members", type=int, default=12)
    ap.add_argument("--embed-device", default="cpu")
    ap.add_argument("--tp", type=int, default=4)
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.kb, encoding="utf-8")]
    clusters = cluster(records, args.sim, args.embed_device)
    clusters = [sorted(c, key=lambda i: records[i]["date"], reverse=True)[: args.max_members] for c in clusters]
    print(f"{len(records)} records -> {len(clusters)} clusters with >=2 members "
          f"(sizes: {Counter(min(len(c), 10) for c in clusters).most_common()})")

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

    def prompt(c: list[int]) -> str:
        dates = sorted(records[i]["date"] for i in c)
        body = "\n\n".join(f"[{k}] ({records[i]['date'][:7]}) س: {records[i]['question']}\nج: {records[i]['answer']}"
                           for k, i in enumerate(c, 1))
        return f"n={len(c)} discussions, from {ar_month(dates[0])} to {ar_month(dates[-1])}\n\n{body}"

    llm = LLM(model=TEACHER, tensor_parallel_size=args.tp, max_model_len=12288, gpu_memory_utilization=0.90,
              enable_prefix_caching=True)
    t0 = time.time()

    # member check: drop records that answer a different question, before the summary is written
    check_params = SamplingParams(temperature=0.0, max_tokens=400,
                                  guided_decoding=GuidedDecodingParams(json=CHECK_SCHEMA))
    checks = llm.chat([[{"role": "system", "content": CHECK_SYSTEM}, {"role": "user", "content": prompt(c)}]
                       for c in clusters], check_params, use_tqdm=True)
    kept_clusters, review, n_removed = [], [], 0
    for c, o in zip(clusters, checks):
        try:
            chk = json.loads(o.outputs[0].text)
            off = {int(k) for k in chk.get("off_topic", []) if 1 <= int(k) <= len(c)}
        except (json.JSONDecodeError, TypeError, ValueError):
            chk, off = {}, set()
        keep = [i for k, i in enumerate(c, 1) if k not in off]
        n_removed += len(c) - len(keep)
        review.append({"main_question": chk.get("main_question", ""), "members": [records[i]["id"] for i in c],
                       "off_topic": [records[c[k - 1]]["id"] for k in sorted(off)],
                       "off_topic_questions": [records[c[k - 1]]["question"] for k in sorted(off)],
                       "summarized": len(keep) >= 2})
        if len(keep) >= 2:
            kept_clusters.append(keep)
    Path("data/interim").mkdir(parents=True, exist_ok=True)
    with open("data/interim/summary_check.jsonl", "w", encoding="utf-8") as f:
        for x in review:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"member check: removed {n_removed} off-topic members; {len(clusters) - len(kept_clusters)} clusters "
          f"left with < 2 members (no summary); {len(kept_clusters)} clusters to summarize")
    clusters = kept_clusters

    params = SamplingParams(temperature=0.2, top_p=0.9, max_tokens=1200,
                            guided_decoding=GuidedDecodingParams(json=SCHEMA))
    outs = llm.chat([[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt(c)}] for c in clusters],
                    params, use_tqdm=True)
    from src.ingest import pii
    from src.ingest.postprocess import FOREIGN_SCRIPT_RE, PLACEHOLDER_RE, build_vocab, sudanize
    vocab = build_vocab([r["answer"] for r in records])
    fixes, drops, summaries = Counter(), Counter(), []
    for n, (c, o) in enumerate(zip(clusters, outs)):
        try:
            s = json.loads(o.outputs[0].text)
        except json.JSONDecodeError:
            drops["json"] += 1; continue
        q, a = tidy(sudanize(s["question"], vocab, fixes)), tidy(sudanize(s["answer"], vocab, fixes))
        blob = " ".join([q, a, *s["facts"]])
        if len(a) < 60 or FOREIGN_SCRIPT_RE.search(blob) or PLACEHOLDER_RE.search(blob) or pii.find_pii(blob):
            drops["gate"] += 1; continue
        dates = sorted(records[i]["date"] for i in c)
        summaries.append({"id": f"sum-{n}", "source": "summary", "date": dates[-1], "date_from": dates[0],
                          "n_discussions": len(c), "members": [records[i]["id"] for i in c], "topic": s["topic"],
                          "question": q, "answer": a, "facts": s["facts"], "german_terms": s["german_terms"],
                          "links": member_links(records, c), "confidence": "high" if len(c) >= 3 else "medium", "time_sensitive": True})
    with open(args.out, "w", encoding="utf-8") as f:
        for r in summaries:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.combined, "w", encoding="utf-8") as f:
        for r in records + summaries:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(summaries)} summaries in {time.time() - t0:.0f}s | dropped {dict(drops)} | dialect fixes {dict(fixes)}")
    print(f"combined group KB: {len(records)} records + {len(summaries)} summaries -> {args.combined}")


if __name__ == "__main__":
    main()
