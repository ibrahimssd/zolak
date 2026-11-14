"""Audit every knowledge-base record against the anonymized discussion it was distilled from (teacher LLM).

For each record the teacher lists the claims of the answer and facts and marks each one supported or not by
the source discussion (thread or info post). Verdicts:
  keep    every claim is supported                      -> record unchanged
  revise  some claims unsupported / meaning errors      -> answer + facts rewritten without them
  drop    the main answer is not in the discussion      -> record removed
Links are kept only when they appear in the source text.

Input : data/processed/kb.jsonl (+ threads.jsonl, info_posts.jsonl)
Output: data/processed/kb_audited.jsonl, qa_audited.jsonl, eval_audited.jsonl (same eval ids, minus drops),
        data/interim/audit.jsonl (teacher verdicts, for review)
Usage:
    source scripts/env.sh
    CUDA_VISIBLE_DEVICES=0,1,2,3 $ZOLAK_VENVS/vllm/bin/python -m src.rag.audit_kb
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

from src.ingest.distill import POST_TMPL, TEACHER, format_thread

SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {"type": "array", "items": {"type": "object", "properties": {
            "text": {"type": "string"}, "supported": {"type": "boolean"}}, "required": ["text", "supported"]}},
        "verdict": {"type": "string", "enum": ["keep", "revise", "drop"]},
        "question": {"type": "string"},
        "answer": {"type": "string"},
        "facts": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["claims", "verdict", "question", "answer", "facts"],
}

SYSTEM = """You check a chatbot knowledge-base record against the WhatsApp discussion it was distilled from.
The record (question, answer, facts) must say ONLY what members said in the discussion.

1. "claims": split the record's answer and facts into short atomic claims. For each claim, "supported" is true
   only if the discussion states it or clearly implies it (same numbers, durations, offices, places, dates).
   General advice that nobody in the discussion gave ("check with the embassy") is NOT supported.
2. "verdict":
   - "keep"   all claims supported and the answer really answers the question.
   - "revise" some claims are unsupported, wrong, or change the meaning (wrong number, wrong office, a word that
              changes the sense), but the main answer is in the discussion.
   - "drop"   the discussion does not answer the question, or most claims are unsupported.
3. Only for "revise": write the corrected "question" (if it needs it), "answer" and "facts": remove the
   unsupported parts and fix the wrong ones, keep EVERY supported detail, numbers and German terms exactly.
   Same style: Sudanese colloquial Arabic. Use dialect words only where the meaning fits: «داير» means
   want/need and never "takes (time)"; for durations say «بتاخد». No names or personal details.
   For "keep" and "drop" return empty strings and an empty facts list."""


def _bare(url: str) -> str:
    return re.sub(r"^https?://(www\.)?", "", url).rstrip("/").lower()


def link_in_source(url: str, source: str) -> bool:
    """The link (ignoring scheme, "www." and a trailing slash) appears in the discussion."""
    return _bare(url) in re.sub(r"https?://(www\.)?", "", source).lower()


def record_text(r: dict) -> str:
    facts = "\n".join(f"- {f}" for f in r.get("facts", []))
    return f"question: {r['question']}\nanswer: {r['answer']}\nfacts:\n{facts}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proc", default="data/processed")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="pilot on the first N records")
    args = ap.parse_args()
    proc = Path(args.proc)

    kb = [json.loads(l) for l in (proc / "kb.jsonl").open(encoding="utf-8")]
    if args.limit:
        kb = kb[: args.limit]
    sources: dict[str, str] = {}
    for l in (proc / "threads.jsonl").open(encoding="utf-8"):
        t = json.loads(l)
        sources[t["thread_id"]] = format_thread(t)
    for l in (proc / "info_posts.jsonl").open(encoding="utf-8"):
        p = json.loads(l)
        sources[p["post_id"]] = POST_TMPL.format(date=p["date"], body=p["text"][:9000])
    missing = [r["id"] for r in kb if r["id"] not in sources]
    if missing:
        print(f"warning: {len(missing)} records without a source (kept unchanged): {missing[:5]}")

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

    jobs = [r for r in kb if r["id"] in sources]
    convs = [[{"role": "system", "content": SYSTEM},
              {"role": "user", "content": f"DISCUSSION:\n{sources[r['id']]}\n\nRECORD:\n{record_text(r)}"}]
             for r in jobs]
    llm = LLM(model=TEACHER, tensor_parallel_size=args.tp, max_model_len=16384, gpu_memory_utilization=0.90,
              enable_prefix_caching=True)
    params = SamplingParams(temperature=0.0, max_tokens=1800, guided_decoding=GuidedDecodingParams(json=SCHEMA))
    t0 = time.time()
    outs = llm.chat(convs, params, use_tqdm=True)

    from src.ingest import pii
    from src.ingest.postprocess import FOREIGN_SCRIPT_RE, PLACEHOLDER_RE, build_vocab, sudanize
    from src.rag.build_summaries import tidy
    vocab = build_vocab([r["answer"] for r in kb])
    verdicts, fixes, review = Counter(), Counter(), []
    audited = {r["id"]: r for r in kb}   # records without a source stay as they are
    claims_total = claims_ok = 0
    for r, o in zip(jobs, outs):
        try:
            a = json.loads(o.outputs[0].text)
        except json.JSONDecodeError:
            verdicts["parse_error_kept"] += 1
            continue
        claims = a.get("claims", [])
        claims_total += len(claims)
        claims_ok += sum(bool(c.get("supported")) for c in claims)
        v = a["verdict"]
        review.append({"id": r["id"], "verdict": v, "unsupported": [c["text"] for c in claims if not c["supported"]],
                       "old": r["answer"], "new": a.get("answer", "")})
        new = dict(r)
        new["links"] = [u for u in r.get("links", []) if link_in_source(u, sources[r["id"]])]
        if v == "drop":
            audited.pop(r["id"])
        elif v == "revise":
            q = tidy(sudanize(a["question"] or r["question"], vocab, fixes))
            ans = tidy(sudanize(a["answer"], vocab, fixes))
            facts = [f for f in a["facts"] if f.strip()]
            blob = " ".join([q, ans, *facts])
            if len(ans) < 40 or FOREIGN_SCRIPT_RE.search(blob) or PLACEHOLDER_RE.search(blob) or pii.find_pii(blob):
                ok = sum(c["supported"] for c in claims) / max(len(claims), 1)
                if ok < 0.8:
                    audited.pop(r["id"])
                    v = "revise_failed_drop"
                else:
                    audited[r["id"]] = new
                    v = "revise_failed_keep"
            else:
                new.update(question=q, answer=ans, facts=facts, audited="revised")
                audited[r["id"]] = new
        else:
            new["audited"] = "kept"
            audited[r["id"]] = new
        verdicts[v] += 1

    kept = list(audited.values())
    eval_ids = {json.loads(l)["id"] for l in (proc / "eval.jsonl").open(encoding="utf-8")}
    for name, rows in (("kb_audited.jsonl", kept), ("eval_audited.jsonl", [r for r in kept if r["id"] in eval_ids]),
                       ("qa_audited.jsonl", [r for r in kept if r["id"] not in eval_ids])):
        with (proc / name).open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    Path("data/interim").mkdir(parents=True, exist_ok=True)
    with open("data/interim/audit.jsonl", "w", encoding="utf-8") as f:
        for x in review:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"audited {len(jobs)} records in {time.time() - t0:.0f}s: {dict(verdicts)}")
    print(f"claims supported: {claims_ok}/{claims_total} ({claims_ok / max(claims_total, 1):.1%})")
    print(f"KB {len(kb)} -> {len(kept)} records | eval {len([r for r in kept if r['id'] in eval_ids])}/{len(eval_ids)}")


if __name__ == "__main__":
    main()
