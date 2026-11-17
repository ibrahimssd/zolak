"""Retrieval experiments: do a re-ranker and follow-up handling put an entry that answers the question in the top 2?

Gold-label hit@2 understates retrieval: many askers' messages are not standalone questions, and another entry often
answers just as well as the labelled one. So the teacher also judges, per query and configuration, whether any of
the two entries answers the question (answerable@2), and whether the query is a real question at all.

Query sets
  real      askers' real wording of dev + held-out records (gold = the record)             configs: base, rerank
  followup  askers' follow-up messages in those threads, with their first question as     configs: alone, combined,
            history (gold = the record)                                                             combined_rerank
  detail    topic-card questions + 👎 questions (data/processed/detail_tests.jsonl)          configs: base, rerank
  feedback  every question in the feedback dataset (data/interim/feedback, no personal data) configs: base, rerank

  cands  (GPU) top-2 entries per query/config -> data/interim/retrieval_lab/cands.jsonl, plus re-ranker CPU latency
  judge  (teacher, 4 GPUs) -> printed table and $ZOLAK_STORAGE/logs/retrieval_lab.json

Usage:
    source scripts/env.sh
    CUDA_VISIBLE_DEVICES=2 LOCAL_EMBED=$ZOLAK_MODELS/zolak-embed-v1 $ZOLAK_VENVS/vllm/bin/python -m src.rag.retrieval_lab cands
    CUDA_VISIBLE_DEVICES=0,1,4,5 $ZOLAK_VENVS/vllm/bin/python -m src.rag.retrieval_lab judge
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "space"))
from retriever import HybridRetriever, doc_text, retrieval_query  # noqa: E402

from src.rag.finetune_embedder import PROC, QUESTION_CUE, asker_wording, hits_gold, load_json, query_sets  # noqa: E402

OUT = Path("data/interim/retrieval_lab")
RERANKER = "BAAI/bge-reranker-v2-m3"
ARABIC = re.compile(r"[؀-ۿ]")

ISQ_SCHEMA = {"type": "object", "properties": {"is_question": {"type": "boolean"}}, "required": ["is_question"]}
ISQ_PROMPT = """Below is a message from a WhatsApp group of Sudanese students in Germany. "is_question" is true if the
message itself asks for information that a help chatbot could answer (for example about visas, residence, study,
jobs, money, housing or travel), and it is understandable without the rest of the chat. It is false for replies,
comments, thanks, jokes, announcements, or fragments that only make sense inside the conversation.

MESSAGE: {q}"""


def followup_set(eval_like: set[str]) -> list[tuple[str, str, str]]:
    """(follow-up message, the asker's first question, gold record id) for threads of the given records."""
    out = []
    for l in (PROC / "threads.jsonl").open(encoding="utf-8"):
        t = json.loads(l)
        if t["thread_id"] not in eval_like:
            continue
        first = asker_wording(t)
        for m in t["messages"]:
            text = " ".join(m["text"].split())
            if m["role"] == "السائل" and m.get("part") == "reply" and 2 <= len(text.split()) <= 25 \
                    and QUESTION_CUE.search(text):
                out.append((text, first, t["thread_id"]))
                break   # one follow-up per thread
    return out


def feedback_questions() -> list[str]:
    from src.ingest import pii
    qs = []
    for p in sorted(Path("data/interim/feedback/data").glob("*.jsonl")):
        for l in p.open(encoding="utf-8"):
            q = " ".join((json.loads(l).get("question") or "").split())
            if q and q not in qs and not pii.find_pii(q) and len(ARABIC.findall(q)) >= 0.5 * len(re.sub(r"\W", "", q)):
                qs.append(q)
    return qs


def cmd_cands(args) -> None:
    from sentence_transformers import CrossEncoder

    kb = load_json(Path("data/processed/kb_group.jsonl"))
    eval_ids = {r["id"] for r in load_json(PROC / "eval.jsonl")}
    sets = query_sets(kb, eval_ids)
    real = sets["dev_real"] + sets["eval_real"]
    fu = followup_set({g for _, g in real})
    detail = [t["query"] for t in load_json(PROC / "detail_tests.jsonl")]
    fb = feedback_questions()

    retr = HybridRetriever(args.index, device="cuda")
    rer = CrossEncoder(RERANKER, max_length=args.max_length, device="cuda")
    rows = []

    def add(set_name, query, gold, config, recs, history=""):
        rows.append({"set": set_name, "query": query, "history": history, "gold": gold, "config": config,
                     "ctx": recs, "hit": bool(gold) and any(hits_gold(r, gold) for r in recs)})

    for config, top in (("base", None), ("rerank", rer)):
        retr.set_reranker(top, args.rerank_top)
        for q, g in real:
            add("real", q, g, config, retr.top_entries(q, 2))
        for q in detail:
            add("detail", q, None, config, retr.top_entries(q, 2))
        for q in fb:
            add("feedback", q, None, config, retr.top_entries(q, 2))
    for config, top, combine in (("alone", None, False), ("combined", None, True), ("combined_rerank", rer, True)):
        retr.set_reranker(top, args.rerank_top)
        for q, first, g in fu:
            sq = retrieval_query(q, [{"role": "user", "content": first}]) if combine else q
            add("followup", q, g, config, retr.top_entries(sq, 2), history=first)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "cands.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    hit = defaultdict(list)
    for r in rows:
        if r["gold"]:
            hit[(r["set"], r["config"])].append(r["hit"])
    print({f"{s}/{c}": f"hit@2 {sum(v) / len(v):.3f} (n={len(v)})" for (s, c), v in hit.items()})
    print(f"followup: combined query used for {sum(retrieval_query(q, [{'role': 'user', 'content': f}]) != q for q, f, _ in fu)}"
          f"/{len(fu)} follow-ups")

    # re-ranker latency on CPU with 2 threads, as on the free cpu-basic Space
    import torch
    torch.set_num_threads(2)
    cpu = CrossEncoder(RERANKER, max_length=args.max_length, device="cpu")
    for n_cands, max_len in ((args.rerank_top, args.max_length), (5, args.max_length), (args.rerank_top, 256)):
        cpu.max_length = max_len
        ts = []
        for q, _ in real[:15]:
            cands = retr.candidates(q, n_cands)
            t0 = time.time()
            cpu.predict([(q, doc_text(r)) for r in cands], show_progress_bar=False)
            ts.append(time.time() - t0)
        ts.sort()
        print(f"re-ranker on 2 CPU threads, {n_cands} candidates, max_length {max_len}: "
              f"median {ts[len(ts) // 2]:.2f}s, max {ts[-1]:.2f}s")


def cmd_judge(args) -> None:
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

    from src.ingest.distill import TEACHER
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "space"))
    from prompts import format_context
    from src.train.eval_rag import CTX_PROMPT, CTX_SCHEMA

    rows = load_json(OUT / "cands.jsonl")
    llm = LLM(model=TEACHER, tensor_parallel_size=args.tp, max_model_len=12288, gpu_memory_utilization=0.90,
              enable_prefix_caching=True)

    def run(prompts: list[str], schema: dict) -> list[dict | None]:
        params = SamplingParams(temperature=0.0, max_tokens=30, guided_decoding=GuidedDecodingParams(json=schema))
        res = []
        for o in llm.chat([[{"role": "user", "content": p}] for p in prompts], params, use_tqdm=True):
            try:
                res.append(json.loads(o.outputs[0].text))
            except json.JSONDecodeError:
                res.append(None)
        return res

    real_qs = sorted({r["query"] for r in rows if r["set"] == "real"})
    isq = dict(zip(real_qs, [bool(v and v["is_question"]) for v in run([ISQ_PROMPT.format(q=q) for q in real_qs],
                                                                        ISQ_SCHEMA)]))

    def question_text(r):
        return f"{r['history']}\n(سؤال متابعة): {r['query']}" if r["set"] == "followup" else r["query"]

    keys = sorted({(question_text(r), tuple(c["id"] for c in r["ctx"])) for r in rows})
    by_key = {(question_text(r), tuple(c["id"] for c in r["ctx"])): r["ctx"] for r in rows}
    verdicts = run([CTX_PROMPT.format(ctx=format_context(by_key[k]), q=k[0]) for k in keys], CTX_SCHEMA)
    answers = {k: bool(v and v["context_answers"]) for k, v in zip(keys, verdicts)}

    report = defaultdict(dict)
    groups = defaultdict(list)
    for r in rows:
        r["answerable"] = answers[(question_text(r), tuple(c["id"] for c in r["ctx"]))]
        groups[(r["set"], r["config"])].append(r)
    pct = lambda xs: round(100 * sum(xs) / len(xs), 1) if xs else None  # noqa: E731
    for (s, c), rs in sorted(groups.items()):
        rep = {"n": len(rs), "answerable@2_%": pct([r["answerable"] for r in rs])}
        if any(r["gold"] for r in rs):
            rep["hit@2_%"] = pct([r["hit"] for r in rs])
        if s == "real":
            qs = [r for r in rs if isq.get(r["query"])]
            rep["n_real_questions"] = len(qs)
            rep["answerable@2_real_questions_%"] = pct([r["answerable"] for r in qs])
            rep["hit@2_real_questions_%"] = pct([r["hit"] for r in qs])
        report[s][c] = rep
    log = Path(os.environ.get("ZOLAK_STORAGE", ".")) / "logs" / "retrieval_lab.json"
    log.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    with (OUT / "judged.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if k != "ctx"} | {"ctx_ids": [c["id"] for c in r["ctx"]]},
                               ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cands")
    c.add_argument("--index", default="data/index")
    c.add_argument("--rerank-top", type=int, default=8)
    c.add_argument("--max-length", type=int, default=512)
    j = sub.add_parser("judge")
    j.add_argument("--tp", type=int, default=4)
    args = ap.parse_args()
    {"cands": cmd_cands, "judge": cmd_judge}[args.cmd](args)


if __name__ == "__main__":
    main()
