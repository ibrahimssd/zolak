"""Phase 5 — compare chatbot models (v2 vs v4) on the production pipeline, judged by the teacher.

  prep   test set + contexts, once for all models (same entries for every model):
           eval_distilled   the held-out eval questions (gold = their record)
           eval_<style>     teacher paraphrases of them (casual / msa / mixed)
           unanswerable     near-miss and off-topic questions the group never answered (eval split)
         context = HybridRetriever.top_entries(q, 2), exactly as the app (no threshold, nothing hidden)
  gen    one model answers every test: build_messages + greedy decoding (repetition penalty 1.1, 300 new tokens,
         as the GPU worker) + the app's link guard. The raw answer is kept for the link metric.
  judge  the teacher decides once per test whether the entries answer it, then grades every answer blind:
         claims supported by the entries, refusal, wrong office, correctness vs the gold answer.
         Prints the comparison table and writes $ZOLAK_STORAGE/logs/eval_rag.json.

Usage:
    source scripts/env.sh
    $ZOLAK_VENVS/vllm/bin/python -m src.train.eval_rag prep
    CUDA_VISIBLE_DEVICES=2 $ZOLAK_VENVS/vllm/bin/python -m src.train.eval_rag gen --name v2 --model $ZOLAK_MODELS/zolak-allam-v2
    CUDA_VISIBLE_DEVICES=2 $ZOLAK_VENVS/vllm/bin/python -m src.train.eval_rag gen --name v4 --model $ZOLAK_MODELS/zolak-allam-v4
    CUDA_VISIBLE_DEVICES=0,1,4,5 $ZOLAK_VENVS/vllm/bin/python -m src.train.eval_rag judge --names v2 v4
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "space"))
from prompts import URL_RE, build_messages, format_context, guard_links  # noqa: E402

PROC = Path("data/processed")
OUT = Path("data/interim/eval_rag")

CTX_SCHEMA = {"type": "object", "properties": {"context_answers": {"type": "boolean"}},
              "required": ["context_answers"]}
CTX_PROMPT = """Below are entries from a Sudanese students' WhatsApp group in Germany and a student's question.
"context_answers" is true if at least one entry answers the question (fully, or its main point). An entry about a
different question, country, office, document, permit type or situation does NOT count, even if it looks similar.

ENTRIES:
{ctx}

QUESTION: {q}"""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "refused": {"type": "boolean"},
        "claims": {"type": "array", "items": {"type": "object", "properties": {
            "text": {"type": "string"}, "supported": {"type": "boolean"}}, "required": ["text", "supported"]}},
        "wrong_office": {"type": "boolean"},
        "correctness": {"type": "string", "enum": ["correct", "partial", "wrong", "n/a"]},
        "entry_points": {"type": "array", "maxItems": 12, "items": {"type": "object", "properties": {
            "text": {"type": "string"}, "included": {"type": "boolean"}}, "required": ["text", "included"]}},
    },
    "required": ["refused", "claims", "wrong_office", "correctness", "entry_points"],
}
JUDGE_PROMPT = """You grade a reply of a chatbot for Sudanese students in Germany. The chatbot must answer ONLY from
the WhatsApp-group entries it was given, and must say that the group has no discussion about the question when
the entries do not answer it.

- "refused": true if the reply says the group has no discussion/information about the question (even if it then
  points to a related topic).
- "claims": the reply's atomic factual claims (numbers, durations, dates, offices, documents, steps, links,
  recommendations). "supported" is true only if the entries state it or clearly imply it. A date is supported when
  it matches the date shown next to an entry. The generic caution «اتأكد من الجهة الرسمية» is supported. The refusal
  statement itself is not a claim.
- "wrong_office": true if the reply names an office, authority, organisation or website that the entries do not
  name.
- "correctness": compare with the REFERENCE answer if one is given, else "n/a". "correct" = the reply gives the
  reference's main point without contradicting it; "partial" = some of it; "wrong" = contradicts it or misses it
  (a refusal is "wrong").
- "entry_points": the points in the ENTRIES that answer the question (steps, documents, amounts, durations, dates,
  places, conditions, where people disagreed), at most 12, each as a short phrase; "included" is true if the reply
  states that point. Empty list if the entries do not answer the question.

ENTRIES:
{ctx}

QUESTION: {q}

REFERENCE ANSWER: {gold}

REPLY:
{answer}"""


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8")] if path.exists() else []


def cmd_prep(args) -> None:
    from retriever import HybridRetriever

    ev = load(PROC / "eval_audited.jsonl") or load(PROC / "eval.jsonl")
    by_id = {r["id"]: r for r in ev}
    tests = [{"set": "eval_distilled", "query": r["question"], "gold": r["id"]} for r in ev]
    for p in load(PROC / "paraphrases.jsonl"):
        if p["id"] in by_id:
            tests.append({"set": f"eval_{p['style']}", "query": p["query"], "gold": p["id"]})
        elif p["id"] is None and p.get("split") == "eval":
            tests.append({"set": "unanswerable", "query": p["query"], "gold": None, "kind": p["style"]})
    # broad questions whose answers should be detailed (topic cards, 👎 questions); reported apart from "all"
    for t in load(PROC / "detail_tests.jsonl"):
        tests.append({"set": "detail", "query": t["query"], "gold": None, "kind": t["source"]})
    retr = HybridRetriever(args.index, device=args.device)
    if args.rerank:
        from sentence_transformers import CrossEncoder
        retr.set_reranker(CrossEncoder(args.rerank, max_length=512, device=args.device), args.rerank_top)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "tests.jsonl").open("w", encoding="utf-8") as f:
        for i, t in enumerate(tests):
            t["qid"] = i
            t["gold_answer"] = by_id[t["gold"]]["answer"] if t["gold"] else None
            t["ctx"] = retr.top_entries(t["query"], k=2)
            t["gold_in_ctx"] = bool(t["gold"]) and any(c["id"] == t["gold"] or t["gold"] in c.get("members", [])
                                                       for c in t["ctx"])
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    counts = defaultdict(int)
    for t in tests:
        counts[t["set"]] += 1
    print(f"{len(tests)} tests {dict(counts)}; gold retrieved in top-2: "
          f"{sum(t['gold_in_ctx'] for t in tests if t['gold'])}/{sum(bool(t['gold']) for t in tests)}")


def cmd_gen(args) -> None:
    from vllm import LLM, SamplingParams

    tests = load(OUT / "tests.jsonl")
    llm = LLM(model=args.model, tensor_parallel_size=1, max_model_len=4096, gpu_memory_utilization=0.85,
              dtype="bfloat16")
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, repetition_penalty=1.1)
    outs = llm.chat([build_messages(t["query"], t["ctx"]) for t in tests], params, use_tqdm=True)
    from dialect_fix import build_vocab, sudanize_text   # the app's dialect normalizer (vocabulary of the index)
    vocab = build_vocab([" ".join([r["question"], r["answer"], *r.get("facts", [])])
                         for r in load(Path(args.index) / "kb.jsonl")])
    with (OUT / f"answers_{args.name}.jsonl").open("w", encoding="utf-8") as f:
        for t, o in zip(tests, outs):
            raw = o.outputs[0].text.strip()   # the model's own text, before the app's normalizer and link guard
            answer = guard_links(sudanize_text(raw, vocab), t["ctx"], final=True)   # what the student sees
            f.write(json.dumps({"qid": t["qid"], "raw": raw, "answer": answer}, ensure_ascii=False) + "\n")
    print(f"{len(tests)} answers -> {OUT / f'answers_{args.name}.jsonl'}")


def summarize(rows: list[dict]) -> dict:
    """rows: test + judge fields for one model (one set or all)."""
    from src.train.dialect import score

    answered = [r for r in rows if not r["refused"]]
    covered = [r for r in rows if r["context_answers"]]
    uncovered = [r for r in rows if not r["context_answers"]]
    gold = [r for r in rows if r["gold"]]
    claims = [c for r in answered for c in r["claims"]]
    raw_urls = [(u.rstrip(".,؛!؟?/"), r) for r in rows for u in URL_RE.findall(r["raw"])]
    ctx_links = lambda r: {l.rstrip("/.") for c in r["ctx"] for l in c.get("links", [])}  # noqa: E731
    dial = [score(r["answer"]) for r in rows]
    raw_dial = [score(r["raw"]) for r in rows]
    pct = lambda xs: round(100 * sum(xs) / len(xs), 1) if xs else None  # noqa: E731
    return {
        "n": len(rows),
        "faithful_%": pct([not any(not c["supported"] for c in r["claims"]) for r in answered]),
        "claims_supported_%": pct([c["supported"] for c in claims]),
        "correctness_%": pct([{"correct": 1, "partial": 0.5}.get(r["correctness"], 0) for r in gold]),
        "refusal_acc_%": pct([r["refused"] == (not r["context_answers"]) for r in rows]),
        "false_refusal_%": pct([r["refused"] for r in covered]),
        "answered_uncovered_%": pct([not r["refused"] for r in uncovered]),
        "wrong_office_%": pct([r["wrong_office"] for r in answered]),
        "raw_links_from_ctx_%": pct([u in ctx_links(r) for u, r in raw_urls]),
        "n_raw_links": len(raw_urls),
        "dialect_sd_per_answer": round(st.mean(d["sd"] for d in dial), 2) if dial else None,
        # share of the entries' relevant points (steps, documents, amounts ...) the reply includes
        "completeness_%": pct([sum(p["included"] for p in r.get("entry_points", [])) / len(r["entry_points"])
                               for r in answered if r["context_answers"] and r.get("entry_points")]),
        "mean_words": round(st.mean(len(r["answer"].split()) for r in rows), 1) if rows else None,
        "non_sudanese_answers_%": pct([d["other"] > 0 for d in dial]),
        "non_sudanese_raw_model_%": pct([d["other"] > 0 for d in raw_dial]),   # before the app's normalizer
    }


def cmd_judge(args) -> None:
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
    from src.ingest.distill import TEACHER

    tests = {t["qid"]: t for t in load(OUT / "tests.jsonl")}
    answers = {n: {a["qid"]: a for a in load(OUT / f"answers_{n}.jsonl")} for n in args.names}
    llm = LLM(model=TEACHER, tensor_parallel_size=args.tp, max_model_len=12288, gpu_memory_utilization=0.90,
              enable_prefix_caching=True)

    def run(prompts: list[str], schema: dict, max_tokens: int) -> list[dict | None]:
        params = SamplingParams(temperature=0.0, max_tokens=max_tokens, guided_decoding=GuidedDecodingParams(json=schema))
        res = []
        for o in llm.chat([[{"role": "user", "content": p}] for p in prompts], params, use_tqdm=True):
            try:
                res.append(json.loads(o.outputs[0].text))
            except json.JSONDecodeError:
                res.append(None)
        return res

    qids = sorted(tests)
    cover = run([CTX_PROMPT.format(ctx=format_context(tests[q]["ctx"]), q=tests[q]["query"]) for q in qids], CTX_SCHEMA, 20)
    for q, c in zip(qids, cover):
        tests[q]["context_answers"] = bool(c and c["context_answers"])
    jobs = [(n, q) for n in args.names for q in qids]
    random.Random(0).shuffle(jobs)   # blind: models interleaved, no model names in the prompt
    verdicts = run([JUDGE_PROMPT.format(ctx=format_context(tests[q]["ctx"]), q=tests[q]["query"],
                                        gold=tests[q]["gold_answer"] or "(none)", answer=answers[n][q]["answer"])
                    for n, q in jobs], JUDGE_SCHEMA, 1500)
    rows = defaultdict(list)
    n_fail = 0
    for (n, q), v in zip(jobs, verdicts):
        if v is None:
            n_fail += 1
            continue
        rows[n].append({**tests[q], **answers[n][q], **v})
    report = {n: {"all": summarize([r for r in rs if r["set"] != "detail"]), **{s: summarize([r for r in rs if r["set"] == s])
                                            for s in sorted({r["set"] for r in rs})}} for n, rs in rows.items()}
    log = Path(os.environ.get("ZOLAK_STORAGE", ".")) / "logs" / f"eval_rag_{OUT.name}.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    for n, rs in rows.items():
        with (OUT / f"judged_{n}.jsonl").open("w", encoding="utf-8") as f:
            for r in sorted(rs, key=lambda r: r["qid"]):
                f.write(json.dumps({k: v for k, v in r.items() if k != "ctx"}, ensure_ascii=False) + "\n")
    metrics = list(next(iter(report.values()))["all"])
    print(f"judge failures: {n_fail}; report -> {log}")
    print("| metric | " + " | ".join(args.names) + " |\n|---|" + "---|" * len(args.names))
    for m in metrics:
        print(f"| {m} | " + " | ".join(str(report[n]["all"][m]) for n in args.names) + " |")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--index", default="data/index")
    p.add_argument("--device", default="cpu")
    p.add_argument("--rerank", default="", help="cross-encoder to re-rank the hybrid candidates (e.g. BAAI/bge-reranker-v2-m3)")
    p.add_argument("--rerank-top", type=int, default=8)
    g = sub.add_parser("gen")
    g.add_argument("--name", required=True)
    g.add_argument("--model", required=True)
    g.add_argument("--index", default="data/index", help="its kb.jsonl gives the normalizer's vocabulary")
    g.add_argument("--max-tokens", type=int, default=300, help="as the GPU worker's MAX_NEW_TOKENS")
    j = sub.add_parser("judge")
    j.add_argument("--names", nargs="+", required=True)
    j.add_argument("--tp", type=int, default=4)
    for s in (p, g, j):
        s.add_argument("--out", default=str(OUT), help="folder for tests, answers and judgements of this experiment")
    args = ap.parse_args()
    globals()["OUT"] = Path(args.out)
    {"prep": cmd_prep, "gen": cmd_gen, "judge": cmd_judge}[args.cmd](args)


if __name__ == "__main__":
    main()
