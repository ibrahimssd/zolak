"""Teacher-written student queries for the embedder fine-tune, the v4 training data and the v2-vs-v4 eval.

1. Paraphrases: for every group record (not summaries) 3 rewordings of its question, same meaning:
     casual  Sudanese WhatsApp dialect (داير، هسي، شنو، ياخ ...)
     msa     Modern Standard Arabic
     mixed   Arabic mixed with German/English terms, WhatsApp style
   Eval records get paraphrases too: they are used only to evaluate (consumers exclude eval ids from training).
2. Unanswerable questions, which the bot must refuse:
     nearmiss  realistic in-domain questions (studying/living in Germany) that the group never answered
     offtopic  questions unrelated to Germany (cooking, football, phones ...)
   A candidate is kept only if the teacher confirms that neither of the 2 entries the chatbot would retrieve
   answers it. Split: --n-eval-unans (half near-miss, half off-topic) for the eval, up to --n-train-unans for training.

Output: data/processed/paraphrases.jsonl
  {"id": record id, "style": casual|msa|mixed, "query"}
  {"id": null, "style": nearmiss|offtopic, "query", "split": train|eval}
Usage (after build_summaries and a bge-m3 index of kb_group.jsonl):
    source scripts/env.sh
    CUDA_VISIBLE_DEVICES=0,1,4,5 $ZOLAK_VENVS/vllm/bin/python -m src.ingest.paraphrase_kb
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

from .distill import TEACHER

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "space"))
from prompts import format_context  # noqa: E402
from retriever import HybridRetriever  # noqa: E402

PARA_SCHEMA = {
    "type": "object",
    "properties": {"casual": {"type": "string"}, "msa": {"type": "string"}, "mixed": {"type": "string"}},
    "required": ["casual", "msa", "mixed"],
}
PARA_PROMPT = """Rewrite this question the way Sudanese students in Germany would really ask it. Keep exactly the same
meaning and the same specifics (country, city, office, document, visa or permit type); do not add new details
and do not copy the wording. Return JSON with:
- "casual": casual Sudanese colloquial Arabic as typed on WhatsApp, with words like داير، هسي، شنو، ياخ، كدا
  only where they fit the meaning («داير» means want/need, never "takes (time)"); may add a little personal context
- "msa": the same question in Modern Standard Arabic
- "mixed": WhatsApp style mixing Arabic with the German/English terms (Termin, Visum, Anmeldung, appointment ...),
  may be short and informal

Question: {q}"""

LIST_SCHEMA = {
    "type": "object",
    "properties": {"questions": {"type": "array", "items": {"type": "string"}}},
    "required": ["questions"],
}
NEARMISS_PROMPT = """Write {n} different, realistic questions that a Sudanese student or newcomer in Germany could ask
in a WhatsApp group about: {topic}.
Make them specific and varied: procedures, costs, documents, deadlines, special cases, edge cases, follow-ups.
Write them in Sudanese colloquial Arabic, German terms in Latin script. Return JSON {{"questions": [...]}}."""
OFFTOPIC_PROMPT = """Write {n} different questions that a Sudanese person might ask a chatbot about: {topic}.
They must have NOTHING to do with studying, living, working, travelling to or bureaucracy in Germany or Europe.
Write them in Sudanese colloquial Arabic. Return JSON {{"questions": [...]}}."""

NEARMISS_TOPICS = [
    "German visas and embassy appointments (Cairo, Khartoum, Addis Ababa, Nairobi, VFS), visa documents and waiting",
    "Sudanese embassy services in Germany or Europe: passports, power of attorney, document attestation",
    "the blocked account (Sperrkonto), bank accounts, money transfers to and from Sudan, cost of living",
    "residence permits, the Ausländerbehörde, Fiktionsbescheinigung, extensions, Blue Card, permanent residence",
    "university admission: uni-assist, APS, applications for Bachelor/Master/PhD, scholarships like DAAD",
    "Studienkolleg and the Feststellungsprüfung",
    "German and English language courses and exams (Goethe, TestDaF, telc, IELTS)",
    "jobs, Minijob, Werkstudent, Ausbildung, job search, taxes and the tax return",
    "finding a flat, WG, student dorms, rental contracts, Anmeldung",
    "health insurance (public, private, travel insurance), doctors and the health system",
    "recognition of Sudanese certificates, anabin, ZAB, translations and attestation",
    "family reunion, asylum procedures, refugees",
    "travel between Sudan, Egypt, Saudi Arabia and Germany, flights, transit visas",
    "daily life in Germany: SIM cards, public transport tickets, driving licence, shopping, Sudanese community",
]
OFFTOPIC_TOPICS = [
    "cooking Sudanese food (kisra, asida, ful, mullah) and sweets",
    "football (Hilal, Merreikh, the national team, European leagues) and other sports",
    "smartphones, laptops, apps and social media",
    "health, fitness, diet and home remedies (not insurance or doctors in Germany)",
    "Sudanese music, poetry, films and culture",
    "school homework: maths, physics, chemistry, geography and history questions",
    "home repairs, gardening, pets and cars in Sudan",
    "general knowledge, jokes, riddles and writing a poem or a message",
]

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {"answered_by": {"type": "array", "items": {"type": "integer"}}},
    "required": ["answered_by"],
}
VERIFY_PROMPT = """Below are entries from a Sudanese students' WhatsApp group in Germany and a student's question.
List in "answered_by" the numbers [i] of the entries that answer the question (fully, or its main point).
An entry about a different question, country, office, document, permit type or situation does NOT count, even if
it looks similar. Return [] if none answers it.

ENTRIES:
{ctx}

QUESTION: {q}"""


def clean_q(q: str) -> str:
    return re.sub(r"\s+", " ", q).strip().strip('"«»').strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kb", default="data/processed/kb_group.jsonl")
    ap.add_argument("--index", default="data/index", help="index the chatbot retrieves from (for the refusal check)")
    ap.add_argument("--out", default="data/processed/paraphrases.jsonl")
    ap.add_argument("--per-topic", type=int, default=30)
    ap.add_argument("--n-train-unans", type=int, default=300)
    ap.add_argument("--n-eval-unans", type=int, default=50)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="pilot: paraphrase only N records")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

    records = [json.loads(l) for l in open(args.kb, encoding="utf-8")]
    records = [r for r in records if r.get("source") != "summary"]
    if args.limit:
        records = random.Random(0).sample(records, min(args.limit, len(records)))
    llm = LLM(model=TEACHER, tensor_parallel_size=args.tp, max_model_len=8192, gpu_memory_utilization=0.90,
              enable_prefix_caching=True)

    def chat(prompts: list[str], schema: dict, temperature: float, max_tokens: int) -> list[dict | None]:
        params = SamplingParams(temperature=temperature, top_p=0.95, max_tokens=max_tokens,
                                guided_decoding=GuidedDecodingParams(json=schema))
        outs = llm.chat([[{"role": "user", "content": p}] for p in prompts], params, use_tqdm=True)
        res = []
        for o in outs:
            try:
                res.append(json.loads(o.outputs[0].text))
            except json.JSONDecodeError:
                res.append(None)
        return res

    rows: list[dict] = []
    # 1. paraphrases
    for r, o in zip(records, chat([PARA_PROMPT.format(q=r["question"]) for r in records], PARA_SCHEMA, 0.8, 400)):
        if not o:
            continue
        for style in ("casual", "msa", "mixed"):
            q = clean_q(o.get(style, ""))
            if len(q) >= 8 and q != r["question"]:
                rows.append({"id": r["id"], "style": style, "query": q})
    print(f"paraphrases: {len(rows)} for {len(records)} records")

    # 2. unanswerable candidates, checked against what the chatbot would actually retrieve
    jobs = [("nearmiss", t) for t in NEARMISS_TOPICS] + [("offtopic", t) for t in OFFTOPIC_TOPICS]
    prompts = [(NEARMISS_PROMPT if kind == "nearmiss" else OFFTOPIC_PROMPT).format(n=args.per_topic, topic=t)
               for kind, t in jobs]
    cands, seen = [], set()
    for (kind, _), o in zip(jobs, chat(prompts, LIST_SCHEMA, 0.9, 3000)):
        for q in (o or {}).get("questions", []):
            q = clean_q(q)
            if len(q) >= 8 and q not in seen:
                seen.add(q)
                cands.append((kind, q))
    retr = HybridRetriever(args.index, device="cpu")
    ctxs = [retr.top_entries(q, k=2) for _, q in cands]
    verdicts = chat([VERIFY_PROMPT.format(ctx=format_context(c), q=q) for (_, q), c in zip(cands, ctxs)],
                    VERIFY_SCHEMA, 0.0, 60)
    unans = {"nearmiss": [], "offtopic": []}
    for (kind, q), v in zip(cands, verdicts):
        if v is not None and not v.get("answered_by"):
            unans[kind].append(q)
    print(f"unanswerable: {len(cands)} candidates -> kept " + ", ".join(f"{k} {len(v)}" for k, v in unans.items()))
    rnd = random.Random(0)
    half = args.n_eval_unans // 2
    for kind in ("nearmiss", "offtopic"):
        rnd.shuffle(unans[kind])
    eval_q = unans["nearmiss"][:half] + unans["offtopic"][: args.n_eval_unans - half]
    rest = [("nearmiss", q) for q in unans["nearmiss"][half:]] + \
           [("offtopic", q) for q in unans["offtopic"][args.n_eval_unans - half:]]
    rnd.shuffle(rest)
    eval_set = set(eval_q)
    for kind in ("nearmiss", "offtopic"):
        rows += [{"id": None, "style": kind, "query": q, "split": "eval"} for q in unans[kind] if q in eval_set]
    rows += [{"id": None, "style": kind, "query": q, "split": "train"} for kind, q in rest[: args.n_train_unans]]

    with open(args.out, "w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> {args.out}: "
          f"{dict(Counter((x['style'], x.get('split', '-')) for x in rows))}")


if __name__ == "__main__":
    main()
