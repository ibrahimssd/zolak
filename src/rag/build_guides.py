"""Topic guides: one detailed, dated entry per broad how-to question, written from many group discussions.

Broad questions («كيف أجدد الإقامة وشنو الأوراق المطلوبة؟») need many discussions, but the chatbot sees only two
short entries, so its answers miss steps and documents. A guide combines the related entries into one step-by-step
answer that the retriever can return as a single entry.

1. Guide questions: the topic-card questions of the web page, plus broad questions the teacher proposes per topic
   from that topic's entries (the students' 👎 questions are NOT used: they stay an independent test).
2. Per question, the 15 most related entries (held-out eval records and summaries containing them are hidden).
3. The teacher writes the guide from those entries only: opening sentence, steps/requirements as a list, costs and
   durations with the month/year reported, disagreements and changes over time; Sudanese dialect.
4. Claim check (same check as make_sft_data): unsupported claims -> one revision -> re-check; still unsupported or
   too few entries -> dropped. Then the dialect normalizer, tidy, and the usual gates.

Output: data/processed/kb_guides.jsonl and data/processed/kb_group_guides.jsonl (group KB + guides, for the index)
Usage:
    source scripts/env.sh
    CUDA_VISIBLE_DEVICES=0,1,4,5 LOCAL_EMBED=$ZOLAK_MODELS/zolak-embed-v1 $ZOLAK_VENVS/vllm/bin/python -m src.rag.build_guides
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "space"))
from dialect_fix import build_vocab, sudanize_text  # noqa: E402
from prompts import format_context  # noqa: E402
from retriever import HybridRetriever  # noqa: E402

from src.ingest import pii  # noqa: E402
from src.ingest.postprocess import FOREIGN_SCRIPT_RE, PLACEHOLDER_RE  # noqa: E402
from src.rag.build_summaries import tidy  # noqa: E402
from src.train.detail_tests import topic_card_questions  # noqa: E402
from src.train.make_sft_data import CHECK_PROMPT, CHECK_SCHEMA, hidden_ids  # noqa: E402

PROC = Path("data/processed")
TOPIC_QUOTA = {"visa_embassy": 8, "travel": 4, "residence_permit": 4, "university_admission": 4,
               "blocked_account_money": 4, "recognition_documents": 3, "family_asylum": 3, "studienkolleg": 3,
               "language": 3, "health_insurance": 3, "jobs_ausbildung": 3, "daily_life": 2, "sudanese_embassy": 2,
               "housing_anmeldung": 2}

PROPOSE_SCHEMA = {"type": "object", "properties": {"questions": {"type": "array", "items": {"type": "string"}}},
                  "required": ["questions"]}
PROPOSE_PROMPT = """Below are questions that Sudanese students in Germany asked in their WhatsApp group about the topic
"{topic}". Propose {n} BROAD practical questions a new student would ask about this topic that need several of these
discussions together to answer well (procedures, required documents, costs and durations, options), for example
«كيف أجدد الإقامة وشنو الأوراق المطلوبة؟». Write them in natural Sudanese colloquial Arabic, German terms in Latin
script, each different from the others.

QUESTIONS FROM THE GROUP:
{qs}"""

GUIDE_SCHEMA = {  # property order matters: decide which entries are relevant before writing
    "type": "object",
    "properties": {
        "used_entries": {"type": "array", "items": {"type": "integer"}},
        "useful": {"type": "boolean"},
        "question": {"type": "string"},
        "answer": {"type": "string"},
        "facts": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
    },
    "required": ["used_entries", "useful", "question", "answer", "facts"],
}
GUIDE_PROMPT = """You write a practical guide for Sudanese students in Germany that answers a broad question, using ONLY
the numbered WhatsApp-group entries below (each shows the month/year it was discussed).

Rules:
1. "used_entries": the numbers [i] of the entries relevant to the question. If fewer than 2 are relevant, "useful"
   is false and the other fields may be empty.
2. Use ONLY what the used entries say. Never add outside knowledge, offices, websites, numbers, steps or documents
   that the entries do not contain, even when you know them.
3. "answer" in natural Sudanese colloquial Arabic (no Egyptian, Gulf or Levantine forms, no formal MSA), German
   terms in Latin script, 6-15 lines:
   - one opening sentence saying this is from the group's experience and between which months/years;
   - the steps or requirements as a short list, each line starting with a dash and a space (documents, where to go, what to bring);
   - costs and durations with the month/year they were reported;
   - where people disagreed or what changed over time;
   - end with «اتأكد من الجهة الرسمية».
   Skip any part the entries do not cover. Use «داير» only to mean want/need, never "takes (time)".
   Sudanese negation has no «ش»: «ما بيحتاجو», not «ما يحتاجوش». No names or personal details. No entry numbers.
4. "question": the broad question, rewritten naturally in Sudanese dialect if needed.
5. "facts": 3-8 short facts from the used entries, each with its month/year.

BROAD QUESTION: {q}

ENTRIES:
{ctx}"""

REVISE_PROMPT = """The guide below contains claims that the entries do not support. Rewrite the guide without these
claims and change nothing else (same style, same structure, Sudanese dialect). Return only the guide text in "answer".

UNSUPPORTED CLAIMS:
{bad}

ENTRIES:
{ctx}

GUIDE:
{answer}"""
REVISE_SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}


BULLET_ECHO = re.compile(r"^[ \t]*(?:-[ \t]*)?«[ \t]*-[ \t]*»[ \t]*", re.M)


def clean_bullets(text: str) -> str:
    """The teacher sometimes copies the quoted marker «- » from the instructions: «- «- » X» -> «- X»."""
    return BULLET_ECHO.sub("- ", text).replace("«- »", "").strip()


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8")] if path.exists() else []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default="data/index")
    ap.add_argument("--n-entries", type=int, default=15)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="pilot: only N guide questions")
    args = ap.parse_args()

    retr = HybridRetriever(args.index, device="cpu")
    kb = retr.records
    by_id = {r["id"]: r for r in kb}
    eval_ids = {r["id"] for r in load(PROC / "eval_audited.jsonl")} or {r["id"] for r in load(PROC / "eval.jsonl")}
    hidden = hidden_ids(kb, eval_ids)

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
    from src.ingest.distill import TEACHER

    llm = LLM(model=TEACHER, tensor_parallel_size=args.tp, max_model_len=16384, gpu_memory_utilization=0.90,
              enable_prefix_caching=True)

    def run(prompts: list[str], schema: dict, temperature: float, max_tokens: int) -> list[dict | None]:
        params = SamplingParams(temperature=temperature, top_p=0.9, max_tokens=max_tokens,
                                guided_decoding=GuidedDecodingParams(json=schema))
        res = []
        for o in llm.chat([[{"role": "user", "content": p}] for p in prompts], params, use_tqdm=True):
            try:
                res.append(json.loads(o.outputs[0].text))
            except json.JSONDecodeError:
                res.append(None)
        return res

    # 1. guide questions
    by_topic = defaultdict(list)
    for r in kb:
        if r.get("source") != "summary" and r["id"] not in hidden:
            by_topic[r.get("topic", "other")].append(r["question"])
    rnd = random.Random(0)
    topics = [t for t in TOPIC_QUOTA if by_topic.get(t)]
    proposals = run([PROPOSE_PROMPT.format(topic=t, n=TOPIC_QUOTA[t],
                                           qs="\n".join("- " + q for q in rnd.sample(by_topic[t], min(40, len(by_topic[t])))))
                     for t in topics], PROPOSE_SCHEMA, 0.3, 1200)
    questions = [(q, "topic_card") for q in topic_card_questions()]
    for t, p in zip(topics, proposals):
        questions += [(q.strip(), t) for q in (p or {}).get("questions", [])[:TOPIC_QUOTA[t]] if q.strip()]
    if args.limit:
        questions = questions[: args.limit]
    print(f"{len(questions)} guide questions: {dict(Counter(s for _, s in questions))}")

    # 2-3. entries and guides
    ctxs = [retr.candidates(q, args.n_entries, exclude_ids=hidden) for q, _ in questions]
    guides = run([GUIDE_PROMPT.format(q=q, ctx=format_context(c)) for (q, _), c in zip(questions, ctxs)],
                 GUIDE_SCHEMA, 0.2, 2500)

    # 4. claim check, one revision, re-check
    drops = Counter()
    cands = []
    for (q, src), ctx, g in zip(questions, ctxs, guides):
        if not g or not g.get("useful") or len([i for i in g.get("used_entries", []) if 1 <= i <= len(ctx)]) < 2:
            drops["not_useful"] += 1
            continue
        used = [ctx[i - 1] for i in sorted(set(g["used_entries"])) if 1 <= i <= len(ctx)]
        cands.append({"q": q, "src": src, "used": used, "g": g})

    def check(items):
        res = run([CHECK_PROMPT.format(ctx=format_context(c["used"]), answer=c["g"]["answer"]) for c in items],
                  CHECK_SCHEMA, 0.0, 2500)
        for c, chk in zip(items, res):
            c["bad"] = None if chk is None else [x["text"] for x in chk.get("claims", []) if not x.get("supported")]

    check(cands)
    to_fix = [c for c in cands if c["bad"]]
    fixed = run([REVISE_PROMPT.format(bad="\n".join("- " + b for b in c["bad"]), ctx=format_context(c["used"]),
                                      answer=c["g"]["answer"]) for c in to_fix], REVISE_SCHEMA, 0.0, 2500)
    for c, f in zip(to_fix, fixed):
        if f and f.get("answer"):
            c["g"]["answer"] = f["answer"]
    check(to_fix)

    vocab = build_vocab([" ".join([r["question"], r["answer"], *r.get("facts", [])]) for r in kb])
    out = []
    for c in cands:
        if c["bad"] is None:
            drops["check_failed"] += 1
            continue
        if c["bad"]:
            drops["unsupported_after_revision"] += 1
            continue
        g = c["g"]
        ans = clean_bullets(tidy(sudanize_text(g["answer"], vocab)))
        quest = tidy(sudanize_text(g["question"] or c["q"], vocab))
        facts = [tidy(sudanize_text(f, vocab)) for f in g.get("facts", []) if f.strip()]
        blob = " ".join([quest, ans, *facts])
        if FOREIGN_SCRIPT_RE.search(blob) or PLACEHOLDER_RE.search(blob) or pii.find_pii(blob) or len(ans) < 200:
            drops["gate"] += 1
            continue
        members = sorted({m for r in c["used"] for m in (r.get("members") or [r["id"]])})
        dates = sorted(by_id[m]["date"] for m in members if m in by_id and by_id[m].get("date")) or \
            sorted(r["date"] for r in c["used"])
        links = list(dict.fromkeys(u for r in c["used"] for u in r.get("links", [])))[:3]
        out.append({"id": f"guide-{len(out) + 1}", "source": "guide", "question": quest, "answer": ans,
                    "facts": facts, "members": members, "n_discussions": len(members), "date_from": dates[0],
                    "date": dates[-1], "topic": Counter(r.get("topic", "other") for r in c["used"]).most_common(1)[0][0],
                    "links": links, "german_terms": [], "guide_for": c["q"], "guide_source": c["src"]})

    # near-duplicate guide questions: keep the guide built from more discussions
    if out:
        vecs = retr.embed([g["question"] for g in out])
        keep = []
        for i in sorted(range(len(out)), key=lambda i: -out[i]["n_discussions"]):
            if all(float(vecs[i] @ vecs[j]) < 0.92 for j in keep):
                keep.append(i)
            else:
                drops["duplicate"] += 1
        out = [out[i] for i in sorted(keep)]
        for n, g in enumerate(out, 1):
            g["id"] = f"guide-{n}"

    with (PROC / "kb_guides.jsonl").open("w", encoding="utf-8") as f:
        for g in out:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    group = load(PROC / "kb_group.jsonl")
    with (PROC / "kb_group_guides.jsonl").open("w", encoding="utf-8") as f:
        for r in group + out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(out)} guides from {len(questions)} questions; dropped {dict(drops)}; median length "
          f"{sorted(len(g['answer']) for g in out)[len(out) // 2] if out else 0} chars; "
          f"-> {PROC / 'kb_guides.jsonl'}, {PROC / 'kb_group_guides.jsonl'} ({len(group) + len(out)} entries)")


if __name__ == "__main__":
    main()
