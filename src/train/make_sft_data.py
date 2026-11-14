"""Step 7a (v4) — training data written from exactly what the model sees.

For every training question the context is what the chatbot itself would retrieve (HybridRetriever.top_entries,
k=2, no threshold), and the prompt is the production prompt (space/prompts.build_messages). The teacher writes
the target from that same message plus a teacher-only rubric: answer only from the entries, cite their dates,
copy links only from «روابط:», name an office only when the entries name it, refuse with a fixed sentence when
no entry answers. A second teacher pass keeps only targets whose claims are all supported by the context.

Questions: qa records (never eval), one casual teacher paraphrase per record, and the unanswerable training
questions (near-miss / off-topic) from data/processed/paraphrases.jsonl. Eval records, and summaries that
contain them, are hidden from retrieval, so eval answers cannot leak into training.

Output (prompt-completion format for TRL): data/processed/sft_train.jsonl, sft_val.jsonl
           data/interim/sft_targets.jsonl (every teacher target with its verdicts, for review)
Usage:
    source scripts/env.sh
    CUDA_VISIBLE_DEVICES=0,1,4,5 $ZOLAK_VENVS/vllm/bin/python -m src.train.make_sft_data
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "space"))
from prompts import SYSTEM_PROMPT, URL_RE, build_messages, guard_links  # noqa: E402

REFUSAL = "ما لقيت نقاش في القروب عن السؤال دا."
MONTH_COMMA = re.compile(r"(يناير|فبراير|مارس|أبريل|مايو|يونيو|يوليو|أغسطس|سبتمبر|أكتوبر|نوفمبر|ديسمبر)، ?(\d{4})")

TARGET_SCHEMA = {  # property order matters: decide which entries answer before writing the reply
    "type": "object",
    "properties": {
        "used_entries": {"type": "array", "items": {"type": "integer"}},
        "refusal": {"type": "boolean"},
        "answer": {"type": "string"},
    },
    "required": ["used_entries", "refusal", "answer"],
}

RUBRIC = """You write the ideal reply of the chatbot «زولك» to the student's question in the last message.
The reply is training data for a small model that will see exactly the same messages, so everything in it must
come from the entries under «معلومات من القروب». The chatbot's own instructions (Arabic) follow at the end.

1. "used_entries": the numbers [i] of the entries that answer the question. An entry about a different question,
   country, office, document, permit type or situation does not count, even if it looks similar.
2. "refusal": true when no entry answers the question.
3. "answer" when refusal is false:
   - Use ONLY what the used entries say. Never add outside knowledge, advice, numbers, offices or steps, even when
     you know them. If both entries answer, combine them: what they agree on, then differences with their dates.
   - Say when the information is from: each entry shows its date in parentheses as (YYYY-MM), a summary shows a
     range (YYYY-MM → YYYY-MM). Write the date of the entry you use as an Arabic month name and the year
     (يناير، فبراير، مارس، أبريل، مايو، يونيو، يوليو، أغسطس، سبتمبر، أكتوبر، نوفمبر، ديسمبر). Use ONLY a date that is
     shown next to an entry you use; never another date, and never say that something was not discussed at a date.
   - Links: only URLs copied character by character from the «روابط:» line of a used entry, at most two.
   - Offices and authorities: name one only if the used entries name it. For visa, residence, money or legal
     topics you may end with «اتأكد من الجهة الرسمية» — never name an office the entries don't mention.
   - 2-6 sentences of natural Sudanese colloquial Arabic (no Egyptian, Gulf or Levantine forms, no formal MSA);
     German terms in Latin script. Use dialect words only where the meaning fits: «داير» means want/need and
     never "takes (time)"; for durations say «بتاخد».
   - Do not mention entry numbers like [1], and no names or personal details.
4. "answer" when refusal is true: start exactly with «ما لقيت نقاش في القروب عن السؤال دا.» and optionally add
   ONE sentence about the closest thing the entries do discuss («في نقاش عن ... لو بيفيدك.»). Nothing else:
   no answer from your own knowledge, no numbers, links or offices.

THE CHATBOT'S INSTRUCTIONS:
""" + SYSTEM_PROMPT

CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {"type": "array", "items": {"type": "object", "properties": {
            "text": {"type": "string"}, "supported": {"type": "boolean"}}, "required": ["text", "supported"]}},
    },
    "required": ["claims"],
}
CHECK_PROMPT = """Check a chatbot reply against the WhatsApp-group entries it had to answer from.
Split the reply into short atomic claims (facts, numbers, durations, dates, offices, documents, steps, links,
recommendations). For each claim, "supported" is true only if the entries state it or clearly imply it.
A date is supported when it matches the date shown next to an entry. The generic caution «اتأكد من الجهة الرسمية»
is supported; naming a specific office, website or advice that the entries do not contain is NOT supported.
For a reply that says the group has no discussion about the question, that statement itself is not a claim.

ENTRIES:
{ctx}

REPLY:
{answer}"""

SMALLTALK = [
    ("السلام عليكم", "وعليكم السلام ورحمة الله، أهلاً بيك يا زول! أنا زولك، اسألني عن أي حاجة بتخص الدراسة "
                     "والحياة في ألمانيا: الفيزا، الـ Sperrkonto، الإقامة، الجامعة، الشغل أو السكن."),
    ("شكرا", "العفو يا زول، ربنا يوفقك. لو عندك أي سؤال تاني أنا موجود."),
    ("شكراً جزيلاً على المساعدة", "تسلم، دا واجب. بالتوفيق في ألمانيا!"),
    ("انت منو؟", "أنا زولك، مساعد بيجاوب أسئلة الطلاب السودانيين في ألمانيا بالعامية، من تجارب أعضاء قروب "
                 "«طلاب السودان في ألمانيا» بعد ما اتمسحت منها أي معلومات شخصية. دي تجارب ناس ما استشارة قانونية."),
    ("بتقدر تساعدني في شنو؟", "بقدر أساعدك في أسئلة الفيزا والسفارة، الحساب المغلق، الإقامة والـ Ausländerbehörde، "
                              "القبول في الجامعات والـ Studienkolleg، اللغة، الشغل، السكن والتأمين الصحي."),
    ("صباح الخير", "صباح النور يا زول! كيفك؟ اتفضل اسأل."),
    ("مساء الخير", "مساء النور! أنا جاهز، اسأل عن أي حاجة."),
    ("انت بتتكلم سوداني؟", "أيوا طبعاً، بتكلم بالعامية السودانية عشان تكون مرتاح وانت بتسأل."),
]


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8")] if path.exists() else []


def hidden_ids(records: list[dict], eval_ids: set[str]) -> frozenset:
    """Eval records and every summary that contains one."""
    return frozenset(eval_ids | {r["id"] for r in records if r.get("source") == "summary"
                                 and eval_ids & set(r.get("members", []))})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proc", default="data/processed")
    ap.add_argument("--qa", default="qa_audited.jsonl")
    ap.add_argument("--eval", default="eval_audited.jsonl")
    ap.add_argument("--index", default="data/index")
    ap.add_argument("--val", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="pilot: N questions; writes only data/interim/sft_targets.pilot<N>.jsonl")
    args = ap.parse_args()
    proc = Path(args.proc)

    from retriever import HybridRetriever
    from src.ingest import pii
    from src.ingest.postprocess import FOREIGN_SCRIPT_RE, PLACEHOLDER_RE, build_vocab, sudanize
    from src.rag.build_summaries import tidy

    qa = load(proc / args.qa)
    eval_ids = {r["id"] for r in load(proc / args.eval)}
    assert qa and eval_ids, "missing qa or eval records"
    assert not eval_ids & {r["id"] for r in qa}, "eval records inside the qa file"
    retr = HybridRetriever(args.index, device="cpu")
    hidden = hidden_ids(retr.records, eval_ids)

    # questions: (query, record id or None, kind)
    casual = {}
    unans = []
    for p in load(proc / "paraphrases.jsonl"):
        if p["id"] is None and p.get("split") == "train":
            unans.append((p["query"], None, p["style"]))
        elif p["id"] is not None and p["style"] == "casual" and p["id"] not in eval_ids:
            casual.setdefault(p["id"], p["query"])
    questions = []
    for r in qa:
        questions.append((r["question"], r["id"], "kb_question"))
        if r["id"] in casual:
            questions.append((casual[r["id"]], r["id"], "paraphrase"))
    questions += unans
    if args.limit:
        questions = random.Random(args.seed).sample(questions, min(args.limit, len(questions)))
    print(f"{len(questions)} questions: {dict(Counter(k for _, _, k in questions))}; hidden from retrieval: "
          f"{len(hidden)} ids ({len(eval_ids)} eval records + summaries containing them)")
    ctxs = [retr.top_entries(q, k=2, exclude_ids=hidden) for q, _, _ in questions]
    assert not any(c["id"] in hidden for ctx in ctxs for c in ctx)

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
    from src.ingest.distill import TEACHER

    llm = LLM(model=TEACHER, tensor_parallel_size=args.tp, max_model_len=12288, gpu_memory_utilization=0.90,
              enable_prefix_caching=True)

    def run(convs: list[list[dict]], schema: dict, temperature: float, max_tokens: int) -> list[dict | None]:
        params = SamplingParams(temperature=temperature, top_p=0.9, max_tokens=max_tokens,
                                guided_decoding=GuidedDecodingParams(json=schema))
        res = []
        for o in llm.chat(convs, params, use_tqdm=True):
            try:
                res.append(json.loads(o.outputs[0].text))
            except json.JSONDecodeError:
                res.append(None)
        return res

    # pass 1: targets from the exact production messages
    prod = [build_messages(q, ctx) for (q, _, _), ctx in zip(questions, ctxs)]
    targets = run([[{"role": "system", "content": RUBRIC}, m[-1]] for m in prod], TARGET_SCHEMA, 0.3, 700)

    vocab = build_vocab([r["answer"] for r in qa])
    fixes, drops = Counter(), Counter()
    cands = []
    for (q, rid, kind), ctx, msgs, t in zip(questions, ctxs, prod, targets):
        if not t:
            drops["json"] += 1
            continue
        ans = tidy(sudanize(re.sub(r"\[\d\]", "", t["answer"]), vocab, fixes))
        ans = MONTH_COMMA.sub(r"\1 \2", ans)              # «يوليو، 2024» -> «يوليو 2024»
        ans = guard_links(ans, ctx, final=True)
        refusal = bool(t["refusal"]) or ans.startswith(REFUSAL)
        if refusal and not ans.startswith(REFUSAL):
            ans = REFUSAL                                   # inconsistent refusal: keep only the fixed sentence
        if refusal and (URL_RE.search(ans) or re.search(r"\d", ans[len(REFUSAL):])):
            ans = REFUSAL                                   # no numbers or links after a refusal
        if FOREIGN_SCRIPT_RE.search(ans) or PLACEHOLDER_RE.search(ans) or pii.find_pii(ans):
            drops["gate"] += 1
            continue
        if not refusal and len(ans) < 40:
            drops["short"] += 1
            continue
        cands.append({"query": q, "record": rid, "kind": kind, "ctx": [c["id"] for c in ctx], "refusal": refusal,
                      "used": t.get("used_entries", []), "answer": ans, "messages": msgs, "ctx_recs": ctx})

    # pass 2: keep only targets whose claims are all supported by the context
    from prompts import format_context
    # the fixed refusal sentence is not a claim: check only what follows it (a pointer to a related topic)
    to_check = [c["answer"][len(REFUSAL):].strip() if c["refusal"] else c["answer"] for c in cands]
    idx = [i for i, t in enumerate(to_check) if t]
    res = run([[{"role": "user", "content": CHECK_PROMPT.format(ctx=format_context(cands[i]["ctx_recs"]),
                                                                 answer=to_check[i])}] for i in idx],
              CHECK_SCHEMA, 0.0, 1200)
    checks: list[dict | None] = [{"claims": []} for _ in cands]
    for i, chk in zip(idx, res):
        checks[i] = chk
    kept = []
    for c, chk in zip(cands, checks):
        claims = [x for x in (chk or {}).get("claims", []) if REFUSAL.rstrip(".") not in x.get("text", "")]
        c["unsupported"] = [x["text"] for x in claims if not x.get("supported")]
        c["n_claims"] = len(claims)
        if c["refusal"] and (chk is None or c["unsupported"]):
            c["answer"], c["unsupported"] = REFUSAL, []     # a doubtful pointer -> keep the plain refusal
            drops["refusal_pointer_removed"] += 1
            kept.append(c)
        elif chk is None:
            drops["check_json"] += 1
        elif c["unsupported"]:
            drops["unsupported"] += 1
        elif not c["refusal"] and not claims:
            drops["no_claims"] += 1
        else:
            kept.append(c)

    review = Path("data/interim") / (f"sft_targets.pilot{args.limit}.jsonl" if args.limit else "sft_targets.jsonl")
    review.parent.mkdir(parents=True, exist_ok=True)
    with review.open("w", encoding="utf-8") as f:
        for c in cands:
            f.write(json.dumps({k: v for k, v in c.items() if k not in ("messages", "ctx_recs")}, ensure_ascii=False) + "\n")
    print(f"targets: {len(questions)} -> {len(cands)} after gates -> {len(kept)} fully supported | "
          f"dropped {dict(drops)} | dialect fixes {dict(fixes)}")
    print("kept by kind:", dict(Counter((c["kind"], "refusal" if c["refusal"] else "answer") for c in kept)))
    if args.limit:
        print(f"pilot: review {review}")
        return

    # split by record (a record's question and its paraphrase stay together); unanswerable by query
    rnd = random.Random(args.seed)
    keys = sorted({c["record"] or c["query"] for c in kept})
    rnd.shuffle(keys)
    val_keys = set(keys[: max(20, int(len(keys) * args.val))])
    rows = {"train": [], "val": []}
    for c in kept:
        kind = "refusal" if c["refusal"] else "grounded"
        rows["val" if (c["record"] or c["query"]) in val_keys else "train"].append(
            {"prompt": c["messages"], "completion": [{"role": "assistant", "content": c["answer"]}], "kind": kind})
    for _ in range(5):  # small talk: no retrieval in the app, so no context here either
        for q, a in SMALLTALK:
            rows["train"].append({"prompt": build_messages(q, []), "completion": [{"role": "assistant", "content": a}],
                                  "kind": "smalltalk"})
    for split, rs in rows.items():
        rnd.shuffle(rs)
        with (proc / f"sft_{split}.jsonl").open("w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"sft_{split}.jsonl: {len(rs)} rows {dict(Counter(r['kind'] for r in rs))}")


if __name__ == "__main__":
    main()
