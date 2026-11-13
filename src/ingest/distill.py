"""Step 4 — distill threads + info posts into clean Sudanese-dialect Q&A with a local teacher LLM (vLLM).

The data never leaves this machine. Teacher output is JSON (schema-constrained decoding).

Input : data/processed/threads.jsonl, data/processed/info_posts.jsonl
Output: data/interim/distill_raw.jsonl   raw teacher outputs (for review / re-runs)
        then `postprocess.py` builds qa.jsonl / kb.jsonl / eval.jsonl

Run (with the vLLM venv):
    source scripts/env.sh
    $ZOLNA_VENVS/vllm/bin/python -m src.ingest.distill --limit 30          # pilot
    $ZOLNA_VENVS/vllm/bin/python -m src.ingest.distill                     # all
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

TEACHER = "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"

TOPICS = [
    "visa_embassy",            # German visa, embassy appointments (Cairo/Khartoum/...), VFS
    "sudanese_embassy",        # Sudanese embassy services: passports, documents (e.g. Brussels/Berlin)
    "blocked_account_money",   # Sperrkonto, transfers, banks, cost of living
    "residence_permit",        # Aufenthaltstitel, Ausländerbehörde, Fiktionsbescheinigung, Blue Card
    "university_admission",    # applications, uni-assist, APS, Master/Bachelor/PhD, scholarships
    "studienkolleg",           # foundation year, Feststellungsprüfung
    "language",                # German/English courses and exams
    "jobs_ausbildung",         # Minijob, Werkstudent, Ausbildung, job search, taxes
    "housing_anmeldung",       # flats, WG, dorms, Anmeldung
    "health_insurance",        # public/private/travel insurance
    "recognition_documents",   # anabin, ZAB, certificate attestation/translation
    "family_asylum",           # family reunion, asylum, refugees
    "travel",                  # travel to/through Egypt, KSA, other countries; flights
    "daily_life",              # SIM cards, transport, community, shopping, etc.
    "other",
]

SCHEMA = {  # property order matters: constrained decoding writes content first, then labels it
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "useful": {"type": "boolean"},
        "question": {"type": "string"},
        "answer": {"type": "string"},
        "facts": {"type": "array", "items": {"type": "string"}},
        "german_terms": {"type": "array", "items": {"type": "string"}},
        "official_links": {"type": "array", "items": {"type": "string"}},
        "topic": {"type": "string", "enum": TOPICS},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "time_sensitive": {"type": "boolean"},
    },
    "required": ["reason", "useful", "question", "answer", "facts", "german_terms", "official_links",
                 "topic", "confidence", "time_sensitive"],
}

SYSTEM = """You are a careful knowledge curator for a chatbot that helps Sudanese students and newcomers in Germany.
You receive an anonymized excerpt from a Sudanese WhatsApp community group. Extract reusable knowledge from it.

Rules:
1. Use ONLY information stated in the excerpt. Do not add outside knowledge, do not correct or "improve" facts.
   If members disagree, say so briefly ("في ناس قالت ... وفي ناس قالت ...").
2. Privacy & generality: never include names, phone numbers, emails, addresses of private people, or personal
   logistics (who is travelling when, who has a car, someone's private plans, money they need). Placeholders like
   [اسم] [رقم] [رابط] @مستخدم must NOT appear in your output. Turn personal stories into general experience
   ("في ناس جربوا ..."). Do not narrate the chat ("واحد من الأعضاء قال ...") more than once.
3. Language of `question` and `answer`: natural SUDANESE Arabic dialect only (العامية السودانية), like a helpful
   Sudanese student talking to a friend. Typical Sudanese forms: عشان، داير/دايرة، عايز، زول، ياتو، شنو، كدا،
   برضو، هسي/هسع، لسه، ساي، شديد، بتاع/حق، قروش، تمشي، بتقدر، ما بتحتاج، حتلقى، قبيل، ياخ.
   Do NOT use Gulf (وش، هذي، مو، ابغى، يعطيك إياه), Egyptian (مش، -ش negation like ما تبقاش، هـ-future like هيوريك،
   دلوقتي، إزاي) or Levantine (هيك، بدك، شو) forms, and do not write formal MSA.
   Use only Arabic script plus German/English terms in Latin script (Sperrkonto, Anmeldung, Aufenthaltstitel,
   uni-assist ...). Never output Chinese or any other script.
4. `question`: one standalone, general question (no "this group", no names) that the excerpt actually answers.
   `answer`: 2-6 sentences, practical, only what members said: amounts, durations, documents, offices, steps.
5. `facts`: 1-6 short atomic facts in Arabic (MSA is fine here), each standing alone, with the key German term if any.
6. `german_terms`: only German/English keywords that are relevant to THIS answer (useful for search).
   `official_links`: only full official URLs (government, embassy, university, DAAD, uni-assist ...) that appear in
   the excerpt, else [].
7. `topic`: the topic of the question/answer you wrote (e.g. money apps/transfers -> blocked_account_money;
   any visa or embassy appointment, including DAAD scholarship visas -> visa_embassy).
8. `confidence`: high = several members agree or first-hand experience; medium = one plausible answer;
   low = vague/contradictory.
9. `time_sensitive`: true if the answer contains fees, amounts, processing/waiting times, appointment availability,
   current embassy situations, or rules that may change. Otherwise false.
10. `useful` = false when the question is not answered in the excerpt, or it is chit-chat/jokes/politics/news/
   condolences/ads, a personal transaction or meetup, or too specific to one person to help others.
   Always write a short English `reason` first.

STYLE EXAMPLE (style only, do not reuse its content):
question: "كيف ألقى سكن طلاب أو WG في مدينة جديدة؟"
answer: "أسهل حاجة تبدا بموقع WG-Gesucht وتكتب رسالة قصيرة عن نفسك، ولو بتقدر أكتبها بالألماني. الناس قالو لازم
ترسل لإعلانات كتيرة عشان الردود بتكون قليلة، وما تدفع أي قروش قبل ما تشوف الأوضة وتمضي العقد. وبرضو قدّم بدري
في الـ Studierendenwerk بتاع جامعتك، بس قايمة الانتظار بتكون طويلة شوية.\""""

THREAD_TMPL = """Date of the conversation: {date}
The QUESTION is marked (سؤال). Later messages may be replies to it or unrelated chatter — use only relevant replies.

{body}"""

POST_TMPL = """Date of the post: {date}
This is a single informative post (no question). Write the `question` as the main question this post answers.

{body}"""

PART_LABEL = {"prelude": "سؤال", "question": "سؤال", "reply": "رد"}


def format_thread(t: dict, max_chars: int = 9000) -> str:
    lines, total = [], 0
    for m in t["messages"]:
        line = f"[{m['role']}] ({PART_LABEL[m['part']]}) {m['ts'][11:16]}: {m['text']}"
        if total + len(line) > max_chars:
            lines.append("... (truncated)")
            break
        lines.append(line)
        total += len(line)
    return THREAD_TMPL.format(date=t["date"], body="\n".join(lines))


def load_jobs(proc: Path) -> list[dict]:
    jobs = []
    for line in (proc / "threads.jsonl").open(encoding="utf-8"):
        t = json.loads(line)
        jobs.append({"src_id": t["thread_id"], "kind": "thread", "date": t["date"], "prompt": format_thread(t)})
    for line in (proc / "info_posts.jsonl").open(encoding="utf-8"):
        p = json.loads(line)
        jobs.append({"src_id": p["post_id"], "kind": "post", "date": p["date"],
                     "prompt": POST_TMPL.format(date=p["date"], body=p["text"][:9000])})
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proc", default="data/processed")
    ap.add_argument("--out", default="data/interim/distill_raw.jsonl")
    ap.add_argument("--model", default=TEACHER)
    ap.add_argument("--tp", type=int, default=4, help="tensor parallel size")
    ap.add_argument("--limit", type=int, default=0, help="pilot: random sample of N jobs")
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--resume", action="store_true", help="skip src_ids already in --out")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

    jobs = load_jobs(Path(args.proc))
    if args.limit:
        jobs = random.Random(0).sample(jobs, args.limit)
    out = Path(args.out if not args.limit else args.out.replace(".jsonl", f".pilot{args.limit}.jsonl"))
    done = set()
    if args.resume and out.exists():
        done = {json.loads(l)["src_id"] for l in out.open(encoding="utf-8")}
        jobs = [j for j in jobs if j["src_id"] not in done]
    print(f"{len(jobs)} jobs ({sum(j['kind'] == 'thread' for j in jobs)} threads, "
          f"{sum(j['kind'] == 'post' for j in jobs)} posts) -> {out}")

    llm = LLM(model=args.model, tensor_parallel_size=args.tp, max_model_len=args.max_model_len,
              gpu_memory_utilization=0.90, enable_prefix_caching=True, trust_remote_code=False)
    params = SamplingParams(temperature=0.2, top_p=0.9, max_tokens=1500,
                            guided_decoding=GuidedDecodingParams(json=SCHEMA))
    convs = [[{"role": "system", "content": SYSTEM}, {"role": "user", "content": j["prompt"]}] for j in jobs]

    t0 = time.time()
    results = llm.chat(convs, params, use_tqdm=True)
    n_ok = 0
    with out.open("a" if args.resume else "w", encoding="utf-8") as f:
        for j, r in zip(jobs, results):
            text = r.outputs[0].text
            try:
                data, err = json.loads(text), None
                n_ok += 1
            except json.JSONDecodeError as e:
                data, err = None, str(e)
            f.write(json.dumps({"src_id": j["src_id"], "kind": j["kind"], "date": j["date"], "teacher": args.model,
                                "output": data, "error": err, "raw": None if data else text},
                               ensure_ascii=False) + "\n")
    print(f"done in {time.time() - t0:.0f}s; parsed {n_ok}/{len(jobs)}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
