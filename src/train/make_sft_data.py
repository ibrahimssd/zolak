"""Step 7a — build the LoRA training set for the student from the distilled Q&A.

Every example uses exactly the prompt the live bot uses (space/prompts.build_messages), so training == inference.
Example kinds:
  grounded    gold KB record (+ a distractor half the time) -> Sudanese answer that cites month/year
  irrelevant  only unrelated records in context -> honest "the group info does not cover this" + right authority
  nocontext   no records -> same honest answer
  smalltalk   greetings / thanks / who are you
Eval items (data/processed/eval.jsonl) are never used here.

Output (prompt-completion format for TRL): data/processed/sft_train.jsonl, data/processed/sft_val.jsonl
Usage: python3 -m src.train.make_sft_data
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "space"))
from prompts import NOT_IN_GROUP, build_messages  # noqa: E402

MONTHS = ["يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو", "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر"]

AUTHORITY = {
    "visa_embassy": "السفارة الألمانية أو موقع Auswärtiges Amt",
    "sudanese_embassy": "السفارة السودانية",
    "blocked_account_money": "البنك أو مزود الـ Sperrkonto",
    "residence_permit": "الـ Ausländerbehörde بتاعتك",
    "university_admission": "الجامعة أو uni-assist",
    "studienkolleg": "الـ Studienkolleg أو الجامعة",
    "language": "معهد اللغة أو جهة الامتحان",
    "jobs_ausbildung": "الـ Agentur für Arbeit أو صاحب الشغل",
    "housing_anmeldung": "الـ Bürgeramt في مدينتك",
    "health_insurance": "شركة التأمين الصحي",
    "recognition_documents": "anabin أو الـ ZAB",
    "family_asylum": "الـ BAMF أو الـ Ausländerbehörde",
    "travel": "شركة الطيران أو سفارة البلد الماشي ليه",
}
DEFAULT_AUTHORITY = "الجهة الرسمية المختصة"

LEAD_INS = [
    "حسب تجارب الناس في القروب ({when})، {answer}",
    "الناس في القروب قالو في {when}: {answer}",
    "من تجارب الأعضاء ({when}): {answer}",
]
CHECK_SUFFIX = [
    " بس المعلومات دي بتتغير، فاتأكد من {auth} قبل ما تعتمد عليها.",
    " وأحسن تتأكد من {auth} لأنو القوانين والرسوم بتتغير.",
]
IRRELEVANT = [
    "المعلومات اللي عندي ما بتجاوب سؤالك دا بالضبط، لا من القروب ولا من المواقع الرسمية، وما داير أألف ليك حاجة. "
    "أحسن تسأل في القروب أو تتأكد من {auth}.",
    "للأسف ما لقيت معلومة مؤكدة عن النقطة دي بالذات، لا في القروب ولا في المواقع الرسمية. عشان ما أديك معلومة "
    "غلط، اتواصل مع {auth}، وممكن تسأل الناس في القروب برضو.",
]
NOCONTEXT = [
    "ما لقيت معلومة عن السؤال دا لا في القروب ولا في المواقع الرسمية. أحسن حاجة تسأل {auth} مباشرة، أو تطرح "
    "السؤال في القروب.",
    "السؤال دا ما لقيت ليهو معلومة مؤكدة، لا من تجارب القروب ولا من المواقع الرسمية. اتأكد من {auth} عشان تاخد "
    "الإجابة الصاح.",
]
# Answers that come from an official site or a web search start by saying clearly that the group had no answer.
OFFICIAL_LEAD = [
    NOT_IN_GROUP + " 🙂 حسب الموقع الرسمي {site}: {answer}",
    NOT_IN_GROUP + ". لقيت في الموقع الرسمي {site}: {answer}",
]
WEB_LEAD = [
    NOT_IN_GROUP + " 🙂 لقيت في موقع {site}: {answer}",
    NOT_IN_GROUP + ". حسب موقع {site} على الإنترنت: {answer}",
]
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


def when(date: str) -> str:
    y, m = date[:4], int(date[5:7])
    return f"{MONTHS[m - 1]} {y}"


def grounded_target(rec: dict, rnd: random.Random) -> str:
    ans = rec["answer"].rstrip()
    text = rnd.choice(LEAD_INS).format(when=when(rec["date"]), answer=ans)
    if rec.get("time_sensitive") or rec["topic"] in AUTHORITY:
        if rnd.random() < 0.6:
            text += rnd.choice(CHECK_SUFFIX).format(auth=AUTHORITY.get(rec["topic"], DEFAULT_AUTHORITY))
    if rec.get("links"):
        text += f" الرابط الرسمي: {rec['links'][0]}"
    return text


def sourced_target(rec: dict, leads: list[str], rnd: random.Random) -> str:
    text = rnd.choice(leads).format(site=rec["site"], answer=rec["answer"].rstrip())
    if rec.get("time_sensitive") and rnd.random() < 0.5:
        text += rnd.choice(CHECK_SUFFIX).format(auth=AUTHORITY.get(rec["topic"], DEFAULT_AUTHORITY))
    return text + f" الرابط: {rec['links'][0]}"


def web_record(rec: dict, chunk: dict) -> dict:
    """The raw page section (English/German) as a live web search would return it."""
    return {"source": "web", "site": rec["site"], "date": rec["date"], "question": chunk["title"],
            "answer": chunk["text"], "facts": [], "links": rec["links"]}


def example(question: str, context: list[dict], target: str, kind: str) -> dict:
    msgs = build_messages(question, context)
    return {"prompt": msgs, "completion": [{"role": "assistant", "content": target}], "kind": kind}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proc", default="data/processed")
    ap.add_argument("--irrelevant", type=float, default=0.20)
    ap.add_argument("--nocontext", type=float, default=0.10)
    ap.add_argument("--val", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--official", default="data/official/kb_official.jsonl")
    ap.add_argument("--chunks", default="data/official/chunks.jsonl")
    args = ap.parse_args()

    rnd = random.Random(args.seed)
    proc = Path(args.proc)
    qa = [json.loads(l) for l in (proc / "qa.jsonl").open(encoding="utf-8")]
    by_topic: dict[str, list[dict]] = {}
    for r in qa:
        by_topic.setdefault(r["topic"], []).append(r)

    official, chunks = [], {}
    off_path = Path(args.official)
    if off_path.exists():
        official = [json.loads(l) for l in off_path.open(encoding="utf-8")]
        chunks = {json.loads(l)["chunk_id"]: json.loads(l) for l in Path(args.chunks).open(encoding="utf-8")}

    rnd.shuffle(qa)
    rnd.shuffle(official)
    n_val = max(20, int(len(qa) * args.val))
    n_val_off = int(len(official) * args.val)
    splits = {"val": qa[:n_val], "train": qa[n_val:]}
    off_splits = {"val": official[:n_val_off], "train": official[n_val_off:]}
    counts: Counter = Counter()
    for split, items in splits.items():
        rows = []
        for rec in items:
            ctx = [rec]
            if rnd.random() < 0.5:
                ctx.append(rnd.choice(qa))
                rnd.shuffle(ctx)
            rows.append(example(rec["question"], ctx, grounded_target(rec, rnd), "grounded"))
            auth = AUTHORITY.get(rec["topic"], DEFAULT_AUTHORITY)
            if rnd.random() < args.irrelevant:
                others = [r for t, rs in by_topic.items() if t != rec["topic"] for r in rs]
                rows.append(example(rec["question"], rnd.sample(others, 2),
                                    rnd.choice(IRRELEVANT).format(auth=auth), "irrelevant"))
            elif rnd.random() < args.nocontext / (1 - args.irrelevant):
                rows.append(example(rec["question"], [], rnd.choice(NOCONTEXT).format(auth=auth), "nocontext"))
        for rec in off_splits[split]:
            ctx = [rec]
            if rnd.random() < 0.4:   # sometimes next to an unrelated group record
                ctx.append(rnd.choice(qa))
                rnd.shuffle(ctx)
            rows.append(example(rec["question"], ctx, sourced_target(rec, OFFICIAL_LEAD, rnd), "official"))
            if rec.get("chunk_id") in chunks and rnd.random() < 0.5:
                rows.append(example(rec["question"], [web_record(rec, chunks[rec["chunk_id"]])],
                                    sourced_target(rec, WEB_LEAD, rnd), "web"))
        if split == "train":
            for _ in range(5):  # small talk, a few repetitions
                for q, a in SMALLTALK:
                    rows.append(example(q, [], a, "smalltalk"))
        rnd.shuffle(rows)
        with (proc / f"sft_{split}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts.update(f"{split}:{r['kind']}" for r in rows)
    print(dict(sorted(counts.items())))


if __name__ == "__main__":
    main()
