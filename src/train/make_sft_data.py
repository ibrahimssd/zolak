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
from prompts import build_messages  # noqa: E402

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
    "المعلومات اللي عندي من تجارب القروب ما بتجاوب سؤالك دا بالضبط، وما داير أألف ليك حاجة. "
    "أحسن تسأل في القروب أو تتأكد من {auth}.",
    "للأسف ما لقيت تجارب في القروب عن النقطة دي بالذات. عشان ما أديك معلومة غلط، اتواصل مع {auth}، "
    "وممكن تسأل الناس في القروب برضو.",
]
NOCONTEXT = [
    "ما عندي تجارب من القروب عن السؤال دا. أحسن حاجة تسأل {auth} مباشرة، أو تطرح السؤال في القروب.",
    "السؤال دا ما لقيت ليهو معلومة مؤكدة في تجارب القروب. اتأكد من {auth} عشان تاخد الإجابة الصاح.",
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
    args = ap.parse_args()

    rnd = random.Random(args.seed)
    proc = Path(args.proc)
    qa = [json.loads(l) for l in (proc / "qa.jsonl").open(encoding="utf-8")]
    by_topic: dict[str, list[dict]] = {}
    for r in qa:
        by_topic.setdefault(r["topic"], []).append(r)

    rnd.shuffle(qa)
    n_val = max(20, int(len(qa) * args.val))
    splits = {"val": qa[:n_val], "train": qa[n_val:]}
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
