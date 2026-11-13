"""Cheap heuristic dialect meter for Sudanese Arabic (used for teacher QA, model bake-off and eval).

score(text) -> {"sd": n Sudanese markers, "other": n Gulf/Egyptian/Levantine markers, "foreign": n CJK/... chars}
Not a classifier — a sanity metric to compare prompts/models on the same inputs. Human review stays the judge.

Usage: python -m src.train.dialect file.jsonl [--field answer] [--path output.answer]
"""
from __future__ import annotations

import argparse
import json
import re

W = r"(?<!\w)"
E = r"(?!\w)"

SUDANESE = [
    "عشان", "علشان", "داير", "دايرة", "دايره", "دايرين", "زول", "زولة", "ياتو", "ياتا", "شنو", "كدا", "برضو", "برضه",
    "هسي", "هسع", "هسه", "ساي", "شديد", "بتاع", "بتاعت", "بتاعة", "حقت", "قروش", "ياخ", "قبيل", "عديل", "تب",
    "بتقدر", "ما بتحتاج", "بكون", "بتكون", "حتلقى", "لسه", "لسع", "ليك", "ليهو", "معاك", "انتو", "نحنا", "ديل",
    "دي", "ده", "دا", "بس", "ممكن تمشي", "اتصل", "كويس", "ساهل", "صعب شوية", "شوية",
]
OTHER = [
    # Gulf
    "وش", "هذي", "هذا", "مو", "ابغى", "أبغى", "تبغى", "يعطيك", "إياه", "اياه", "وايد", "زين", "حيل",
    # Egyptian
    "مش", "دلوقتي", "إزاي", "ازاي", "إيه", "أوي", "اوي", "كده",
    # Levantine
    "هيك", "بدك", "بدي", "شو", "منيح", "هلق", "هلأ",
]
# Egyptian -ش negation (ما تبقاش، ما يضيعش)
OTHER_PATTERNS = [re.compile(r"(?<!\w)ما \w{2,}ش(?!\w)")]
FOREIGN = re.compile("[Ѐ-ӿ֐-׿ऀ-෿぀-ヿ㐀-鿿가-힯]")
# Note: "كده"/"ده" are shared Sudanese/Egyptian; "كده" counted as other, "كدا"/"ده" as Sudanese — keep it rough.

SD_RE = re.compile(W + "(?:" + "|".join(map(re.escape, sorted(SUDANESE, key=len, reverse=True))) + ")" + E)
OTHER_RE = re.compile(W + "(?:" + "|".join(map(re.escape, sorted(OTHER, key=len, reverse=True))) + ")" + E)


def score(text: str) -> dict:
    other = len(OTHER_RE.findall(text)) + sum(len(p.findall(text)) for p in OTHER_PATTERNS)
    return {"sd": len(SD_RE.findall(text)), "other": other, "foreign": len(FOREIGN.findall(text))}


def summarize(texts: list[str]) -> dict:
    s = [score(t) for t in texts]
    n = max(len(s), 1)
    return {
        "n": len(s),
        "sd_per_text": round(sum(x["sd"] for x in s) / n, 2),
        "other_per_text": round(sum(x["other"] for x in s) / n, 2),
        "texts_with_other": sum(x["other"] > 0 for x in s),
        "texts_with_foreign": sum(x["foreign"] > 0 for x in s),
    }


def _get(rec: dict, path: str):
    for k in path.split("."):
        rec = rec.get(k) if isinstance(rec, dict) else None
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--path", default="output.answer", help="dotted path to the text field")
    args = ap.parse_args()
    for f in args.files:
        texts = [t for l in open(f, encoding="utf-8") if isinstance(t := _get(json.loads(l), args.path), str) and t]
        print(f, summarize(texts))


if __name__ == "__main__":
    main()
