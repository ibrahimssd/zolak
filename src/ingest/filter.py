"""Step 2b — normalize and filter anonymized messages.

Normalization (keeps Sudanese orthography; search-time normalization lives in the retriever):
  * remove tatweel, zero-width chars, WhatsApp *bold*/_italic_/~strike~ markers
  * collapse letter elongation (ههههههه -> هه, شكراااا -> شكراا) and emoji runs
  * collapse whitespace / blank lines

Each message gets `flags` (reasons) and `keep` (bool). Dropped categories:
  empty, emoji_only, low_info (greeting / thanks / laughter / dua only), membership (add-me requests),
  name_list, donation, exchange_offer (P2P EUR<->SDG offers), condolence, religious, promo, duplicate
Kept messages also get `is_question`.

Input : data/interim/anon.jsonl
Output: data/processed/messages.jsonl  (all messages with flags; downstream uses keep==True)
        data/interim/filter_review.txt (samples per flag for manual review)

Usage: python -m src.ingest.filter
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

AR = "؀-ۿ"
TATWEEL = "ـ"
ZERO_WIDTH = re.compile("[​-‍⁠﻿­]")
EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿\U0001F1E6-\U0001F1FF‍️⃣]")
WA_FORMAT = re.compile(r"(?<!\w)([*_~])(?=\S)(.+?)(?<=\S)\1(?!\w)", re.S)
ELONGATION = re.compile(r"([^\d\s])\1{2,}")  # 3+ repeats of a non-digit char -> 2
PLACEHOLDERS = re.compile(r"\[(?:اسم|رقم|رابط|ايميل)\]|@مستخدم")

# --- low-information phrases (whole message consists only of these) ---
LOW_INFO_PHRASES = [
    r"ال?سلام عليكم( و ?رحمة? الله( و ?بركاته)?)?", r"و ?عليكم ال?سلام( و ?رحمة? الله( و ?بركاته)?)?",
    r"سلام(ات)?", r"صباح الخير", r"صباح النور", r"مساء الخير", r"مساء النور", r"اهلا?(ن| و ?سهلا?)?",
    r"مرحبا?", r"هلا", r"شكرا?(ن)?( ج[ز]?يلا?)?( ليك| ليكم| ليكي| يا شباب| جدا| كتير| كثيرا?)?",
    r"(مشكور|مشكورة|مشكورين)( يا \S+)?", r"تسلم(ي|و)?( يا \S+)?", r"الله يسلمك|الله يسلمكم|الله يسلمك?ي",
    r"جزاك?(م|ي)? الله (كل )?خير(ا|اً)?", r"و ?اياك?(م|ي)?", r"بارك الله (فيك|فيكم|فيكي)", r"العفو", r"عفوا",
    r"تمام", r"اوك(ي)?", r"ok(ay)?", r"thanks?( you)?", r"thx", r"danke( schön)?", r"تم", r"حاضر",
    r"اب?مين", r"امين", r"آمين", r"ان ?شاء ?الله", r"ما ?شاء ?الله", r"الحمد ?لله", r"الله يوفقك?(م|ي)?",
    r"ربنا يوفقك?(م|ي)?", r"(ه|خ)+", r"ل+و+ل+", r"\.+", r"؟+", r"\?+", r"!+", r"يا شباب", r"يا جماعة",
    r"عيد(كم)? مبارك", r"كل عام و ?(انتو|انتم|انتي|انت) بخير", r"رمضان كريم", r"جمعة مباركة",
    r"تمام التمام", r"زي ما قال \S+", r"\+1", r"نفس السؤال", r"متابع(ة|ين)?", r"منتظر(ين)?",
]
LOW_INFO_RE = re.compile(
    r"^(?:\s*(?:" + "|".join(LOW_INFO_PHRASES) + r")\s*[.,،!؟?🙏]*\s*)+$", re.IGNORECASE
)

MEMBERSHIP_RE = re.compile(
    r"(ممكن|الرجاء|لو سمحت[وم]?|ارجو|أرجو|نرجو)?\s*(تضيف|اضافة|إضافة|اضافه|إضافه|تضيفو|تدخلو|ادخال|إدخال)"
    r"[^\n]{0,60}(القروب|الجروب|هنا|زميل|زميلة|الاخ|الأخ|الاخت|الأخت|اخونا|اختنا|صحبتي|صاحبي|طالب|طالبة)",
)
DONATION_RE = re.compile(
    r"(تبرع|التبرعات|الفزعة|فزعة|الحوجة|gofund|go ?fund ?me|paypal\.me|نفير|حملة دعم|دعم المرضى|علاج المريض)",
    re.IGNORECASE,
)
DONATION_SOFT_RE = re.compile(r"(ساهم|المساهمة|مساهمة)")
ACCOUNT_RE = re.compile(r"(رقم (ال)?حساب|IBAN|\[رقم\]|بنكك)", re.IGNORECASE)
# P2P currency exchange offers: "عندي 150 يورو بنكك", "مطلوب يورو مقابل بنكك"
EXCHANGE_RE = re.compile(r"(بنكك|فوري|جنيه سوداني|سعر الترس|سعر التراس)")
EXCHANGE_INTENT_RE = re.compile(r"(مقابل|مطلوب|متوفر|محتاج|محتاجة|عايز|عايزة|عاوز|عاوزه|عاوزة|عندي|عندنا|عايزين|بسعر)")
CONDOLENCE_RE = re.compile(
    r"(\bتنعي\b|\bينعى\b|\bتنعى\b|انتقل(ت)? (الى|إلى) رحمة الله|البقاء لله|[إا]نا لله\s*و\s*[إا]نا [إا]ليه راجعون|"
    r"مراسم (ال)?دفن|\bالعزاء\b|\bعزاء\b|\bالمرحوم(ة)?\b|\bالفقيد(ة)?\b|\bتوفي(ت)?\b|\bوفاة\b|\bالوفاة\b|"
    r"ربنا يرحم(ه|ها)|اللهم ارحم(ه|ها)|الله يرحم(ه|ها))",
)
RELIGIOUS_RE = re.compile(
    r"(صلى الله عليه وسلم|ﷺ|قال تعالى|قال رسول الله|حديث شريف|رواه (البخاري|مسلم)|سورة|زكاة الفطر|"
    r"الفطرة|الفديه|الفدية|خطبة|ليلة القدر|دعاء|الأذكار|اذكار|صيام|التكبير)",
)
PROMO_RE = re.compile(
    r"(تواصل معنا|للتواصل والاستفسار|للتسجيل|سجل الآن|سجل الان|عرض خاص|خصم \d|خصومات|كود خصم|احجز مقعدك|"
    r"انضم (الينا|إلينا)|join (my|our)|follow (us|me)|subscribe|اشترك|متابعة القناة|قناتنا|صفحتنا|"
    r"إعلان|اعلان|برومو|promo|discount)", re.IGNORECASE,
)
# short but meaningful answers inside a thread — never low_info
ANSWER_WORD_RE = re.compile(r"^\s*(لا|لأ|لاء|ايوا|ايوه|أيوه|أيوا|اي|نعم|yes|no|ja|nein|ممكن|صاح|صح|غلط)\s*[.!،,]*\s*$", re.I)
NUMBERED_LINE_RE = re.compile(r"^\s*\d{1,2}\s*[-.)]\s*\S")

QUESTION_WORDS = re.compile(
    r"(؟|\?|\bشنو\b|\bكيف\b|\bكيفن\b|\bوين\b|\bمتين\b|\bمنو\b|\bياتو\b|\bلي شنو\b|\bكم\b|\bهل\b|"
    r"في زول|فى زول|في حد|فيي? زول|عندو معلومة|عندو فكرة|عندو خبرة|بسأل|بسال|"
    r"عندي (سؤال|استفسار)|عندنا (سؤال|استفسار)|استفسار بسيط|استفسار عن|سؤال عن|سؤالي|محتاج(ة)? (معلومة|مساعدة)|"
    r"ممكن (حد|زول|تساعد|تفيد)|افيدونا|أفيدونا|تفيدوني|any(one|body)|does anyone|how (do|can|to)|is there|wo kann)",
    re.IGNORECASE,
)


def normalize(text: str) -> str:
    t = ZERO_WIDTH.sub("", text).replace(TATWEEL, "")
    t = WA_FORMAT.sub(r"\2", t)
    t = ELONGATION.sub(r"\1\1", t)
    t = re.sub(f"({EMOJI.pattern})(?:{EMOJI.pattern}){{2,}}", r"\1", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r" *\n *", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def content_words(text: str) -> list[str]:
    t = PLACEHOLDERS.sub(" ", EMOJI.sub(" ", text))
    return re.findall(r"[^\W_]+", t)  # letters/digits in any script (no Arabic punctuation)


def classify(text: str) -> list[str]:
    flags = []
    words = content_words(text)
    if not text.strip():
        return ["empty"]
    if not words:
        return ["emoji_only"]
    stripped = EMOJI.sub("", PLACEHOLDERS.sub("", text)).strip()
    if (LOW_INFO_RE.match(stripped) or (len(words) == 1 and len(words[0]) <= 2 and not words[0].isdigit())) \
            and not ANSWER_WORD_RE.match(stripped):
        flags.append("low_info")
    if MEMBERSHIP_RE.search(text) and len(words) < 40:
        flags.append("membership")
    lines = [l for l in text.splitlines() if l.strip()]
    numbered = [l for l in lines if NUMBERED_LINE_RE.match(l)]
    if len(numbered) >= 3 and sum(len(content_words(l)) <= 4 for l in numbered) >= 0.7 * len(numbered) \
            and text.count("[اسم]") >= 2:
        flags.append("name_list")
    if DONATION_RE.search(text) or (DONATION_SOFT_RE.search(text) and ACCOUNT_RE.search(text)):
        flags.append("donation")
    if EXCHANGE_RE.search(text) and EXCHANGE_INTENT_RE.search(text) and len(words) < 40 \
            and not re.search(r"[؟?]|كيف|طريقة|في زول عارف", text):
        flags.append("exchange_offer")
    if CONDOLENCE_RE.search(text):
        flags.append("condolence")
    if len(RELIGIOUS_RE.findall(text)) >= 2 or (RELIGIOUS_RE.search(text) and len(words) > 80):
        flags.append("religious")
    if len(PROMO_RE.findall(text)) >= 2 or (PROMO_RE.search(text) and text.count("[رابط]") >= 2):
        flags.append("promo")
    return flags


def is_question(text: str) -> bool:
    return bool(QUESTION_WORDS.search(text)) and len(content_words(text)) >= 3


DROP = {"empty", "emoji_only", "low_info", "membership", "name_list", "donation", "exchange_offer", "condolence",
        "religious", "promo", "duplicate"}


def run(msgs: list[dict]) -> list[dict]:
    seen: dict[str, int] = {}
    for m in msgs:
        m["text"] = normalize(m["text"])
        flags = classify(m["text"])
        key = re.sub(r"\s+", " ", m["text"]).strip()
        if len(key) > 40:
            if key in seen:
                flags.append("duplicate")
            else:
                seen[key] = m["id"]
        m["flags"] = flags
        m["keep"] = not (set(flags) & DROP)
        m["is_question"] = m["keep"] and is_question(m["text"])
    return msgs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="data/interim/anon.jsonl")
    ap.add_argument("--out", default="data/processed/messages.jsonl")
    args = ap.parse_args()

    msgs = run([json.loads(l) for l in open(args.input, encoding="utf-8")])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for m in msgs:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    flag_counts = Counter(fl for m in msgs for fl in m["flags"])
    kept = [m for m in msgs if m["keep"]]
    print(f"{len(msgs)} messages -> kept {len(kept)} ({len(kept) / len(msgs):.0%}), "
          f"questions {sum(m['is_question'] for m in kept)}")
    print("flags:", dict(flag_counts.most_common()))

    rnd = random.Random(0)
    with open("data/interim/filter_review.txt", "w", encoding="utf-8") as f:
        for fl in [*DROP, "kept_question", "kept_other"]:
            pool = ([m for m in kept if m["is_question"]] if fl == "kept_question" else
                    [m for m in kept if not m["is_question"]] if fl == "kept_other" else
                    [m for m in msgs if fl in m["flags"]])
            f.write(f"\n########## {fl} ({len(pool)})\n")
            for m in rnd.sample(pool, min(15, len(pool))):
                f.write(f"--- #{m['id']} {m['flags']}\n{m['text'][:400]}\n")


if __name__ == "__main__":
    main()
