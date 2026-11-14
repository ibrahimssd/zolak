"""Sudanese-dialect normalizer: replaces common Egyptian, Gulf and MSA forms with Sudanese ones.

Word-level rules that keep the meaning. Shared by the data pipeline (src/ingest/postprocess.sudanize: teacher
outputs, audit, summaries, SFT targets) and the Space, which applies it to every answer it sends.

Not touched on purpose: «زين» (also the Zain mobile network) and «وش» (Sudanese "face") depend on context; Latin
script (URLs, German terms) never matches the Arabic-letter patterns.
"""
from __future__ import annotations

import re
from collections import Counter

_W = r"(?<![\w])"
_E = r"(?![\w])"
_AR = "[ء-ي]"   # Arabic letters (no digits, no Latin)


def _give(m: re.Match) -> str:  # Gulf «يعطيك» -> Sudanese «يدّيك»
    return f"{m.group(1)}دّي{m.group(2) or ''}"


RULES = [
    (re.compile(_W + r"ما ?فيش" + _E), "ما في"),
    (re.compile(_W + r"مش" + _E), "ما"),
    (re.compile(_W + r"مو" + _E), "ما"),
    (re.compile(_W + r"هذي" + _E), "دي"),
    (re.compile(_W + r"هذه" + _E), "دي"),
    (re.compile(_W + r"هذا" + _E), "دا"),
    (re.compile(_W + r"دلوقتي" + _E), "هسي"),
    (re.compile(_W + r"[إا]زاي" + _E), "كيف"),
    (re.compile(_W + r"كده" + _E), "كدا"),        # Sudanese spelling
    (re.compile(_W + r"وش" + _E), "شنو"),         # Gulf "what" (Sudanese «وش» = face only appears inflected: وشك، الوش)
    (re.compile(_W + r"شو" + _E), "شنو"),         # Levantine "what"
    (re.compile(_W + r"يعطيك [إا]ياه" + _E), "بيدّيك ليهو"),
    (re.compile(_W + r"(بي|بت|ي|ت)عطي(ك|كم|ه|ها|هم|ني)?" + _E), _give),
]
# Egyptian negation circumfix «ما ...ش» (also joined: «ماكانوش»)
NEG_SH = re.compile(_W + r"ما ?(" + _AR + r"{2,})ش" + _E)
# real words that end in ش: never strip
PROTECT = {"فتش", "بفتش", "يفتش", "تفتش", "نفتش", "بيفتش", "بتفتش", "عاش", "بعيش", "يعيش", "تعيش", "نعيش",
           "بيعيش", "بتعيش", "غش", "بغش", "يغش", "بيغش", "نقاش", "نقش", "رش", "كاش", "كش", "طش", "هش", "دش", "فاش",
           "بحش", "يحش", "عطش", "قماش", "جيش", "ريش", "عيش", "حشيش"}
_SUFFIXES = ("لكم", "لك", "لي", "هم", "ها", "هو", "كم", "نا", "وا", "و", "ه", "ك", "ي")


def build_vocab(texts: list[str]) -> Counter:
    return Counter(w for t in texts for w in re.findall(r"\w+", t))


def _known(stem: str, vocab: Counter, min_count: int) -> bool:
    """The stem, or a light variant of it, is a word of the corpus: with/without a leading «ب» (بيحتاج ↔ يحتاج),
    or without an object/plural suffix (يحتاجو -> يحتاج, لقيتها -> لقيت)."""
    cands = {stem, stem[1:] if stem.startswith("ب") else "ب" + stem}
    for s in list(cands):
        for suf in _SUFFIXES:
            if s.endswith(suf) and len(s) - len(suf) >= 2:
                cands.add(s[: -len(suf)])
    return any(vocab.get(c, 0) >= min_count for c in cands)


def sudanize_text(text: str, vocab: Counter, counts: Counter | None = None, min_count: int = 1) -> str:
    counts = counts if counts is not None else Counter()

    def strip_sh(m: re.Match) -> str:
        stem = m.group(1)
        if stem + "ش" in PROTECT or not _known(stem, vocab, min_count):
            return m.group(0)
        counts["neg_sh"] += 1
        return f"ما {stem}"   # «ما تحتاجش» -> «ما تحتاج», «ماكانوش» -> «ما كانو»

    # negation first, so the word rules also see the bare verb («ما بيعطيكش» -> «ما بيعطيك» -> «ما بيدّيك»)
    text = NEG_SH.sub(strip_sh, text)
    for rx, rep in RULES:
        text, n = rx.subn(rep, text)
        counts["rule"] += n
    return text
