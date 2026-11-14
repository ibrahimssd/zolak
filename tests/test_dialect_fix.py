"""Sudanese-dialect normalizer (space/dialect_fix.py), applied to the knowledge base, training targets and every
answer the Space sends."""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "space"))
from dialect_fix import build_vocab, sudanize_text  # noqa: E402

VOCAB = build_vocab(["بيعطي الورق", "بيعطي برضو", "بيحتاج ورق", "ما بيحتاج", "يحتاج كتير", "لقيت الورق", "لقيت شقة", "كانو هناك", "كانو جايين",
                     "ينصح بيها", "ينصح الناس", "بتأثر شديد", "بتأثر برضو"])


def fix(text: str) -> str:
    return sudanize_text(text, VOCAB, Counter())


def test_egyptian_negation():
    assert fix("ما يحتاجوش ورق") == "ما يحتاجو ورق"          # plural suffix: stem يحتاج is known
    assert fix("ما لقيتهاش") == "ما لقيتها"                   # object suffix
    assert fix("ماكانوش هناك") == "ما كانو هناك"              # joined form
    assert fix("ما ينصحش بيها") == "ما ينصح بيها"
    assert fix("ما بتأثرش") == "ما بتأثر"


def test_word_rules():
    assert fix("ما فيش مشكلة") == "ما في مشكلة"
    assert fix("هذا الموضوع") == "دا الموضوع"
    assert fix("هذه الورقة") == "دي الورقة"
    assert fix("البنك يعطيك إياه") == "البنك بيدّيك ليهو"
    assert fix("بيعطيك ورقة") == "بيدّيك ورقة"
    assert fix("البنك ما بيعطيكش ورق") == "البنك ما بيدّيك ورق"   # negation and word rule together


def test_untouched():
    assert fix("ما بفتش في الموضوع") == "ما بفتش في الموضوع"   # real verb ending in ش
    assert fix("ما بتعيش") == "ما بتعيش"
    assert fix("شبكة زين في السودان") == "شبكة زين في السودان"  # Zain network
    assert fix("https://www.daad.de/de/ ما Sperrkonto") == "https://www.daad.de/de/ ما Sperrkonto"
    assert fix("ما بيجيش") == "ما بيجيش"                       # stem unknown to the corpus: left alone
    assert fix("المشكلة مشي عادي") == "المشكلة مشي عادي"       # only whole words
