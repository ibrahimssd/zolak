import re

from src.ingest.distill import SCHEMA, TOPICS, format_thread
from src.ingest.postprocess import gate

GOOD = {
    "useful": True, "reason": "answered", "topic": "blocked_account_money",
    "question": "الحساب المغلق بكم في السنة؟",
    "answer": "على حسب الناس في القروب، المبلغ حوالي 11904 يورو في السنة وبتفتحو عن طريق Fintiba او Expatrio.",
    "facts": ["مبلغ Sperrkonto حوالي 11904 يورو في السنة"], "german_terms": ["Sperrkonto"],
    "official_links": [], "confidence": "high", "time_sensitive": True,
}


def test_gate_accepts_good():
    assert gate(GOOD, None) is None


def test_gate_rejects():
    assert gate({**GOOD, "useful": False}, None) == "not_useful"
    assert gate({**GOOD, "confidence": "low"}, None) == "low_confidence"
    assert gate({**GOOD, "answer": "ايوا"}, None) == "too_short"
    assert gate({**GOOD, "answer": GOOD["answer"] + " اسأل [اسم]"}, None) == "placeholder"
    assert gate({**GOOD, "answer": GOOD["answer"] + " كلمو على +49 176 12345678"}, None) == "pii_pattern"
    name_re = re.compile("Qamar Zed", re.I)
    assert gate({**GOOD, "answer": GOOD["answer"] + " قالت qamar zed"}, name_re) == "member_name"


def test_gate_foreign_script():
    assert gate({**GOOD, "answer": GOOD["answer"] + " 至少 6 شهور"}, None) == "foreign_script"


def test_schema_content_before_labels():
    keys = list(SCHEMA["properties"])
    assert keys.index("answer") < keys.index("topic") < keys.index("time_sensitive")


def test_schema_topics_and_format():
    assert SCHEMA["properties"]["topic"]["enum"] == TOPICS
    t = {"date": "2025-01-02", "messages": [
        {"role": "السائل", "part": "question", "ts": "2025-01-02T10:05:00", "text": "في زول عارف؟"},
        {"role": "عضو1", "part": "reply", "ts": "2025-01-02T10:07:00", "text": "ايوا"},
    ]}
    s = format_thread(t)
    assert "2025-01-02" in s and "[السائل] (سؤال) 10:05" in s and "[عضو1] (رد) 10:07" in s


def test_sudanize_rules():
    from collections import Counter
    from src.ingest.postprocess import build_vocab, sudanize
    vocab = build_vocab(["ما تحتاج تمشي", "تحتاج ورق"])
    c = Counter()
    assert sudanize("ما فيش مشكلة", vocab, c) == "ما في مشكلة"
    assert sudanize("ما تحتاجش ورق", vocab, c) == "ما تحتاج ورق"
    assert sudanize("ما بتفتش", vocab, c) == "ما بتفتش"            # stem not in corpus: untouched
    assert sudanize("المشكلة مشي عادي", vocab, c) == "المشكلة مشي عادي"  # only whole words
