"""Follow-up questions are searched together with the previous user question (space/retriever.retrieval_query)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "space"))
from retriever import retrieval_query  # noqa: E402

HIST = [{"role": "user", "content": "شنو المطلوب في كشف الحساب البنكي للسفارة؟"},
        {"role": "assistant", "content": "لازم كشف حساب آخر ٣ شهور ..."}]


def test_followup_word_adds_previous_question():
    assert retrieval_query("بالنسبه للاهلي لازم؟", HIST) == "شنو المطلوب في كشف الحساب البنكي للسفارة؟ بالنسبه للاهلي لازم؟"
    assert retrieval_query("طيب والرسوم كم تقريباً في السفارة دي هسي؟", HIST).startswith("شنو المطلوب")


def test_short_question_adds_previous_question():
    assert retrieval_query("وكم مدته؟", HIST).startswith("شنو المطلوب")


def test_standalone_question_unchanged():
    q = "كيف أجدد الإقامة في Ausländerbehörde وشنو الأوراق المطلوبة؟"
    assert retrieval_query(q, HIST) == q


def test_no_history_unchanged():
    assert retrieval_query("بالنسبه للاهلي لازم؟", []) == "بالنسبه للاهلي لازم؟"
    assert retrieval_query("بالنسبه للاهلي لازم؟", None) == "بالنسبه للاهلي لازم؟"


def test_question_word_starting_with_waw_is_not_a_followup():
    q = "وين بقدر أعمل Anmeldung في برلين بسرعة لو ما عندي عقد سكن؟"   # «وين» = where, a full question
    assert retrieval_query(q, HIST) == q
