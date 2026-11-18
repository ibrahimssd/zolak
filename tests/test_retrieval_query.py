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


# --- accepting a topic the last answer offered («في نقاش عن X لو بيفيدك») -------------------------------------
from retriever import accepted_suggestion  # noqa: E402


def _offer(text):
    return [{"role": "user", "content": "هل الشنطة الكبيرة مسموحة؟"}, {"role": "assistant", "content": text}]


OFFER = "ما لقيت نقاش في القروب عن السؤال دا. في نقاش عن شروط الشنطة الظهر مع AJet وPegasus لو بيفيدك."


def test_yes_reply_takes_the_offered_topic():
    for reply in ("تمام جيبها", "ايوه", "أيوه كلمني عنه", "واصل", "yes", "اكيد بيفيدني", "تمام"):
        assert accepted_suggestion(reply, _offer(OFFER)) == "شروط الشنطة الظهر مع AJet وPegasus", reply


def test_offer_variants():
    h = _offer("ما لقيت نقاش عن السؤال دا. في نقاش عن شروط التأمين الصحي للعائلة ودخول الأهل بفيزا سياحية، لو بيفيدك.")
    assert accepted_suggestion("ايوه", h) == "شروط التأمين الصحي للعائلة ودخول الأهل بفيزا سياحية"
    h = _offer("ما عندي معلومات عن المدة. في نقاشات عن شروط الجنسية والرسوم لو بيفيدك.")
    assert accepted_suggestion("ايوا", h) == "شروط الجنسية والرسوم"


def test_no_offer_or_new_question():
    assert accepted_suggestion("ايوه", _offer("لازم كشف حساب آخر ٣ شهور.")) is None
    assert accepted_suggestion("كيف أجدد الإقامة في Ausländerbehörde وشنو الأوراق المطلوبة؟", _offer(OFFER)) is None
    assert accepted_suggestion("بالنسبه للاهلي لازم؟", _offer(OFFER)) is None
    assert accepted_suggestion("ايوه", []) is None
