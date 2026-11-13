from src.ingest import pii
from src.ingest.anonymize import anonymize, build_name_lexicon
from src.ingest.filter import classify, is_question, normalize

SALT = b"\x01" * 32


def _rec(i, sender, text, kind="message"):
    return {"id": i, "ts": "2024-05-01T10:00:00", "sender": sender, "text": text, "edited": False, "kind": kind}


def scrub(text: str) -> str:
    counts: dict = {}
    return pii.scrub_numbers(pii.scrub_urls(pii.normalize_digits(text), counts), counts)


def test_phone_numbers_scrubbed():
    for t in ["كلمني +49 176 12345678", "رقمي 0176 12345678", "0912345678 ده رقمي",
              "00249912345678", "+٢٤٩ ٩١ ٢٣٤ ٥٦٧٨", "wa.me/4917612345678"]:
        out = scrub(t)
        assert not pii.find_pii(out), (t, out)
        assert "[رقم]" in out or "[رابط]" in out


def test_amounts_and_dates_survive():
    for t in ["المبلغ 11904 يورو في السنة", "الموعد 26.09.2025", "عندي 1500€", "سنة 2024/2025", "B1 و C1"]:
        assert scrub(t) == pii.normalize_digits(t), t


def test_urls_allowlist():
    out = scrub("شوف https://www.auswaertiges-amt.de/de/visa و https://chat.whatsapp.com/ABC و https://www.tum.de/x")
    assert "auswaertiges-amt.de" in out and "tum.de" in out
    assert "chat.whatsapp.com" not in out and "[رابط]" in out


def test_email_and_iban():
    out = scrub("ايميل test.user@gmail.com و IBAN DE89 3704 0044 0532 0130 00")
    assert "gmail" not in out and "3704" not in out


def test_anonymize_speakers_mentions_names():
    recs = [
        _rec(0, None, "~ Qamar Zed added +49 176 0000000", kind="system"),
        _rec(1, "+49 176 1111111", "سلام يا Qamar، كلم ⁨@⁨+49 176 2222222⁩ او اتصل 0176 33334444"),
        _rec(2, "+49 176 2222222", "تمام، ارسل لي على Qamar.Zed@web.de"),
        _rec(3, "Ahmed Berlin", "انا في Berlin"),
    ]
    out, totals, lex = anonymize(recs, SALT)
    texts = " ".join(m["text"] for m in out)
    assert "Qamar" not in texts and "2222222" not in texts and "33334444" not in texts
    assert "Berlin" in texts, "city tokens from contact names must not be scrubbed"
    assert out[0]["speaker"].startswith("U") and out[0]["speaker"] != out[1]["speaker"]
    assert out[0]["mentions"] == [out[1]["speaker"]]
    assert all(not pii.find_pii(m["text"]) for m in out)


def test_lexicon_ignores_junk():
    recs = [_rec(0, "Zizo9 Berlin", "x"), _rec(1, "😎", "y"), _rec(2, "Bad", "z")]
    full, tokens = build_name_lexicon(recs)
    assert "Berlin" not in tokens and "Bad" not in full and "😎" not in full
    assert "Zizo9" in tokens


def test_filter_flags():
    assert "low_info" in classify("شكرا جزيلا 🙏")
    assert "low_info" not in classify("لأ")  # a real answer
    assert "condolence" not in classify("السلام عليكم ورحمه الله، بسأل عن السكن")
    assert "condolence" not in classify("بالتوفيق ان شاء الله")
    assert "condolence" in classify("انا لله وانا اليه راجعون")
    assert "exchange_offer" in classify("محتاج بنكك مقابل 250 يورو")
    assert "exchange_offer" not in classify("في زول عارف طريقة احول بيها بنكك ليورو؟")
    assert is_question("في زول عارف مواعيد السفارة في برلين؟")
    assert not is_question("ربنا يرحمه و يغفر ليه")
    assert normalize("هههههههه *مهم* جداااا") == "هه مهم جداا"


def test_url_allowed_rejects_garbage():
    assert not pii.url_allowed("[رابط]")
    assert not pii.url_allowed("http://[not-an-ip]/x")
