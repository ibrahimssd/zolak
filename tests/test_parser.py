# pii-scan: ignore-file  (synthetic WhatsApp-format fixture with fake numbers)
from src.ingest.parse_whatsapp import parse

SAMPLE = """4/18/24, 13:16 - Messages and calls are end-to-end encrypted. Only people in this chat can read, listen to, or share them. Learn more.
6/20/21, 14:42 - +49 1521 0000000 created group "‎‫طلاب‬‎"
4/18/24, 13:16 - ~ Someone added you
4/18/24, 14:00 - +49 176 1111111: سلام عليكم
 بسأل عن الحساب المغلق
4/18/24, 14:05 - +249 91 222 3333: <Media omitted>
4/18/24, 14:06 - +249 91 222 3333: This message was deleted
4/18/24, 14:07 - Ali Test: Ali Test.vcf (file attached)
4/18/24, 14:08 - Ali Test: الرقم 11904 يورو <This message was edited>
4/18/24, 14:09 - +49 176 1111111: ملاحظة: ده مهم
"""


def test_parse_kinds_and_multiline():
    msgs = parse(SAMPLE.splitlines())
    kinds = [m.kind for m in msgs]
    assert kinds == ["system", "system", "system", "message", "media", "deleted", "contact_card", "message", "message"]
    assert msgs[3].text == "سلام عليكم\n بسأل عن الحساب المغلق"
    assert msgs[3].ts == "2024-04-18T14:00:00"
    assert msgs[1].ts.startswith("2021-06-20")


def test_edited_tag_and_colon_in_text():
    msgs = parse(SAMPLE.splitlines())
    assert msgs[7].edited and msgs[7].text == "الرقم 11904 يورو"
    # a colon inside the message text must not change the sender
    assert msgs[8].sender == "+49 176 1111111" and msgs[8].text == "ملاحظة: ده مهم"


def test_bidi_marks_removed():
    msgs = parse(SAMPLE.splitlines())
    assert "‎" not in msgs[1].text and "‫" not in msgs[1].text


def test_normalize_keeps_links_verbatim():
    from src.ingest.filter import normalize
    url = "https://www.daad.de/de/studium/Fintiba_blocked_account_info_"
    assert normalize(f"شوف الرابط دا {url} شكراااا") == f"شوف الرابط دا {url} شكراا"
    assert normalize("www.uni-assist.de كتيييير مهم") == "www.uni-assist.de كتيير مهم"
