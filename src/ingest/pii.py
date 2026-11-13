"""PII patterns  (pii-scan: ignore-file — this module defines the patterns)
 + scrubbing primitives shared by anonymize.py, tests and the CI scanner.

Placeholders (Arabic so the model sees natural-looking text):
    [رقم]   phone / account / ID number
    [ايميل] email
    [رابط]  non-allowlisted URL
    [اسم]   person name
    @مستخدم  @mention of a group member
"""
from __future__ import annotations

import re
from urllib.parse import urlparse

PH_NUM = "[رقم]"
PH_EMAIL = "[ايميل]"
PH_URL = "[رابط]"
PH_NAME = "[اسم]"
PH_MENTION = "@مستخدم"

# Arabic-Indic (U+0660..) and Extended/Persian (U+06F0..) digits -> ASCII.
DIGIT_TRANS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "0123456789" * 2)

# @mention as exported by WhatsApp: "@⁨~Name⁩" or "@⁨+49 176 ...⁩"
MENTION_ISOLATE_RE = re.compile("@⁨(.*?)⁩")
MENTION_DIGITS_RE = re.compile(r"@\+?\d[\d ]{6,}\d")
ISOLATES_RE = re.compile("[⁨⁩]")

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"'،]+", re.IGNORECASE)
# bare domains like "wa.me/49176..." or "chat.whatsapp.com/xyz" without scheme
BARE_SENSITIVE_URL_RE = re.compile(r"\b(?:wa\.me|chat\.whatsapp\.com|t\.me)/\S+", re.IGNORECASE)

IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){3,7}(?:[ ]?[A-Z0-9]{1,3})?\b")
# International phone: +49 / 0049 / +249 / 00249 ... followed by >=7 more digits (spaces, dashes, parens allowed)
INTL_PHONE_RE = re.compile(r"(?<![\w+])(?:\+|00)\s?\d{1,3}[\s\-./()]*(?:\d[\s\-./()]*){6,13}\d(?!\d)")
# Local phone: German 0151.., 030.., Sudanese 09.., 01.. — leading 0, total 9-13 digits
LOCAL_PHONE_RE = re.compile(r"(?<![\w.,/])0\d{2,4}[\s\-/]?\d{3,4}[\s\-]?\d{2,6}(?![\w.,/])")
# Any long digit run (account numbers, IDs, phones without separators)
LONG_DIGITS_RE = re.compile(r"(?<!\d)\d{9,}(?!\d)")
# Passport-like IDs: letter + 7-8 digits (Sudanese passports: P01234567)
PASSPORT_RE = re.compile(r"\b[A-Z]\d{7,8}\b")

# Official / institutional domains whose links are useful and non-personal.
ALLOWED_DOMAINS = {
    "auswaertiges-amt.de", "diplo.de", "daad.de", "uni-assist.de", "make-it-in-germany.com",
    "goethe.de", "kmk.org", "anabin.kmk.org", "bamf.de", "berlin.de", "service.berlin.de",
    "gesetze-im-internet.de", "hochschulstart.de", "hochschulkompass.de", "studienwahl.de",
    "studierendenwerk.de", "studentenwerk.de", "deutschland.de", "bundesregierung.de",
    "arbeitsagentur.de", "anerkennung-in-deutschland.de", "bafoeg.de", "mygermanuniversity.com",
    "study-in-germany.de", "daad.org", "tk.de", "aok.de", "barmer.de", "dak.de",
    "fintiba.com", "expatrio.com", "coracle.de", "wise.com", "vfsglobal.com", "tlscontact.com",
    "testdaf.de", "telc.net", "ielts.org", "ets.org", "zab.kmk.org", "europa.eu",
    "ksavisa.sa", "visa.mofa.gov.sa", "vc.tasheer.com", "egyptvisa.gov.eg",
    "sudaneseingermany.de", "uni-assist.de", "ausbildung.de", "stepstone.de", "indeed.com",
}
UNIVERSITY_DOMAIN_RE = re.compile(
    r"(^|\.)(uni-[\w-]+|tu-[\w-]+|tum|rwth-aachen|kit|fh-[\w-]+|hs-[\w-]+|th-[\w-]+|"
    r"hochschule-[\w-]+|[\w-]*university[\w-]*|[\w-]*universitaet[\w-]*)\.(de|eu|com|org)$"
    r"|\.edu$|\.ac\.[a-z]{2}$|\.edu\.[a-z]{2}$"
)


def normalize_digits(text: str) -> str:
    return text.translate(DIGIT_TRANS)


def url_allowed(url: str) -> bool:
    u = url if "://" in url else "http://" + url
    host = (urlparse(u).hostname or "").lower().removeprefix("www.")
    if not host:
        return False
    if UNIVERSITY_DOMAIN_RE.search(host):
        return True
    return any(host == d or host.endswith("." + d) for d in ALLOWED_DOMAINS)


def scrub_urls(text: str, counts: dict) -> str:
    def repl(m: re.Match) -> str:
        url = m.group(0).rstrip(").,؛!؟?")
        tail = m.group(0)[len(url):]
        if url_allowed(url):
            counts["url_kept"] = counts.get("url_kept", 0) + 1
            return m.group(0)
        counts["url"] = counts.get("url", 0) + 1
        return PH_URL + tail

    text = URL_RE.sub(repl, text)
    text, n = BARE_SENSITIVE_URL_RE.subn(PH_URL, text)
    counts["url"] = counts.get("url", 0) + n
    return text


def scrub_numbers(text: str, counts: dict) -> str:
    """Emails, IBANs, phones, passport IDs, long digit runs. Protects allowlisted URLs."""
    # Protect URLs (their digits are not PII) by swapping them out temporarily.
    kept: list[str] = []

    def stash(m: re.Match) -> str:
        kept.append(m.group(0))
        return f"\x00{len(kept) - 1}\x00"

    text = URL_RE.sub(stash, text)
    for name, rx, ph in (
        ("email", EMAIL_RE, PH_EMAIL),
        ("iban", IBAN_RE, PH_NUM),
        ("phone", INTL_PHONE_RE, PH_NUM),
        ("phone", LOCAL_PHONE_RE, PH_NUM),
        ("long_digits", LONG_DIGITS_RE, PH_NUM),
        ("passport", PASSPORT_RE, PH_NUM),
    ):
        text, n = rx.subn(ph, text)
        if n:
            counts[name] = counts.get(name, 0) + n
    return re.sub("\x00(\\d+)\x00", lambda m: kept[int(m.group(1))], text)


def find_pii(text: str) -> list[str]:
    """Return PII-looking substrings still present (used by tests / CI leak scan)."""
    t = normalize_digits(text)
    t_wo_urls = URL_RE.sub(" ", t)
    hits: list[str] = []
    for rx in (EMAIL_RE, INTL_PHONE_RE, LOCAL_PHONE_RE, LONG_DIGITS_RE, IBAN_RE, MENTION_ISOLATE_RE):
        hits += [m.group(0) for m in rx.finditer(t_wo_urls)]
    hits += [u for u in URL_RE.findall(t) if not url_allowed(u.rstrip(").,؛!؟?"))]
    hits += BARE_SENSITIVE_URL_RE.findall(t)
    return hits
