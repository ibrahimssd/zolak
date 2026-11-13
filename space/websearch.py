"""Live web search fallback, restricted to official German domains.

Used only when neither the group knowledge nor the crawled official pages match the question. Questions are in
Sudanese Arabic while official pages are English/German, so the query is built from a keyword map (plus any Latin
terms the student typed). Results outside OFFICIAL_DOMAINS are discarded; at most 2 pages are fetched with short
timeouts, and the passage closest to the question is returned as a context record with source="web".
"""
from __future__ import annotations

import re
import time
from datetime import date
from functools import lru_cache
from urllib.parse import urlparse

OFFICIAL_DOMAINS = (
    "auswaertiges-amt.de", "diplo.de", "study-in-germany.de", "daad.de", "make-it-in-germany.com", "bamf.de",
    "anerkennung-in-deutschland.de", "uni-assist.de", "arbeitsagentur.de", "bundesregierung.de", "berlin.de",
    "studierendenwerke.de", "germany.info", "integrationsbeauftragte.de",
)

# Arabic (Sudanese) keywords -> German/English search terms
KEYWORDS = [
    (r"حساب (ال)?مغلق|حساب (ال)?مقفول|بلوك|blocked|sperrkonto", "Sperrkonto blocked account"),
    (r"فيزا|تأشيرة|تاشيرة|visa|visum", "student visa Germany"),
    (r"موافقة (ال)?أمنية|موافقة امنية", "visa security check processing time"),
    (r"إقامة|اقامة|aufenthalt|ausländerbehörde", "residence permit Aufenthaltstitel"),
    (r"تسجيل (ال)?سكن|anmeldung|بلدية|bürgeramt", "Anmeldung registration Bürgeramt"),
    (r"تأمين|تامين|صحي|krankenversicherung|insurance", "health insurance students Krankenversicherung"),
    (r"شغل|شتغل|عمل|وظيفة|werkstudent|minijob|ساعات|ساعة", "working while studying international students hours"),
    (r"معادلة|اعتراف|توثيق|anabin|anerkennung|zab", "recognition foreign degree Anerkennung"),
    (r"قبول|جامعة|ماستر|بكالوريوس|uni-assist|aps", "university admission international students"),
    (r"كولج|تحضيري|studienkolleg", "Studienkolleg preparatory course"),
    (r"لغة|(?<!\w)[أا]لماني(?!ا)|كورس|testdaf|goethe|telc", "German language requirements studies"),
    (r"لم الشمل|زوجة|زوجي|عيلة|عائلة|family", "family reunification Germany"),
    (r"سكن|شقة|wg|wohnung|سكن طلاب", "student accommodation Germany"),
    (r"opportunity|chancenkarte|كرت الفرصة|بطاقة الفرصة", "Chancenkarte opportunity card"),
    (r"بلو كارد|البطاقة الزرقاء|blue card|blaue karte", "EU Blue Card Germany"),
]


def build_query(question: str) -> str | None:
    q = question.lower()
    terms = [en for rx, en in KEYWORDS if re.search(rx, q, re.I)]
    latin = re.findall(r"[A-Za-zÄÖÜäöüß][A-Za-zÄÖÜäöüß\-]{3,}", question)
    if not terms and not latin:
        return None
    return " ".join(dict.fromkeys(terms + latin)) + " Germany"


def _official(url: str) -> str | None:
    host = (urlparse(url).hostname or "").lower().removeprefix("www.")
    return next((d for d in OFFICIAL_DOMAINS if host == d or host.endswith("." + d)), None)


@lru_cache(maxsize=256)
def _search(query: str) -> tuple[tuple[str, str], ...]:
    from ddgs import DDGS
    hits: list[tuple[str, str]] = []
    # try the full query worldwide, then German region, then a shorter query (DDG sometimes returns nothing)
    short = " ".join(query.split()[:4])
    for q, region in ((query, "wt-wt"), (query, "de-de"), (short, "wt-wt")):
        try:
            for r in DDGS().text(q, max_results=15, region=region):
                url = r.get("href") or r.get("url") or ""
                if _official(url) and url not in [u for u, _ in hits]:
                    hits.append((url, r.get("title", "")))
        except Exception:
            continue
        if hits:
            break
    return tuple(hits[:2])


@lru_cache(maxsize=256)
def _page_text(url: str) -> str:
    import requests
    import trafilatura
    r = requests.get(url, timeout=6, headers={"User-Agent": "Mozilla/5.0 (ZolakBot; Sudanese student helper)"})
    if r.status_code != 200:
        return ""
    return trafilatura.extract(r.text, include_links=False, favor_precision=True) or ""


def search_official(question: str, embed, max_chars: int = 900) -> list[dict]:
    """Return 0-1 context records (source="web") from official sites. Never raises."""
    t0 = time.time()
    try:
        query = build_query(question)
        if not query:
            return []
        hits = _search(query)
        best, best_sim = None, -1.0
        qv = embed([question])[0]
        for url, title in hits:
            text = _page_text(url)
            paras = [p for p in re.split(r"\n+", text) if len(p) > 80]
            if not paras:
                continue
            vecs = embed(paras[:60])
            sims = vecs @ qv
            i = int(sims.argmax())
            passage = " ".join(paras[i:i + 3])[:max_chars]
            if float(sims[i]) > best_sim:
                best_sim, best = float(sims[i]), {
                    "source": "web", "site": _official(url), "date": date.today().isoformat(),
                    "question": title or _official(url), "answer": passage, "facts": [], "links": [url]}
        print(f"web search: {query!r} -> {len(hits)} official hits, best sim {best_sim:.2f} in {time.time() - t0:.1f}s",
              flush=True)
        return [best] if best and best_sim >= 0.35 else []
    except Exception as e:  # network, rate limit, parsing: fall back to "not found"
        print(f"web search failed: {type(e).__name__}: {e}", flush=True)
        return []
