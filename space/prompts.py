"""Chatbot prompt — the single source of truth used by the Space, the bake-off and SFT data generation.

Context records carry their source:
  group     (source "thread"/"post"/"raw_*")  experience shared in the WhatsApp group
  official  (source "official")                a page of an official German website, crawled monthly
  web       (source "web")                     a live web search result from an official domain
"""
from __future__ import annotations

NOT_IN_GROUP = "ما لقيتها في القروب، لكن حرم ما بترجع ساي"

SYSTEM_PROMPT = f"""إنت "زولك"، مساعد بيساعد الطلاب والقادمين الجداد من السودان في ألمانيا.
- اتكلم بالعامية السودانية، بلطف وباختصار (٢-٦ جمل)، وخلي المصطلحات الألمانية زي ما هي (Sperrkonto, Anmeldung ...).
- كل معلومة مكتوب جنبها مصدرها وتاريخها: "من القروب" أو "موقع رسمي" أو "بحث في الإنترنت".
- لو المعلومة من القروب: قول إنها من تجارب الناس في القروب واذكر تاريخها بالضبط لأنو المعلومات بتتغير.
- لو المعلومة ما من القروب (موقع رسمي أو بحث في الإنترنت): ابدا بوضوح بـ «{NOT_IN_GROUP}»،
  وبعدين جاوب من النص واذكر اسم الموقع والرابط.
- ما تذكر أي تاريخ أو رقم أو رابط ما مكتوب في المعلومات، ولو المعلومات ما بتجاوب السؤال قول كدا بوضوح وما تألف.
- في مواضيع الفيزا والإقامة والقروش والقانون: دي معلومات للمساعدة ما استشارة قانونية، ونصح الزول يتأكد من الجهة
  الرسمية (السفارة، Ausländerbehörde، الجامعة ...).
- ما تدي أي معلومات شخصية عن أي زول."""

NO_CONTEXT_NOTE = "ما في معلومات من القروب ولا من المواقع الرسمية عن السؤال ده."


def _cut(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + " …"


def source_label(r: dict) -> str:
    src = r.get("source", "")
    if src == "official":
        return f"موقع رسمي: {r.get('site', '')}"
    if src == "web":
        return f"بحث في الإنترنت: {r.get('site', '')}"
    return "من القروب"


def format_context(records: list[dict], max_facts: int = 4, max_q: int = 160, max_a: int = 420) -> str:
    """Compact, dated, source-labelled context block (keeps prompt processing cheap)."""
    if not records:
        return NO_CONTEXT_NOTE
    parts = []
    for i, r in enumerate(records, 1):
        limit = 900 if r.get("source") == "web" else max_a   # web passages are raw page text
        lines = [f"[{i}] ({r['date'][:7]} · {source_label(r)}) س: {_cut(r['question'], max_q)}",
                 f"ج: {_cut(r['answer'], limit)}"]
        facts = r.get("facts", [])[:max_facts]
        if facts:
            lines.append("حقائق: " + " | ".join(facts))
        if r.get("links"):
            lines.append("روابط: " + " ".join(r["links"][:2]))
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def build_messages(question: str, records: list[dict], history: list[dict] | None = None) -> list[dict]:
    user = f"المعلومات المتاحة:\n{format_context(records)}\n\nسؤال الطالب: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, *(history or []), {"role": "user", "content": user}]
