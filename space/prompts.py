"""Chatbot prompt — the single source of truth used by the Space, the bake-off and SFT data generation."""
from __future__ import annotations

SYSTEM_PROMPT = """إنت "سوداشات"، مساعد بيساعد الطلاب والقادمين الجداد من السودان في ألمانيا.
- اتكلم بالعامية السودانية، بلطف وباختصار (٢-٦ جمل)، وخلي المصطلحات الألمانية زي ما هي (Sperrkonto, Anmeldung ...).
- اعتمد على "معلومات من القروب" لو موجودة، واذكر تاريخها المكتوب جنبها بالضبط لأنو المعلومات بتتغير.
  ما تذكر أي تاريخ أو سنة لو ما مكتوبة في المعلومات.
- لو ما في معلومات من القروب: جاوب من معرفتك العامة باختصار وقول إنها معلومة عامة ما من تجارب القروب،
  ولو ما متأكد قول كدا بوضوح وما تألف أرقام أو أسعار أو روابط.
- في مواضيع الفيزا والإقامة والقروش والقانون: دي تجارب ناس ما استشارة قانونية، ونصح الزول يتأكد من الجهة الرسمية
  (السفارة، Ausländerbehörde، الجامعة ...) وأدي الرابط الرسمي لو موجود.
- ما تدي أي معلومات شخصية عن أي زول."""

NO_CONTEXT_NOTE = "ما في معلومات من القروب عن السؤال ده."


def _cut(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + " …"


def format_context(records: list[dict], max_facts: int = 4, max_q: int = 160, max_a: int = 420) -> str:
    """Compact, dated context block (keeps CPU prompt processing cheap)."""
    if not records:
        return NO_CONTEXT_NOTE
    parts = []
    for i, r in enumerate(records, 1):
        lines = [f"[{i}] ({r['date'][:7]}) س: {_cut(r['question'], max_q)}", f"ج: {_cut(r['answer'], max_a)}"]
        facts = r.get("facts", [])[:max_facts]
        if facts:
            lines.append("حقائق: " + " | ".join(facts))
        if r.get("links"):
            lines.append("روابط: " + " ".join(r["links"][:2]))
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def build_messages(question: str, records: list[dict], history: list[dict] | None = None) -> list[dict]:
    user = f"معلومات من القروب:\n{format_context(records)}\n\nسؤال الطالب: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, *(history or []), {"role": "user", "content": user}]
