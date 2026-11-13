"""Zolna — Hugging Face Space (free CPU): hybrid retrieval over the community KB + llama.cpp generation.

Environment (Space settings → Variables and secrets):
  HF_TOKEN        secret; read private model/dataset, write feedback
  MODEL_REPO      private model repo holding the GGUF            (e.g. <user>/zolna-models)
  MODEL_FILE      GGUF file name                                 (default ALLaM-7B-Instruct-preview-Q4_K_M.gguf)
  DATA_REPO       private dataset repo with index/{kb.faiss,kb.jsonl,config.json}
  FEEDBACK_REPO   private dataset repo for 👍/👎 feedback (optional)
Local testing: LOCAL_MODEL=/path/model.gguf LOCAL_INDEX=/path/index python app.py

API (used by the GitHub Pages site via @gradio/client):
  /ask       (question: str, history: list[{"role","content"}]) -> streamed answer text;
             the final chunk contains SOURCES_SEP followed by a short sources line
  /feedback  (question: str, answer: str, rating: "up"|"down") -> "ok"
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path

import gradio as gr
from huggingface_hub import CommitScheduler, hf_hub_download, snapshot_download
from llama_cpp import Llama

from prompts import build_messages
from retriever import HybridRetriever

TOKEN = os.environ.get("HF_TOKEN")
MODEL_REPO = os.environ.get("MODEL_REPO", "")
MODEL_FILE = os.environ.get("MODEL_FILE", "ALLaM-7B-Instruct-preview-Q4_K_M.gguf")
DATA_REPO = os.environ.get("DATA_REPO", "")
FEEDBACK_REPO = os.environ.get("FEEDBACK_REPO", "")
N_THREADS = int(os.environ.get("N_THREADS", os.cpu_count() or 2))
# MIN_SIM calibrated on the raw-thread KB: relevant top hits 0.52-0.72, off-topic mostly 0.34-0.51
TOP_K, MIN_SIM, MAX_TOKENS, HISTORY_TURNS = 2, 0.55, 220, 2
SOURCES_SEP = "\n\n---\n"

# ---------------------------------------------------------------- load
t0 = time.time()
model_path = os.environ.get("LOCAL_MODEL") or hf_hub_download(MODEL_REPO, MODEL_FILE, token=TOKEN)
index_dir = os.environ.get("LOCAL_INDEX") or str(
    Path(snapshot_download(DATA_REPO, repo_type="dataset", allow_patterns=["index/*"], token=TOKEN)) / "index")
retriever = HybridRetriever(index_dir)
llm = Llama(model_path=model_path, n_ctx=4096, n_threads=N_THREADS, n_batch=512, verbose=False)
IS_QWEN3 = "qwen3" in Path(model_path).name.lower()
print(f"loaded model + {len(retriever.records)} KB records in {time.time() - t0:.0f}s (threads={N_THREADS})")

scheduler = None
FEEDBACK_DIR = Path("feedback")
FEEDBACK_DIR.mkdir(exist_ok=True)
FEEDBACK_FILE = FEEDBACK_DIR / f"feedback-{uuid.uuid4()}.jsonl"
if FEEDBACK_REPO and TOKEN:
    scheduler = CommitScheduler(repo_id=FEEDBACK_REPO, repo_type="dataset", folder_path=FEEDBACK_DIR,
                                path_in_repo="data", every=10, private=True, token=TOKEN)


# ---------------------------------------------------------------- core
def _clean_history(history: list | None) -> list[dict]:
    msgs = []
    for m in (history or [])[-2 * HISTORY_TURNS:]:
        if isinstance(m, dict) and m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str):
            msgs.append({"role": m["role"], "content": m["content"].split(SOURCES_SEP)[0][:1500]})
    return msgs


def _sources_line(recs: list[dict]) -> str:
    if not recs:
        return "ℹ️ معلومة عامة، ما من تجارب القروب — اتأكد من الجهات الرسمية."
    months = sorted({r["date"][:7] for r in recs}, reverse=True)
    return "📅 من تجارب أعضاء القروب (" + "، ".join(months) + ") — المعلومات بتتغير، اتأكد من الجهات الرسمية."


def answer(question: str, history: list | None = None):
    question = re.sub(r"\s+", " ", (question or "")).strip()[:1000]
    if not question:
        yield "اكتب سؤالك 🙂"
        return
    hits = [h for h in retriever.search(question, k=TOP_K) if h.dense_sim >= MIN_SIM]
    recs = [h.rec for h in hits]
    msgs = build_messages(question, recs, _clean_history(history))
    if IS_QWEN3:
        msgs[-1]["content"] += " /no_think"
    text = ""
    for chunk in llm.create_chat_completion(messages=msgs, stream=True, temperature=0.3, top_p=0.9,
                                            max_tokens=MAX_TOKENS, repeat_penalty=1.05):
        delta = chunk["choices"][0]["delta"].get("content")
        if delta:
            text += delta
            yield text
    yield text.strip() + SOURCES_SEP + _sources_line(recs)


def feedback(question: str, answer_text: str, rating: str) -> str:
    if rating not in ("up", "down"):
        return "ignored"
    row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "rating": rating,
           "question": (question or "")[:1000], "answer": (answer_text or "")[:3000], "model": MODEL_FILE}
    if scheduler:
        with scheduler.lock:
            with FEEDBACK_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return "ok"


# ---------------------------------------------------------------- UI
EXAMPLES = [
    "الحساب المغلق Sperrkonto بفتحو كيف؟",
    "الموافقة الأمنية للفيزا بتاخد قدر شنو؟",
    "جوازي السوداني قرب يخلص وانا في ألمانيا، اجدده وين؟",
    "ممكن اشتغل وانا طالب؟ كم ساعة مسموح؟",
]
HEADER = """## زولنا 🇸🇩 — مساعد الطلاب السودانيين في ألمانيا
بيجاوب من تجارب أعضاء قروب «طلاب السودان في ألمانيا» (بعد حذف أي معلومات شخصية).
⚠️ دي تجارب ناس ما استشارة قانونية — اتأكد دايماً من الجهات الرسمية."""

with gr.Blocks(title="Zolna", theme=gr.themes.Soft()) as demo:
    gr.Markdown(HEADER, rtl=True)
    gr.ChatInterface(
        fn=answer, type="messages", examples=EXAMPLES, cache_examples=False,
        chatbot=gr.Chatbot(type="messages", rtl=True, height=480),
        textbox=gr.Textbox(placeholder="اكتب سؤالك هنا…", rtl=True),
    )
    with gr.Row(visible=False):  # API-only endpoints for the GitHub Pages front-end
        q_in, h_in, a_out = gr.Textbox(), gr.JSON(), gr.Textbox()
        fq, fa, fr, f_out = gr.Textbox(), gr.Textbox(), gr.Textbox(), gr.Textbox()
    q_in.submit(answer, [q_in, h_in], a_out, api_name="ask")
    fr.submit(feedback, [fq, fa, fr], f_out, api_name="feedback")

demo.queue(default_concurrency_limit=1, max_size=40)

if __name__ == "__main__":
    demo.launch(server_name=os.environ.get("HOST", "0.0.0.0"), server_port=int(os.environ.get("PORT", 7860)))
