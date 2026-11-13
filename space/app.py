"""Zolak — Hugging Face Space (free CPU): hybrid retrieval over the community KB + llama.cpp generation.

Environment (Space settings → Variables and secrets):
  HF_TOKEN        secret; read private model/dataset, write feedback
  MODEL_REPO      private model repo holding the GGUF            (e.g. <user>/zolak-models)
  MODEL_FILE      GGUF file name                                 (default zolak-allam-v1-Q4_K_M.gguf, the LoRA fine-tune)
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

# Pin every thread pool BEFORE the libraries load. Inside a container the libraries see all host cores
# (e.g. 384) while the CPU quota is 2: OpenMP (faiss), the tokenizers pool and torch would start hundreds of
# busy-waiting threads and starve llama.cpp (measured: no answer after 8 min instead of ~1 min).
def _available_cpus() -> int:
    """CPUs this process may really use: the container's CFS quota (cgroup v2/v1), else the affinity mask.
    os.cpu_count() reports the host's cores (16 on a Hugging Face cpu-basic Space with a 2-vCPU quota)."""
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()[:2]
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    try:
        quota = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
        period = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
        if quota > 0:
            return max(1, quota // period)
    except (OSError, ValueError):
        pass
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return os.cpu_count() or 2


_THREADS = os.environ.get("N_THREADS") or str(_available_cpus())
os.environ["N_THREADS"] = _THREADS
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_var, _THREADS)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import gradio as gr
from huggingface_hub import CommitScheduler, hf_hub_download, snapshot_download

from prompts import build_messages
from retriever import HybridRetriever

TOKEN = os.environ.get("HF_TOKEN")
# BACKEND: "zerogpu" = full-precision model with PyTorch on Hugging Face ZeroGPU (included in PRO, quota-based)
#          "llamacpp" = 4-bit GGUF on CPU (any machine)
BACKEND = os.environ.get("BACKEND", "llamacpp")
MODEL_REPO = os.environ.get("MODEL_REPO", "")
MODEL_FILE = os.environ.get("MODEL_FILE", "zolak-allam-v2-Q4_K_M.gguf")     # llamacpp
MODEL_SUBDIR = os.environ.get("MODEL_SUBDIR", "zolak-allam-v2")             # zerogpu (safetensors folder)
DATA_REPO = os.environ.get("DATA_REPO", "")
FEEDBACK_REPO = os.environ.get("FEEDBACK_REPO", "")
N_THREADS = int(_THREADS)
try:  # torch's own pools (intra-op and inter-op) follow the same limit
    import torch
    torch.set_num_threads(N_THREADS)
    torch.set_num_interop_threads(1)
except (ImportError, RuntimeError):
    pass
# MIN_SIM calibrated on the raw-thread KB: relevant top hits 0.52-0.72, off-topic mostly 0.34-0.51
TOP_K, MIN_SIM, MAX_TOKENS, HISTORY_TURNS = 2, 0.55, 220, 2
SOURCES_SEP = "\n\n---\n"

# ---------------------------------------------------------------- load
t0 = time.time()
index_dir = os.environ.get("LOCAL_INDEX") or str(
    Path(snapshot_download(DATA_REPO, repo_type="dataset", allow_patterns=["index/*"], token=TOKEN)) / "index")
retriever = HybridRetriever(index_dir)   # embedder stays on CPU: GPU quota is spent only on generation

if BACKEND == "zerogpu":
    from threading import Thread

    import spaces
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

    model_dir = os.environ.get("LOCAL_MODEL") or str(
        Path(snapshot_download(MODEL_REPO, allow_patterns=[f"{MODEL_SUBDIR}/*"], token=TOKEN)) / MODEL_SUBDIR)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    hf_model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16)
    hf_model.to("cuda")  # on ZeroGPU the weights move to the GPU when one is attached to a call

    @spaces.GPU(duration=60)
    def generate(msgs: list[dict]):
        ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to("cuda")
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(input_ids=ids, attention_mask=torch.ones_like(ids), streamer=streamer, max_new_tokens=MAX_TOKENS,
                      do_sample=True, temperature=0.3, top_p=0.9, repetition_penalty=1.05)
        Thread(target=hf_model.generate, kwargs=kwargs).start()
        yield from streamer
else:
    from llama_cpp import Llama

    model_path = os.environ.get("LOCAL_MODEL") or hf_hub_download(MODEL_REPO, MODEL_FILE, token=TOKEN)
    llm = Llama(model_path=model_path, n_ctx=4096, n_threads=N_THREADS, n_batch=512, verbose=False)
    is_qwen3 = "qwen3" in Path(model_path).name.lower()

    def generate(msgs: list[dict]):
        if is_qwen3:
            msgs[-1]["content"] += " /no_think"
        for chunk in llm.create_chat_completion(messages=msgs, stream=True, temperature=0.3, top_p=0.9,
                                                max_tokens=MAX_TOKENS, repeat_penalty=1.05):
            delta = chunk["choices"][0]["delta"].get("content")
            if delta:
                yield delta

print(f"loaded {BACKEND} model + {len(retriever.records)} KB records in {time.time() - t0:.0f}s (threads={N_THREADS})")
BUSY = "⏳ زولك عليه ضغط شديد هسي (حصة الـ GPU اليومية خلصت). جرب تاني بعد شوية."

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
    text = ""
    for attempt in (1, 2):
        try:
            for piece in generate(msgs):
                text += piece
                yield text
            break
        except Exception as e:  # ZeroGPU: quota used up or no GPU free right now (never billed)
            print(f"generation error (attempt {attempt}): {type(e).__name__}: {e}", flush=True)
            transient = "quota" not in str(e).lower() and not text
            if attempt == 1 and transient:
                time.sleep(2)
                continue
            # every final message ends with SOURCES_SEP so the web page knows the answer is complete
            yield (text + "\n\n" if text else "") + BUSY + SOURCES_SEP + "⏳"
            return
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
HEADER = """## زولك 🇸🇩 — مساعد الطلاب السودانيين في ألمانيا
بيجاوب من تجارب أعضاء قروب «طلاب السودان في ألمانيا» (بعد حذف أي معلومات شخصية).
⚠️ دي تجارب ناس ما استشارة قانونية — اتأكد دايماً من الجهات الرسمية."""

with gr.Blocks(title="Zolak", theme=gr.themes.Soft()) as demo:
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
    # CORS: Gradio echoes the caller's origin whenever the server's own host name is not localhost, so the GitHub
    # Pages site can call https://<domain>/ behind the reverse proxy without extra settings.
    demo.launch(server_name=os.environ.get("HOST", "0.0.0.0"), server_port=int(os.environ.get("PORT", 7860)))
