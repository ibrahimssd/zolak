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
  /correction (question: str, answer: str, correction: str) -> "ok"   the student's answer to «أها الصح شنو؟» after 👎
Feedback rows (FEEDBACK_REPO, data/feedback-*.jsonl; schema in the dataset card written by deploy_hf.py):
  {ts, rating, question, answer, entries: [ids of the 2 retrieved KB entries], model, correction (null unless sent)}
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

from dialect_fix import build_vocab, sudanize_text
from prompts import build_messages, guard_links
from retriever import HybridRetriever, retrieval_query

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
# Always the 2 most relevant group entries (no similarity cut-off): the model answers ONLY from them and says so
# when they don't cover the question.
TOP_K, MAX_TOKENS, HISTORY_TURNS = 2, 220, 2
SOURCES_SEP = "\n\n---\n"
# greetings / thanks: no retrieval, so the reply doesn't quote unrelated group entries
SMALLTALK_RE = re.compile(
    r"^(?:(?:ال)?سلام(?: عليكم)?(?: ورحمة الله(?: وبركاته)?)?|مرحب[اًا]?|هلا|اهلين|أهلين|اهلا|أهلا|كيفك|كيف الحال|"
    r"صباح الخير|مساء الخير|شكرا(?: جزيلا)?|شكراً|مشكور|تسلم|يسلمو|الله يديك العافية|hi|hello|hey|thanks?|thank you|"
    r"ok|okay|تمام|طيب|اوكي)[\s!.،؟?🙏❤️👍😊]*$", re.IGNORECASE)

# ---------------------------------------------------------------- load
t0 = time.time()
index_dir = os.environ.get("LOCAL_INDEX") or str(
    Path(snapshot_download(DATA_REPO, repo_type="dataset", allow_patterns=["index/*"], token=TOKEN)) / "index")
retriever = HybridRetriever(index_dir)   # embedder stays on CPU: GPU quota is spent only on generation
# words of the knowledge base: the dialect normalizer strips Egyptian «ما ...ش» only around known stems
VOCAB = build_vocab([" ".join([r["question"], r["answer"], *r.get("facts", [])]) for r in retriever.records])
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
                      do_sample=False, repetition_penalty=1.1)
        Thread(target=hf_model.generate, kwargs=kwargs).start()
        yield from streamer
elif BACKEND == "remote":
    # Generation runs in the private ZeroGPU worker Space (space_gpu/). Calling it with the owner's token makes
    # GPU time count against the owner's PRO quota instead of each anonymous visitor's tiny quota.
    from gradio_client import Client

    import threading

    import httpx

    GPU_SPACE = os.environ.get("GPU_SPACE", "ibrahimssd11/zolak-gpu")
    GPU_URL = "https://" + GPU_SPACE.replace("/", "-").replace(".", "-").lower() + ".hf.space"
    # A call that hangs (e.g. the worker went to sleep and the cached client is stale) would hold one of the queue's
    # slots forever and make every later visitor wait. Give up, reconnect and let answer() retry once.
    FIRST_CHUNK_S, TOTAL_S = 120, 210
    _gpu = {"client": None}

    def _gpu_client():
        if _gpu["client"] is None:
            _gpu["client"] = Client(GPU_SPACE, hf_token=TOKEN, verbose=False)
        return _gpu["client"]

    def _keep_awake():
        """Ping the worker every 15 min so it does not fall asleep (a cold start reloads the 14 GB model). The
        request only reaches the web server; no GPU is attached, so it costs no ZeroGPU quota."""
        while True:
            try:
                httpx.get(f"{GPU_URL}/config", headers={"Authorization": f"Bearer {TOKEN}"}, timeout=120)
            except Exception as e:
                print(f"keep-awake ping failed: {type(e).__name__}: {e}", flush=True)
            time.sleep(15 * 60)

    threading.Thread(target=_keep_awake, daemon=True).start()

    def generate(msgs: list[dict]):
        try:
            job = _gpu_client().submit(json.dumps(msgs, ensure_ascii=False), api_name="/generate")
            prev, t0 = "", time.time()
            while True:
                outs = job.outputs()
                out = outs[-1] if outs else ""
                if out and len(out) > len(prev):
                    yield out[len(prev):]
                    prev = out
                if job.done():
                    break
                waited = time.time() - t0
                if (not prev and waited > FIRST_CHUNK_S) or waited > TOTAL_S:
                    job.cancel()
                    raise TimeoutError(f"GPU worker sent {'no answer' if not prev else 'no end'} in {waited:.0f}s")
                time.sleep(0.2)
            outs = job.outputs()
            out = outs[-1] if outs else ""
            if out and len(out) > len(prev):
                yield out[len(prev):]
                prev = out
            if not prev and job.exception() is not None:
                raise job.exception()
        except Exception:
            _gpu["client"] = None   # reconnect next time (a restarted worker has a new session)
            raise
else:
    from llama_cpp import Llama

    model_path = os.environ.get("LOCAL_MODEL") or hf_hub_download(MODEL_REPO, MODEL_FILE, token=TOKEN)
    llm = Llama(model_path=model_path, n_ctx=4096, n_threads=N_THREADS, n_batch=512, verbose=False)
    is_qwen3 = "qwen3" in Path(model_path).name.lower()

    def generate(msgs: list[dict]):
        if is_qwen3:
            msgs[-1]["content"] += " /no_think"
        for chunk in llm.create_chat_completion(messages=msgs, stream=True, temperature=0.0,
                                                max_tokens=MAX_TOKENS, repeat_penalty=1.1):
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


# the model's refusal opening ("للأسف ما لقيت نقاش في القروب ...")
NOT_FOUND_RE = re.compile(r"^\W*(?:للأسف|معليش|والله)?[،,!\s]*ما (?:لقيت|لقينا|في) (?:أي )?(?:نقاش|تجارب|معلومات?)")


def _sources_line(recs: list[dict], answer_text: str = "") -> str:
    if not recs:
        return ""
    if NOT_FOUND_RE.match(answer_text.strip()):   # the model said the group never discussed it
        return "ℹ️ ما لقيت نقاش في القروب عن السؤال دا."
    parts = []
    for r in recs:
        if r.get("source") in ("summary", "guide"):
            a, b, n = r["date_from"][:7], r["date"][:7], r["n_discussions"]
            kind = "ملخص" if r["source"] == "summary" else "دليل من"
            parts.append((f"{kind} نقاشين" if n == 2 else f"{kind} {n} نقاشات") + (f" ({b})" if a == b else f" ({a} → {b})"))
        else:
            parts.append(f"نقاش {r['date'][:7]}")
    return "📅 من القروب: " + "، ".join(parts) + " — المعلومات بتتغير، اتأكد من الجهات الرسمية."


def _clean(text: str, recs: list[dict], final: bool = False) -> str:
    """Sudanese dialect forms («ما فيش» -> «ما في», «هذا» -> «دا», ...) and only links from the retrieved entries."""
    return guard_links(sudanize_text(text, VOCAB), recs, final=final)


def answer(question: str, history: list | None = None):
    question = re.sub(r"\s+", " ", (question or "")).strip()[:1000]
    if not question:
        yield "اكتب سؤالك 🙂"
        return
    # a follow-up («بالنسبه للاهلي لازم؟») is searched together with the previous question
    recs = [] if SMALLTALK_RE.match(question) else \
        retriever.top_entries(retrieval_query(question, _clean_history(history)), k=TOP_K)
    print(f"retrieved: {[r['id'] for r in recs]}", flush=True)
    msgs = build_messages(question, recs, _clean_history(history))
    text = ""
    for attempt in (1, 2):
        try:
            for piece in generate(msgs):
                text += piece
                yield _clean(text, recs)
            break
        except Exception as e:  # ZeroGPU: quota used up or no GPU free right now (never billed)
            print(f"generation error (attempt {attempt}): {type(e).__name__}: {e}", flush=True)
            transient = "quota" not in str(e).lower() and not text
            if attempt == 1 and transient:
                time.sleep(2)
                continue
            # every final message ends with SOURCES_SEP so the web page knows the answer is complete
            yield (_clean(text, recs, final=True) + "\n\n" if text else "") + BUSY + SOURCES_SEP + "⏳"
            return
    text = _clean(text.strip(), recs, final=True)
    yield text + SOURCES_SEP + _sources_line(recs, text)


MODEL_NAME = MODEL_SUBDIR if BACKEND in ("zerogpu", "remote") else MODEL_FILE


def _log_feedback(row: dict) -> None:
    """Append one row to this replica's feedback file; the CommitScheduler pushes it to the private dataset."""
    if scheduler:
        with scheduler.lock:
            with FEEDBACK_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _feedback_row(question: str, answer_text: str, rating: str) -> dict:
    question = re.sub(r"\s+", " ", (question or "")).strip()[:1000]
    try:  # retrieval is deterministic: these are the 2 entries the answer was written from (for later training)
        entries = [] if not question or SMALLTALK_RE.match(question) else \
            [r["id"] for r in retriever.top_entries(question, k=TOP_K)]
    except Exception as e:
        print(f"feedback retrieval failed: {type(e).__name__}: {e}", flush=True)
        entries = []
    # always the same columns in the same order: the dataset viewer needs one schema across all files
    return {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "rating": rating, "question": question,
            "answer": (answer_text or "")[:3000], "entries": entries, "model": MODEL_NAME, "correction": None}


def feedback(question: str, answer_text: str, rating: str) -> str:
    if rating not in ("up", "down"):
        return "ignored"
    _log_feedback(_feedback_row(question, answer_text, rating))
    return "ok"


def correction(question: str, answer_text: str, correction_text: str) -> str:
    """After 👎 the page asks «أها الصح شنو؟»; the student's correction is kept with the question, the wrong
    answer and the retrieved entries. It is raw user text: anonymize it before any training use."""
    correction_text = (correction_text or "").strip()[:2000]
    if not correction_text:
        return "ignored"
    row = _feedback_row(question, answer_text, "down")
    row["correction"] = correction_text
    _log_feedback(row)
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
        cq, ca, cc, c_out = gr.Textbox(), gr.Textbox(), gr.Textbox(), gr.Textbox()
    q_in.submit(answer, [q_in, h_in], a_out, api_name="ask")
    fr.submit(feedback, [fq, fa, fr], f_out, api_name="feedback")
    cc.submit(correction, [cq, ca, cc], c_out, api_name="correction")

demo.queue(default_concurrency_limit=4 if BACKEND == "remote" else 1, max_size=40)

if __name__ == "__main__":
    # CORS: Gradio echoes the caller's origin whenever the server's own host name is not localhost, so the GitHub
    # Pages site can call https://<domain>/ behind the reverse proxy without extra settings.
    demo.launch(server_name=os.environ.get("HOST", "0.0.0.0"), server_port=int(os.environ.get("PORT", 7860)))
