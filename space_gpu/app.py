"""Zolak GPU worker — private ZeroGPU Space that only generates text.

The public Space (space/app.py, BACKEND=remote) does retrieval and prompt building, then calls this Space's
/generate endpoint with the owner's token, so GPU time counts against the owner's PRO ZeroGPU quota instead of
each anonymous visitor's tiny quota. Nothing is billed: when the quota is used up, calls fail and the public
Space shows a "busy, try later" message.

Environment: HF_TOKEN (secret, read the private model repo), MODEL_REPO, MODEL_SUBDIR.
API: /generate (messages_json: str) -> streamed text (cumulative)
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Thread

import gradio as gr
import spaces
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

TOKEN = os.environ.get("HF_TOKEN")
MODEL_REPO = os.environ.get("MODEL_REPO", "ibrahimssd11/zolak-models")
MODEL_SUBDIR = os.environ.get("MODEL_SUBDIR", "zolak-allam-v2")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", 300))

model_dir = Path(snapshot_download(MODEL_REPO, allow_patterns=[f"{MODEL_SUBDIR}/*"], token=TOKEN)) / MODEL_SUBDIR
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16)
model.to("cuda")  # ZeroGPU attaches a GPU per call
print(f"loaded {MODEL_SUBDIR}")


@spaces.GPU(duration=30)  # an answer takes ~5-10 s; a small reservation stretches the daily quota
def generate(messages_json: str):
    msgs = json.loads(messages_json)
    ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to("cuda")
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    kwargs = dict(input_ids=ids, attention_mask=torch.ones_like(ids), streamer=streamer,
                  max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=0.3, top_p=0.9, repetition_penalty=1.05)
    Thread(target=model.generate, kwargs=kwargs).start()
    text = ""
    for piece in streamer:
        text += piece
        yield text


with gr.Blocks(title="Zolak GPU worker") as demo:
    gr.Markdown("Private GPU worker for Zolak. Use the public app instead.")
    inp, out = gr.Textbox(visible=False), gr.Textbox(visible=False)
    inp.submit(generate, inp, out, api_name="generate")

demo.queue(default_concurrency_limit=4, max_size=64)

if __name__ == "__main__":
    demo.launch()
