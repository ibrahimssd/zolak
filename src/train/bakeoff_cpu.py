"""Step 6 (CPU variant) — bake-off of quantized GGUF students with llama.cpp, i.e. exactly how the free HF Space
will run them. Quality: same 12 probes / prompt as bakeoff.py. Speed: llama-bench pinned to 2 cores.

Usage:
    source scripts/env.sh
    python3 -m src.train.bakeoff_cpu                       # all *-Q4_K_M.gguf in $ZOLNA_MODELS/gguf
    python3 -m src.train.bakeoff_cpu --only Qwen3-4B       # one model
    python3 -m src.train.bakeoff_cpu --summarize
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from src.train.bakeoff import PROBES, build_messages, summarize, to_model_messages

STORAGE = Path(os.environ.get("ZOLNA_STORAGE", f"/temp/{os.environ.get('USER', 'user')}/zolna"))
LLAMA_BIN = STORAGE / "llama.cpp" / "build" / "bin"
GGUF_DIR = Path(os.environ.get("ZOLNA_MODELS", STORAGE / "models")) / "gguf"
OUT_DIR = Path("data/interim/bakeoff/probe_cpu")
SPEED_CORES = "100,101"     # 2 cores, like the free Space
QUALITY_THREADS = 32        # be nice on the shared server

# Typical request on the Space: system prompt + 3 context records + question ≈ 700 tokens in, ≈ 150 tokens out
TYPICAL_PROMPT_TOKENS, TYPICAL_GEN_TOKENS = 700, 150


def _post(url: str, payload: dict, timeout: int = 600) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _wait_health(port: int, proc: subprocess.Popen, timeout: int = 600) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError("llama-server exited during startup")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError):
            pass
        time.sleep(2)
    raise TimeoutError("llama-server did not become healthy")


def chat(port: int, msgs: list[dict], is_qwen3: bool) -> dict:
    payload = {"messages": msgs, "temperature": 0.3, "top_p": 0.9, "max_tokens": 400, "repeat_penalty": 1.05}
    if is_qwen3:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    return _post(f"http://127.0.0.1:{port}/v1/chat/completions", payload)


def run_quality(gguf: Path, port: int = 8091) -> None:
    name = gguf.name.removesuffix(".gguf")
    is_qwen3 = "qwen3" in name.lower()
    log = open(STORAGE / "logs" / f"llama_server_{name}.log", "w")
    proc = subprocess.Popen([str(LLAMA_BIN / "llama-server"), "-m", str(gguf), "-t", str(QUALITY_THREADS),
                             "-c", "4096", "-np", "1", "--jinja", "--host", "127.0.0.1", "--port", str(port)],
                            stdout=log, stderr=subprocess.STDOUT)
    try:
        _wait_health(port, proc)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        system_ok = True
        with (OUT_DIR / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for key, q, ctx in PROBES:
                msgs = build_messages(q, ctx)
                try:
                    r = chat(port, to_model_messages(msgs, system_ok), is_qwen3)
                except urllib.error.HTTPError:
                    if not system_ok:
                        raise
                    system_ok = False  # template rejects the system role (e.g. Gemma-2) -> merge into user turn
                    r = chat(port, to_model_messages(msgs, system_ok), is_qwen3)
                t = r.get("timings", {})
                f.write(json.dumps({
                    "key": key, "question": q, "context_ids": [c["id"] for c in ctx],
                    "answer": r["choices"][0]["message"]["content"].strip(), "system_ok": system_ok,
                    "prompt_tokens": t.get("prompt_n"), "gen_tokens": t.get("predicted_n"),
                }, ensure_ascii=False) + "\n")
        print(f"{name}: {len(PROBES)} answers (system role: {system_ok})")
    finally:
        proc.terminate()
        proc.wait(timeout=60)
        log.close()


def run_speed(gguf: Path) -> dict:
    cmd = ["taskset", "-c", SPEED_CORES, str(LLAMA_BIN / "llama-bench"), "-m", str(gguf), "-t", "2",
           "-p", str(TYPICAL_PROMPT_TOKENS), "-n", "64", "-r", "2", "-o", "json"]
    res = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
    pp = next(r["avg_ts"] for r in res if r["n_prompt"] > 0)
    tg = next(r["avg_ts"] for r in res if r["n_gen"] > 0)
    est = TYPICAL_PROMPT_TOKENS / pp + TYPICAL_GEN_TOKENS / tg
    out = {"model": gguf.name, "size_gb": round(gguf.stat().st_size / 1e9, 2), "prompt_tok_s": round(pp, 1),
           "gen_tok_s": round(tg, 2), "first_token_s": round(TYPICAL_PROMPT_TOKENS / pp, 1),
           "full_answer_s": round(est, 1)}
    print(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="")
    ap.add_argument("--skip-quality", action="store_true")
    ap.add_argument("--skip-speed", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()

    speed_path = OUT_DIR / "speed.json"
    if args.summarize:
        summarize(OUT_DIR)
        if speed_path.exists():
            print("\nspeed on 2 cores (typical request: 700 tokens in, 150 out):")
            for s in json.loads(speed_path.read_text()):
                print(f"  {s['model']:40s} {s['size_gb']:5.2f} GB  prompt {s['prompt_tok_s']:6.1f} tok/s  "
                      f"gen {s['gen_tok_s']:5.2f} tok/s  first token {s['first_token_s']:5.1f}s  "
                      f"full answer {s['full_answer_s']:5.1f}s")
        return

    ggufs = sorted(p for p in GGUF_DIR.glob("*-Q4_K_M.gguf") if args.only in p.name)
    speeds = json.loads(speed_path.read_text()) if speed_path.exists() else []
    for g in ggufs:
        if not args.skip_quality:
            run_quality(g)
        if not args.skip_speed:
            speeds = [s for s in speeds if s["model"] != g.name] + [run_speed(g)]
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            speed_path.write_text(json.dumps(speeds, indent=1))


if __name__ == "__main__":
    main()
