"""Step 6 — base-model bake-off for the student (7-9B, + 3-4B speed fallback).

modes:
  probe : 12 hand-written Sudanese student questions (10 without context, 1 grounded, 1 "unknown") — runs before
          the KB exists, to see which bases can speak Sudanese at all.
  eval  : data/processed/eval.jsonl with the gold KB record as context (grounded generation), question taken from
          eval_paraphrases.jsonl ("casual" style) when available; + 20 items with empty context (should admit
          it doesn't know).

One model per process (vLLM does not free GPU memory between engines). Outputs:
  data/interim/bakeoff/<mode>/<model_slug>.jsonl  ({"key", "question", "context_ids", "answer", "secs"})
Summary (dialect meter + side-by-side review file):
  python -m src.train.bakeoff --summarize --mode probe

Usage:
  source scripts/env.sh
  for m in ...; do CUDA_VISIBLE_DEVICES=4 $SUDACHAT_VENVS/vllm/bin/python -m src.train.bakeoff --mode probe --model $m; done
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "space"))
from prompts import build_messages  # noqa: E402

CANDIDATES = [
    "humain-ai/ALLaM-7B-Instruct-preview",
    "Qwen/Qwen3-8B",
    "QCRI/Fanar-1-9B-Instruct",
    "silma-ai/SILMA-9B-Instruct-v1.0",
    "Qwen/Qwen3-4B",
]

_CTX_SECURITY = {
    "id": "probe-ctx-1", "date": "2025-06-01", "question": "الموافقة الأمنية للفيزا بتاخد قدر شنو؟",
    "answer": "في ناس قالت الموافقة الأمنية بتاخد من شهرين لي أربعة شهور، وفي ناس طلعت ليهم في أقل من شهر. "
              "المهم تكون الأوراق كاملة وتتابع مع السفارة بالإيميل.",
    "facts": ["مدة الموافقة الأمنية (Sicherheitsbefragung) بين شهر وأربعة شهور حسب تجارب الأعضاء في 2025"],
    "links": [],
}
PROBES = [
    ("p01", "السلام عليكم، داير اعرف الحساب المغلق بفتحو كيف؟", []),
    ("p02", "انا قبلوني ماستر في ألمانيا، شنو الحاجات البعملها اول ما اصل؟", []),
    ("p03", "في زول بيعرف كيف احجز موعد في السفارة الألمانية في القاهرة؟", []),
    ("p04", "Anmeldung دي شنو وبعملها وين؟", []),
    ("p05", "ممكن اشتغل وانا طالب؟ كم ساعة مسموح؟", []),
    ("p06", "جوازي قرب يخلص وانا هنا في ألمانيا، اعمل شنو؟", []),
    ("p07", "التأمين الصحي للطلاب بكم وياتو احسن؟", []),
    ("p08", "كيف ارسل قروش لأهلي في السودان من هنا؟", []),
    ("p09", "شنو الفرق بين Ausbildung والجامعة؟", []),
    ("p10", "عندي B1 ألماني، بقدر اقدم على Ausbildung؟", []),
    ("p11", "الموافقة الامنية بتطول قدر شنو؟", [_CTX_SECURITY]),
    ("p12", "سعر تذكرة القطر من برلين لميونخ كم؟", []),
]


def slug(model: str) -> str:
    return model.replace("/", "__")


def load_eval_items(proc: Path, n_unknown: int = 20) -> list[tuple[str, str, list[dict]]]:
    kb = {json.loads(l)["id"]: json.loads(l) for l in (proc / "kb.jsonl").open(encoding="utf-8")}
    evals = [json.loads(l) for l in (proc / "eval.jsonl").open(encoding="utf-8")]
    para = {}
    pp = proc / "eval_paraphrases.jsonl"
    if pp.exists():
        for l in pp.open(encoding="utf-8"):
            p = json.loads(l)
            if p["style"] == "casual":
                para[p["id"]] = p["query"]
    items = [(f"g-{e['id']}", para.get(e["id"], e["question"]), [kb[e["id"]]]) for e in evals]
    rnd = random.Random(0)
    items += [(f"u-{e['id']}", para.get(e["id"], e["question"]), []) for e in rnd.sample(evals, min(n_unknown, len(evals)))]
    return items


def supports_system(tokenizer) -> bool:
    try:
        tokenizer.apply_chat_template([{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
                                      tokenize=False, add_generation_prompt=True)
        return True
    except Exception:
        return False


def to_model_messages(msgs: list[dict], system_ok: bool) -> list[dict]:
    if system_ok or msgs[0]["role"] != "system":
        return msgs
    sys_msg, first, rest = msgs[0], msgs[1], msgs[2:]
    return [{"role": "user", "content": sys_msg["content"] + "\n\n" + first["content"]}, *rest]


def run(model: str, mode: str, out_dir: Path, max_tokens: int) -> None:
    from vllm import LLM, SamplingParams

    items = PROBES if mode == "probe" else load_eval_items(Path("data/processed"))
    llm = LLM(model=model, tensor_parallel_size=1, max_model_len=4096, gpu_memory_utilization=0.85)
    tok = llm.get_tokenizer()
    system_ok = supports_system(tok)
    convs = [to_model_messages(build_messages(q, ctx), system_ok) for _, q, ctx in items]
    # Qwen3: disable "thinking" so we compare plain answers
    kw = {"chat_template_kwargs": {"enable_thinking": False}} if "qwen3" in model.lower() else {}
    params = SamplingParams(temperature=0.3, top_p=0.9, max_tokens=max_tokens, repetition_penalty=1.05)
    t0 = time.time()
    outs = llm.chat(convs, params, use_tqdm=False, **kw)
    secs = (time.time() - t0) / max(len(items), 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{slug(model)}.jsonl").open("w", encoding="utf-8") as f:
        for (key, q, ctx), o in zip(items, outs):
            f.write(json.dumps({"key": key, "question": q, "context_ids": [c["id"] for c in ctx],
                                "answer": o.outputs[0].text.strip(), "secs": round(secs, 2),
                                "system_ok": system_ok}, ensure_ascii=False) + "\n")
    print(f"{model}: {len(items)} answers -> {out_dir}  (system role supported: {system_ok})")


def summarize(out_dir: Path) -> None:
    from src.train.dialect import score, summarize as dsum

    files = sorted(out_dir.glob("*.jsonl"))
    rows = {f.stem: [json.loads(l) for l in f.open(encoding="utf-8")] for f in files}
    if not rows:
        print(f"no results in {out_dir}")
        return
    print(f"{'model':45s} {'sd/ans':>7s} {'other/ans':>9s} {'w/other':>7s} {'foreign':>7s} {'avg chars':>9s}")
    for name, rs in rows.items():
        s = dsum([r["answer"] for r in rs])
        avg = sum(len(r["answer"]) for r in rs) / max(len(rs), 1)
        print(f"{name:45s} {s['sd_per_text']:7.2f} {s['other_per_text']:9.2f} {s['texts_with_other']:7d} "
              f"{s['texts_with_foreign']:7d} {avg:9.0f}")
    review = out_dir / "side_by_side.md"
    keys = [r["key"] for r in next(iter(rows.values()))]
    with review.open("w", encoding="utf-8") as f:
        for i, k in enumerate(keys):
            q = next(iter(rows.values()))[i]["question"]
            f.write(f"\n## {k}: {q}\n")
            for name, rs in rows.items():
                sc = score(rs[i]["answer"])
                f.write(f"\n**{name}** (sd={sc['sd']} other={sc['other']}):\n\n{rs[i]['answer']}\n")
    print(f"side-by-side review -> {review}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="")
    ap.add_argument("--mode", choices=["probe", "eval"], default="probe")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    out_dir = Path("data/interim/bakeoff") / args.mode
    if args.summarize:
        summarize(out_dir)
    else:
        run(args.model, args.mode, out_dir, args.max_tokens)


if __name__ == "__main__":
    main()
