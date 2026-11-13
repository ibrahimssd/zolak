"""Generate realistic student-style paraphrases of eval questions (for retrieval recall + end-to-end eval).

For each eval item the teacher writes 3 variants: casual Sudanese dialect, very short/keyword style,
and Arabizi/English/German-mixed. Output: data/processed/eval_paraphrases.jsonl  {"id", "style", "query"}

Usage:
    source scripts/env.sh
    CUDA_VISIBLE_DEVICES=0,1,2,3 $SUDACHAT_VENVS/vllm/bin/python -m src.ingest.paraphrase_eval
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .distill import TEACHER

SCHEMA = {
    "type": "object",
    "properties": {
        "casual": {"type": "string"},
        "short": {"type": "string"},
        "mixed": {"type": "string"},
    },
    "required": ["casual", "short", "mixed"],
}

PROMPT = """Rewrite this question the way different Sudanese students in Germany would actually type it in a chat.
Do not copy the wording; keep the same meaning. Return JSON with:
- "casual": natural Sudanese dialect, conversational, may include a little context about themselves
- "short": very short, keyword-like, maybe without question mark or with a typo
- "mixed": Arabizi (Arabic in Latin letters, e.g. "3ayez", "kif") or mixed Arabic/English/German

Question: {q}"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="data/processed/eval.jsonl")
    ap.add_argument("--out", default="data/processed/eval_paraphrases.jsonl")
    ap.add_argument("--model", default=TEACHER)
    ap.add_argument("--tp", type=int, default=4)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

    items = [json.loads(l) for l in open(args.eval, encoding="utf-8")]
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, max_model_len=4096, gpu_memory_utilization=0.90)
    params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=300,
                            guided_decoding=GuidedDecodingParams(json=SCHEMA))
    res = llm.chat([[{"role": "user", "content": PROMPT.format(q=it["question"])}] for it in items], params)
    n = 0
    with Path(args.out).open("w", encoding="utf-8") as f:
        for it, r in zip(items, res):
            try:
                o = json.loads(r.outputs[0].text)
            except json.JSONDecodeError:
                continue
            for style in ("casual", "short", "mixed"):
                f.write(json.dumps({"id": it["id"], "style": style, "query": o[style]}, ensure_ascii=False) + "\n")
                n += 1
    print(f"wrote {n} paraphrases for {len(items)} eval items -> {args.out}")


if __name__ == "__main__":
    main()
