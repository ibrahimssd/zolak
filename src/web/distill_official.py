"""Turn crawled official pages into Sudanese-dialect Q&A records (teacher LLM, local GPUs).

Input : data/official/chunks.jsonl (src/web/crawl_official.py)
Output: data/official/kb_official.jsonl  records like kb.jsonl, plus source="official", site, url
        data/official/distill_official_raw.jsonl (raw teacher output)

Usage:
    source scripts/env.sh
    CUDA_VISIBLE_DEVICES=0,1,2,3 $ZOLAK_VENVS/vllm/bin/python -m src.web.distill_official
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.ingest.distill import TEACHER, TOPICS

SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "items": {
            "type": "array", "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                    "facts": {"type": "array", "items": {"type": "string"}},
                    "german_terms": {"type": "array", "items": {"type": "string"}},
                    "topic": {"type": "string", "enum": TOPICS},
                    "time_sensitive": {"type": "boolean"},
                },
                "required": ["question", "answer", "facts", "german_terms", "topic", "time_sensitive"],
            },
        },
    },
    "required": ["reason", "items"],
}

SYSTEM = """You turn text from an OFFICIAL German website into practical Q&A for Sudanese students and newcomers in
Germany (visa, Sperrkonto/blocked account, residence permit, Anmeldung, university admission, recognition of
certificates, work rules, health insurance, language, Studienkolleg, family reunification, costs).

Rules:
1. Use ONLY information in the text. Never add outside knowledge. Keep amounts, durations, office names exactly.
2. Write up to 3 items, each a question a Sudanese student would really ask, and an answer of 2-5 sentences.
   No item if the text is not useful for them (marketing, navigation, unrelated topics) -> "items": [].
3. Language of question and answer: natural SUDANESE Arabic dialect (عشان، داير، زول، شنو، كدا، برضو، بتقدر،
   ما بتحتاج، حتلقى). No Egyptian/Gulf/Levantine forms (مش، مو، هذي، دلوقتي، إزاي، بدك، -ش negation), no formal MSA.
   Keep German/English terms in Latin script (Sperrkonto, Anmeldung, Aufenthaltstitel, uni-assist, APS ...).
4. facts: 1-4 short atomic facts in Arabic with the German term. german_terms: relevant search keywords.
5. time_sensitive: true for amounts, fees, deadlines, processing times or rules that change.
Write a short English "reason" first."""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="data/official/chunks.jsonl")
    ap.add_argument("--out", default="data/official/distill_official_raw.jsonl")
    ap.add_argument("--tp", type=int, default=4)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

    chunks = [json.loads(l) for l in open(args.chunks, encoding="utf-8")]
    llm = LLM(model=TEACHER, tensor_parallel_size=args.tp, max_model_len=8192, gpu_memory_utilization=0.90,
              enable_prefix_caching=True)
    params = SamplingParams(temperature=0.2, top_p=0.9, max_tokens=1800,
                            guided_decoding=GuidedDecodingParams(json=SCHEMA))
    convs = [[{"role": "system", "content": SYSTEM},
              {"role": "user", "content": f"Website: {c['site']} ({c['url']})\nPage: {c['title']}\n\n{c['text']}"}]
             for c in chunks]
    t0 = time.time()
    res = llm.chat(convs, params, use_tqdm=True)
    n_items = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for c, r in zip(chunks, res):
            try:
                out = json.loads(r.outputs[0].text)
            except json.JSONDecodeError:
                out = {"reason": "json_error", "items": []}
            n_items += len(out.get("items", []))
            f.write(json.dumps({**{k: c[k] for k in ("chunk_id", "url", "title", "site", "fetched", "text")},
                                "output": out}, ensure_ascii=False) + "\n")
    print(f"{len(chunks)} chunks -> {n_items} raw items in {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
