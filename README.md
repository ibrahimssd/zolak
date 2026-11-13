# زولك — Zolak

A free chatbot that answers questions from Sudanese students and newcomers in Germany **in Sudanese Arabic**.
It covers visas, blocked accounts (Sperrkonto), residence permits, university admission, jobs, housing and more.

Answers are grounded in the shared experience of the community WhatsApp group «طلاب السودان في ألمانيا».
The group's messages were **anonymized** before use: names, phone numbers, emails, account numbers and private
links were removed.

> ⚠️ Answers reflect community experience, not legal advice. Always confirm with the official authorities
> (embassy, Ausländerbehörde, university).

## How it works

```
Browser ──► GitHub Pages (web/) ──► public Hugging Face Space (space/, CPU)
                                     ├─ hybrid retrieval: top-2 group entries (bge-m3 + BM25), summaries first
                                     └─► private ZeroGPU Space (space_gpu/): Zolak-ALLaM-7B → streamed Sudanese answer
```

- **Answers come only from the group.** The model gets the two most relevant group entries and answers from
  them alone, never from general knowledge. If they don't cover the question, it says the group hasn't
  discussed it. Rules and fees change, so every answer says when the information was shared.
- **Knowledge base:** 1,445 Q&A distilled from 2,779 anonymized discussions by a large teacher model
  (Qwen3-235B, run locally), normalized to Sudanese dialect and deduplicated.
- **Group summaries:** similar questions are clustered, and the teacher writes one summary per cluster
  (256 summaries covering 664 discussions): what most people said, where they disagreed, and when. Retrieval
  returns the summary in place of a single post from that cluster.
- **Model:** ALLaM-7B-Instruct fine-tuned with LoRA to answer in Sudanese dialect, cite the date of the
  information and admit when the group has no answer (held-out test: date cited 6% → 100%,
  honest "don't know" 5% → 100%).
- **Hosting (flat cost):** a public CPU Space does retrieval and serves the API. A private ZeroGPU Space
  generates the answers on the owner's Hugging Face PRO quota. No pay-per-use hardware is used.

## Privacy

- This repository contains **code only**. Chat data never leaves the processing machine, except the anonymized
  knowledge base, which lives in a private Hugging Face dataset.
- `tests/test_pii_leak.py` gates every data artifact: phone numbers, emails, IBANs and member names.
- CI runs `scripts/pii_scan.py` on every push, so chat lines or personal data can't be committed.
- The feedback buttons store only the question, the answer and the rating. No IP address or user information is
  stored.

## Repository layout

| Path | What |
|---|---|
| `src/ingest/` | WhatsApp parsing → anonymization → filtering → question threads → (LLM distillation) |
| `src/rag/` | Knowledge-base and retrieval-index builders |
| `src/train/` | Model bake-off, dialect meter, SFT data and LoRA fine-tuning |
| `space/` | Public backend app (Gradio + retriever; generation via the GPU worker) |
| `space_gpu/` | Private ZeroGPU worker that runs the fine-tuned model |
| `web/` | Static chat page for GitHub Pages |
| `scripts/` | Pipeline runners, GGUF conversion, deployment, privacy scan |


## Running the pipeline (maintainers)

```bash
source scripts/env.sh                  # caches/models on large storage
scripts/preprocess.sh export.txt       # parse → anonymize → filter → threads (+ leak tests)
python3 -m src.ingest.distill           # teacher LLM: threads → dated Q&A (kb.jsonl)
python3 -m src.rag.build_summaries      # teacher LLM: one summary per question cluster (kb_group.jsonl)
python3 -m src.rag.build_index --kb data/processed/kb_group.jsonl
python3 scripts/deploy_hf.py --step data    # leak-tested upload of the index (private dataset)
python3 scripts/deploy_hf.py --step gpu     # private ZeroGPU worker
python3 scripts/deploy_hf.py --step space   # public gateway
```
