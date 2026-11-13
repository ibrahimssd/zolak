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
Browser ──► GitHub Pages (web/)  ──►  Oracle Cloud Always Free VM (deploy/oracle: Docker + Caddy HTTPS)
                                        ├─ hybrid retrieval: bge-m3 + BM25 over an anonymized, dated knowledge base
                                        └─ llama.cpp: Zolak-ALLaM-7B (LoRA fine-tune, 4-bit) → streamed answer in Sudanese dialect
```

- **Facts come from retrieval, style from the model.** Rules and fees change, so every answer says when the
  information was shared in the group.
- **Knowledge base:** 1,445 Q&A distilled from 2,779 anonymized discussions by a large teacher model
  (Qwen3-235B, run locally), normalized to Sudanese dialect and deduplicated.
- **Model:** ALLaM-7B-Instruct fine-tuned with LoRA to answer in Sudanese dialect, cite the date of the
  information and admit when the group has no answer (held-out test: date cited 6% → 100%,
  honest "don't know" 5% → 100%).
- **Hosting:** `deploy/oracle/` runs the backend on an Oracle Cloud Always Free ARM VM (2 cores, 12 GB)
  behind Caddy with automatic HTTPS; see `deploy/oracle/setup.sh`.

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
| `src/train/` | Model bake-off, dialect meter, fine-tuning (upcoming) |
| `space/` | Chatbot backend app (Gradio + llama.cpp + retriever) |
| `deploy/oracle/` | Docker image, Caddy HTTPS proxy and VM setup script for the backend |
| `web/` | Static chat page for GitHub Pages |
| `scripts/` | Pipeline runners, GGUF conversion, deployment, privacy scan |


## Running the pipeline (maintainers)

```bash
source scripts/env.sh                  # caches/models on large storage
scripts/preprocess.sh export.txt       # parse → anonymize → filter → threads (+ leak tests)
python3 -m src.rag.build_raw_kb        # starter KB from threads
python3 -m src.rag.build_index --kb data/processed/kb_raw.jsonl
python3 scripts/deploy_hf.py --step all --wheel-url <release asset URL>
```
