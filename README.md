# سوداشات — SudaChat

A free chatbot that answers questions from Sudanese students and newcomers in Germany **in Sudanese Arabic**.
It covers visas, blocked accounts (Sperrkonto), residence permits, university admission, jobs, housing and more.

Answers are grounded in the shared experience of the community WhatsApp group «طلاب السودان في ألمانيا».
The group's messages were **anonymized** before use: names, phone numbers, emails, account numbers and private
links were removed.

> ⚠️ Answers reflect community experience, not legal advice. Always confirm with the official authorities
> (embassy, Ausländerbehörde, university).

## How it works

```
Browser ──► GitHub Pages (web/)  ──►  Hugging Face Space (space/, free CPU)
                                        ├─ hybrid retrieval: bge-m3 + BM25 over an anonymized, dated knowledge base
                                        └─ llama.cpp: ALLaM-7B-Instruct (4-bit) → streamed answer in Sudanese dialect
```

- **Facts come from retrieval, style from the model.** Rules and fees change, so every answer says when the
  information was shared in the group.
- **Current version:** base ALLaM-7B with a knowledge base built directly from anonymized chat threads.
  **Next:** a cleaned Q&A knowledge base (made by a larger model) and a Sudanese-dialect fine-tune.

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
| `space/` | Hugging Face Space app (Gradio + llama.cpp + retriever) |
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
