# زولك · Zolak

Zolak is a chatbot for Sudanese students and newcomers in Germany. You ask in Sudanese Arabic, and it answers in
Sudanese Arabic from what members of the WhatsApp group «طلاب السودان في ألمانيا» have shared about visas, the
blocked account (Sperrkonto), residence permits, university admission, jobs and housing.

**Try it:** https://ibrahimssd.github.io/zolak/

> Answers are community experience, not legal advice. Rules and fees change, so check with the embassy, the
> Ausländerbehörde or your university before you act on them.

## Why

New students ask the same questions over and over: how long the security clearance takes, whether the Sperrkonto
can be paid in instalments, where to renew a Sudanese passport. The answers are already in the group, but they are
buried in thousands of messages and many of them are out of date. Zolak collects them, answers in the dialect people
actually write in, tells you when the information was shared, and says so when the group never discussed your
question.

## How it works

Offline, on our own server:

1. The WhatsApp export is anonymized before anything else happens: names, phone numbers, emails, account numbers
   and private links are removed.
2. Messages are grouped into questions and their replies (2,779 answered questions).
3. A large open model (Qwen3-235B, running locally) turns each thread into a short question and answer. It then
   checks every claim against the original discussion and removes what the discussion doesn't support; 1,318 entries
   are kept.
4. Questions that keep coming back get one summary each (183 summaries): what most people said, where they
   disagreed, and when.
5. A dialect normalizer replaces Egyptian, Gulf and formal forms with Sudanese ones.

When you ask a question:

1. Search finds the two most relevant entries. It combines an embedding model (bge-m3, fine-tuned on how members
   actually phrase their questions) with keyword search, which matches German terms like Sperrkonto exactly.
2. Zolak-ALLaM-7B, an ALLaM-7B model fine-tuned to answer only from those two entries, writes the answer in
   Sudanese Arabic and says when the information is from. If the entries don't answer the question, it says the group
   hasn't discussed it.
3. Before the answer is shown, links that aren't in the entries are removed and the dialect normalizer runs again.

```
Browser ─► GitHub Pages (web/) ─► Hugging Face Space, CPU (space/): finds the two entries
                                      └─► private ZeroGPU Space (space_gpu/): Zolak-ALLaM-7B writes the answer
```

## How well it works

We tested it on 566 questions it had never seen: 129 held-out questions from the group, each also reworded three
ways (formal Arabic, casual Sudanese, and Arabic mixed with German or English), plus 50 questions the group never
answered. The teacher model graded every answer:

| | |
|---|---|
| Answers with no claim that the entries don't support | 91% |
| Correct against the reference answer (partly correct counts half) | 92% |
| Answers that name an office the entries don't mention | 1% |
| Answers with non-Sudanese dialect forms | 0.2% |

The weak spot: when the two entries are on the right topic but don't actually answer the question, Zolak sometimes
answers anyway instead of saying it doesn't know (35 of 88 such cases). The grader is a model, not a person, so
feedback from students is the check that matters most.

## Feedback

Every answer has 👍 and 👎. After a 👎 the page asks «أها الصح شنو؟» ("so what's right?"), and the student can write
the correct answer from their own experience. Ratings and corrections go to a private dataset and, once anonymized,
help train the next version.

## Privacy

- This repository holds code only. The chat export never leaves the processing server; only the anonymized
  knowledge base and the models are uploaded, to private Hugging Face repositories.
- `tests/test_pii_leak.py` checks every data file for phone numbers, emails, IBANs and member names, and the upload
  script refuses to publish anything that fails.
- CI runs `scripts/pii_scan.py` on every push, so chat lines or personal data can't end up in the repository.
- Feedback stores the question, the answer, the rating, the IDs of the two entries used and, if given, the
  correction. No IP address or user information is stored.

## Cost

About €10 a month for a Hugging Face PRO subscription. The page and the search run on free hosting (GitHub Pages
and a free CPU Space), and answers are generated on ZeroGPU, which PRO includes. Nothing is billed per use. The
trade-off is speed: an answer usually takes a few seconds, but when several people ask at once they are answered one
after another.

## Repository layout

| Folder | Contents |
|---|---|
| `src/ingest/` | Parsing the export, anonymization, filtering, question threads, distillation with the teacher model |
| `src/rag/` | Link repair, the claim-by-claim audit, summaries, dialect normalization, embedder fine-tuning, the search index |
| `src/train/` | Training data, LoRA fine-tuning, evaluation, the dialect meter |
| `space/` | The public Space: search, prompt, dialect and link filters, feedback |
| `space_gpu/` | The private GPU worker that runs the model |
| `web/` | The chat page served by GitHub Pages |
| `scripts/` | Environment setup, preprocessing, deployment, the privacy scan |
| `tests/` | Unit tests and the leak test |

## Running the pipeline

For maintainers with access to the export and GPUs. The teacher-model steps need four large GPUs; the top of each
script says which environment and GPUs it expects.

```bash
scripts/make_venvs.sh                      # once per machine
source scripts/env.sh                      # caches, models and data on large storage

scripts/preprocess.sh export.txt           # parse, anonymize, filter, threads, leak test
python -m src.ingest.distill               # teacher: threads → questions and answers
python -m src.ingest.postprocess           # quality gates, de-duplication, held-out test split
python -m src.rag.fix_links                # repair and check every link
python -m src.rag.audit_kb                 # teacher: check each entry against its discussion
python -m src.rag.build_summaries --kb data/processed/kb_audited.jsonl
python -m src.rag.sudanize_kb              # dialect normalizer over the knowledge base
python -m src.rag.build_index --kb data/processed/kb_group.jsonl
python -m src.ingest.paraphrase_kb         # teacher: rewordings and unanswerable questions
python -m src.rag.finetune_embedder train --out $ZOLAK_MODELS/zolak-embed-v1
python -m src.rag.build_index --kb data/processed/kb_group.jsonl \
    --embed-model $ZOLAK_MODELS/zolak-embed-v1 --embed-repo <user>/zolak-models
python -m src.train.make_sft_data          # teacher: answers written from the retrieved entries
python -m src.train.sft_lora --run <name>  # LoRA fine-tune of ALLaM-7B, merged into zolak-allam-<name>
python -m src.train.eval_rag prep          # then: gen --name <name> --model <dir>, judge --names <name>

python scripts/deploy_hf.py --step embed   # fine-tuned embedder (private)
python scripts/deploy_hf.py --step weights --model-subdir zolak-allam-<name>
python scripts/deploy_hf.py --step data    # search index, only after the leak test passes
python scripts/deploy_hf.py --step gpu --model-subdir zolak-allam-<name>
python scripts/deploy_hf.py --step space --model-subdir zolak-allam-<name>
```
