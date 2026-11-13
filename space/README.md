---
title: Zolak
emoji: ☕
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 5.49.1
python_version: "3.10"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Sudanese-dialect helper for Sudanese students in Germany
---

# زولك — Zolak

Answers questions from Sudanese students and newcomers in Germany, in Sudanese dialect, grounded in the
anonymized experience of the community group "طلاب السودان في ألمانيا".

- Model: ALLaM-7B-Instruct (Q4_K_M, llama.cpp) — will be replaced by a Sudanese-dialect fine-tune.
- Retrieval: BAAI/bge-m3 + BM25 over a private, anonymized knowledge base.
- These are community experiences, not legal advice. Always confirm with the official authorities.
