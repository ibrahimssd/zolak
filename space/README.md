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

# زولك · Zolak

This Space is the backend of Zolak, a chatbot that answers questions from Sudanese students and newcomers in
Germany in Sudanese Arabic, from the anonymized experience of the group «طلاب السودان في ألمانيا».

Use the chat page: https://ibrahimssd.github.io/zolak/

- The Space searches a private, anonymized knowledge base (fine-tuned bge-m3 plus keyword search) and sends the two
  best entries to a private GPU worker running Zolak-ALLaM-7B, which answers only from them.
- Answers are community experience, not legal advice. Always confirm with the official authorities.
- Code: https://github.com/ibrahimssd/zolak
