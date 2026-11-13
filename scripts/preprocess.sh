#!/usr/bin/env bash
# Full preprocessing pipeline: parse -> anonymize (+NER) -> filter -> threads -> leak gate.
# Usage: scripts/preprocess.sh [path/to/whatsapp_export.txt]   (default: data/raw/chat.txt)
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh   # model caches on /temp

INPUT="${1:-data/raw/chat.txt}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p data/interim data/processed

echo "== 1/4 parse";      python3 -m src.ingest.parse_whatsapp --input "$INPUT"
echo "== 2/4 anonymize";  CUDA_VISIBLE_DEVICES="$GPU" python3 -m src.ingest.anonymize --ner 2>&1 | grep -vE "UNEXPECTED|Notes:|^- |^$" || true
echo "== 3/4 filter";     python3 -m src.ingest.filter
echo "== 4/4 threads";    python3 -m src.ingest.threads
echo "== leak gate";      python3 -m pytest -q tests
