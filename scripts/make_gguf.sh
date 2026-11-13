#!/usr/bin/env bash
# Convert a HF model (repo id from the local cache, or a local directory) to GGUF and quantize it.
# Usage: scripts/make_gguf.sh <repo_id|local_dir> [quant=Q4_K_M] [out_name]
# Output: $ZOLNA_MODELS/gguf/<out_name>-<quant>.gguf   (the intermediate bf16 GGUF is deleted)
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

SRC="$1"; QUANT="${2:-Q4_K_M}"; NAME="${3:-$(basename "$SRC")}"
LLAMA="$ZOLNA_STORAGE/llama.cpp"; PY="$ZOLNA_VENVS/llamacpp/bin/python"
OUT_DIR="$ZOLNA_MODELS/gguf"; mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/$NAME-$QUANT.gguf"; TMP="$OUT_DIR/$NAME-bf16.tmp.gguf"

if [ -d "$SRC" ]; then
  MODEL_DIR="$SRC"
else
  MODEL_DIR=$("$PY" -c "from huggingface_hub import snapshot_download as s; print(s('$SRC', local_files_only=True))")
fi

echo "== convert $SRC -> bf16"
"$PY" "$LLAMA/convert_hf_to_gguf.py" "$MODEL_DIR" --outtype bf16 --outfile "$TMP" > "$ZOLNA_STORAGE/logs/convert_$NAME.log" 2>&1 \
  || { tail -5 "$ZOLNA_STORAGE/logs/convert_$NAME.log"; exit 1; }
echo "== quantize -> $QUANT"
"$LLAMA/build/bin/llama-quantize" "$TMP" "$OUT" "$QUANT" 32 > "$ZOLNA_STORAGE/logs/quantize_$NAME.log" 2>&1 \
  || { tail -5 "$ZOLNA_STORAGE/logs/quantize_$NAME.log"; exit 1; }
rm -f "$TMP"
ls -lh "$OUT"
