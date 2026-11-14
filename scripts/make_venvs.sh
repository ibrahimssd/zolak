#!/usr/bin/env bash
# (Re)create the project venvs under $ZOLAK_VENVS. Venvs are not relocatable, so rebuild them on every new node.
# Usage: scripts/make_venvs.sh [vllm|train|space|qa ...]     (default: all four)
#   vllm  : teacher (distill/audit/summaries/paraphrases) + faiss/bm25 for GPU index builds
#   train : LoRA SFT + embedder fine-tune; also the general-purpose python3 for preprocessing and tests
#           (run scripts with PATH="$ZOLAK_VENVS/train/bin:$PATH")
#   space : the cpu-basic gateway's dependencies, for local retrieval/app tests
#   qa    : Playwright + Chromium (report figures) and gradio_client (live checks)
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
PY="${PYTHON:-python3}"
ST_STACK=(sentence-transformers==3.4.1 scipy==1.15.3 scikit-learn==1.7.2 "faiss-cpu>=1.8,<1.12" rank-bm25==0.2.2)

mk() {  # fresh venv with current pip
  rm -rf "$ZOLAK_VENVS/$1"
  "$PY" -m venv "$ZOLAK_VENVS/$1"
  "$ZOLAK_VENVS/$1/bin/pip" install -q --upgrade pip wheel
}

build() {
  local V="$ZOLAK_VENVS/$1/bin"
  case "$1" in
    vllm)
      mk vllm
      "$V/pip" install -r requirements/distill.txt "${ST_STACK[@]}" pytest ;;
    train)
      mk train
      "$V/pip" install -r requirements/train.txt
      # --no-deps keeps the torch/transformers pins from requirements/train.txt untouched
      "$V/pip" install --no-deps "${ST_STACK[@]}"
      "$V/pip" install joblib threadpoolctl Pillow pytest requests ;;
    space)
      mk space
      "$V/pip" install -r space/requirements-remote.txt pytest ;;
    qa)
      mk qa
      "$V/pip" install playwright gradio_client requests
      PLAYWRIGHT_BROWSERS_PATH="$ZOLAK_STORAGE/ms-playwright" "$V/python" -m playwright install chromium ;;
    *) echo "unknown venv: $1" >&2; exit 2 ;;
  esac
  "$V/pip" check || echo "WARN: pip check reported conflicts in $1" >&2
  echo "== $1 ready: $("$V/python" --version)"
}

for v in "${@:-vllm train space qa}"; do for x in $v; do build "$x"; done; done
