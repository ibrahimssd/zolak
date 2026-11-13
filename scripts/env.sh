# Source this before any heavy step:  source scripts/env.sh
# Puts all large files (HF models, checkpoints, venvs, pip/tmp caches) on /temp, not the (nearly full) root disk.
export ZOLNA_STORAGE="${ZOLNA_STORAGE:-/temp/$USER/zolna}"

export HF_HOME="$ZOLNA_STORAGE/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export VLLM_CACHE_ROOT="$ZOLNA_STORAGE/vllm_cache"
export TORCH_HOME="$ZOLNA_STORAGE/torch_cache"
export PIP_CACHE_DIR="$ZOLNA_STORAGE/pip_cache"
export TMPDIR="$ZOLNA_STORAGE/tmp"
export CUDA_DEVICE_ORDER=PCI_BUS_ID   # CUDA ids == nvidia-smi ids (needed when picking GPUs by index)

export ZOLNA_MODELS="$ZOLNA_STORAGE/models"            # merged / GGUF models
export ZOLNA_CHECKPOINTS="$ZOLNA_STORAGE/checkpoints"  # LoRA training runs
export ZOLNA_VENVS="$ZOLNA_STORAGE/venvs"              # vllm + training venvs

mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$VLLM_CACHE_ROOT" "$TORCH_HOME" "$PIP_CACHE_DIR" "$TMPDIR" \
         "$ZOLNA_MODELS" "$ZOLNA_CHECKPOINTS" "$ZOLNA_VENVS"
