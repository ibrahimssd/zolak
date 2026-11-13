# Source this before any heavy step:  source scripts/env.sh
# Puts all large files (HF models, checkpoints, venvs, pip/tmp caches) on /temp, not the (nearly full) root disk.
export ZOLAK_STORAGE="${ZOLAK_STORAGE:-/temp/$USER/zolak}"

export HF_HOME="$ZOLAK_STORAGE/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export VLLM_CACHE_ROOT="$ZOLAK_STORAGE/vllm_cache"
export TORCH_HOME="$ZOLAK_STORAGE/torch_cache"
export PIP_CACHE_DIR="$ZOLAK_STORAGE/pip_cache"
export TMPDIR="$ZOLAK_STORAGE/tmp"
export CUDA_DEVICE_ORDER=PCI_BUS_ID   # CUDA ids == nvidia-smi ids (needed when picking GPUs by index)

export ZOLAK_MODELS="$ZOLAK_STORAGE/models"            # merged / GGUF models
export ZOLAK_CHECKPOINTS="$ZOLAK_STORAGE/checkpoints"  # LoRA training runs
export ZOLAK_VENVS="$ZOLAK_STORAGE/venvs"              # vllm + training venvs

mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$VLLM_CACHE_ROOT" "$TORCH_HOME" "$PIP_CACHE_DIR" "$TMPDIR" \
         "$ZOLAK_MODELS" "$ZOLAK_CHECKPOINTS" "$ZOLAK_VENVS"
