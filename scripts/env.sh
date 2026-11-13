# Source this before any heavy step:  source scripts/env.sh
# Puts all large files (HF models, checkpoints, venvs, pip/tmp caches) on /temp, not the (nearly full) root disk.
export SUDACHAT_STORAGE="${SUDACHAT_STORAGE:-/temp/$USER/sudaChat}"

export HF_HOME="$SUDACHAT_STORAGE/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export VLLM_CACHE_ROOT="$SUDACHAT_STORAGE/vllm_cache"
export TORCH_HOME="$SUDACHAT_STORAGE/torch_cache"
export PIP_CACHE_DIR="$SUDACHAT_STORAGE/pip_cache"
export TMPDIR="$SUDACHAT_STORAGE/tmp"
export CUDA_DEVICE_ORDER=PCI_BUS_ID   # CUDA ids == nvidia-smi ids (needed when picking GPUs by index)

export SUDACHAT_MODELS="$SUDACHAT_STORAGE/models"            # merged / GGUF models
export SUDACHAT_CHECKPOINTS="$SUDACHAT_STORAGE/checkpoints"  # LoRA training runs
export SUDACHAT_VENVS="$SUDACHAT_STORAGE/venvs"              # vllm + training venvs

mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$VLLM_CACHE_ROOT" "$TORCH_HOME" "$PIP_CACHE_DIR" "$TMPDIR" \
         "$SUDACHAT_MODELS" "$SUDACHAT_CHECKPOINTS" "$SUDACHAT_VENVS"
