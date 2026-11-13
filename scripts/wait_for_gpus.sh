#!/usr/bin/env bash
# Wait until N GPUs are free AND healthy, then print N of their ids (comma-separated, nvidia-smi/PCI order).
#   free    = no compute processes and < 1 GB used
#   healthy = a CUDA context can be created (catches GPUs stuck in an ECC error state, which nvidia-smi
#             still lists as idle). Unhealthy GPUs are remembered for the rest of the wait.
# Usage: GPUS=$(scripts/wait_for_gpus.sh 4 [poll_seconds])     (EXCLUDE_GPUS="4,5" to skip known-bad GPUs)
# Callers must also export CUDA_DEVICE_ORDER=PCI_BUS_ID so CUDA ids match nvidia-smi ids.
# Note: no `head` in a pipeline — under pipefail an early-closed pipe (SIGPIPE) would abort the script.
set -euo pipefail
N="${1:-4}"; POLL="${2:-60}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
declare -A bad=()
EXCLUDE_GPUS="${EXCLUDE_GPUS:-}"
for g in ${EXCLUDE_GPUS//,/ }; do bad[$g]=1; done

healthy() {
  CUDA_VISIBLE_DEVICES="$1" timeout 60 python3 -c "import torch; torch.cuda.mem_get_info()" >/dev/null 2>&1
}

while true; do
  busy_uuids=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u || true)
  free=()
  while IFS=', ' read -r idx uuid mem; do
    [ -n "${bad[$idx]:-}" ] && continue
    if [ "$mem" -lt 1000 ] && ! grep -q "$uuid" <<<"$busy_uuids"; then free+=("$idx"); fi
  done < <(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits)
  ok=()
  for g in "${free[@]}"; do
    if healthy "$g"; then ok+=("$g"); else bad[$g]=1; echo "$(date +%H:%M) GPU $g failed CUDA health check, skipping" >&2; fi
    [ "${#ok[@]}" -ge "$N" ] && break
  done
  if [ "${#ok[@]}" -ge "$N" ]; then
    (IFS=,; echo "${ok[*]:0:$N}"); exit 0
  fi
  echo "$(date +%H:%M) waiting: healthy free GPUs = [$(IFS=,; echo "${ok[*]}")] (need $N; bad: ${!bad[*]})" >&2
  sleep "$POLL"
done
