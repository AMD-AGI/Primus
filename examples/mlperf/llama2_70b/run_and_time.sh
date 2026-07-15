#!/bin/bash
# One-shot MLPerf Llama2-70B LoRA via primus-cli (dataset + checkpoint hooks).
#
# MLPerf overrides live in primus/backends/megatron_bridge/patches/mlperf_llama2_70b/ and are
# applied automatically via @register_patch when llama2_70b_lora_mxfp4 is selected.
#
# Hooks (megatron_bridge) run automatically before training:
#   00_install_requirements.sh — pip deps
#   01_convert_checkpoints.sh    — HF → Megatron checkpoint (needs HF_TOKEN)
#   02_prepare_mlperf_dataset.sh — SCROLLS gov-report .npy + metadata (needs HF_TOKEN)
#
# Usage (inside Primus container, repo root):
#   export HF_TOKEN=...
#   bash examples/mlperf/llama2_70b/run_and_time.sh
#
# Optional:
#   PACKED_DATA_DIR=/data/mlperf_llama2
#   MLLOG_VERBOSE_LOGS=1
#   PRIMUS_LOG_GPU_MEM=1   # GPU mem every log_interval (default on in config_MI355X)
#   SEED=1234

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRIMUS_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PRIMUS_ROOT}"

DATA_ROOT="${PACKED_DATA_DIR:-${DATA_PATH:-${PRIMUS_ROOT}/data/mlperf_llama2}}"
export PACKED_DATA_DIR="${DATA_ROOT}"
export DATA_PATH="${DATA_ROOT}"
export HF_HOME="${HF_HOME:-${DATA_ROOT}/.cache/huggingface}"
mkdir -p "${DATA_ROOT}" "${HF_HOME}"

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "[ERROR] HF_TOKEN is required (meta-llama/Llama-2-70b-hf + MLPerf dataset hub access)." >&2
    exit 1
fi
export HF_TOKEN

export SEED="${SEED:-$RANDOM}"

# Exact MLPerf 6.0 MI355X env (MXFP4, AITER, NCCL, MLLOG, 550 iters, lr=0.0006, ...)
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/config_MI355X_1x8x1.sh"

CONFIG="${CONFIG:-${EXP}}"

echo "[INFO] Primus root:     ${PRIMUS_ROOT}"
echo "[INFO] Data root:       ${DATA_ROOT}"
echo "[INFO] HF cache:        ${HF_HOME}"
echo "[INFO] Training config: ${CONFIG}"

exec ./runner/primus-cli direct train posttrain --config "${CONFIG}" "$@"
