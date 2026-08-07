#!/bin/bash
# MLPerf Llama2-70B LoRA (Megatron-Bridge posttrain) — timed run with :::MLLOG output.
#
# Hooks run automatically before training:
#   00_install_requirements.sh — pip deps (+ mlperf-logging when ENABLE_MLLOG=1)
#   01_convert_checkpoints.sh    — HF → Megatron checkpoint (needs HF_TOKEN)
#   02_prepare_mlperf_dataset.sh — SCROLLS gov-report .npy + metadata (needs HF_TOKEN)
#
# Usage (inside Primus container):
#   export HF_TOKEN=...
#   source examples/mlperf/llama2_70b/config_MI355X_1x8x1.sh
#   bash examples/mlperf/llama2_70b/run_and_time.sh
#
# Optional:
#   PACKED_DATA_DIR=/data
#   MLPERF_VERBOSE_LOGS=1
#   PRIMUS_LOG_GPU_MEM=0

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRIMUS_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

RESULTS_DIR=/results

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

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/config_MI355X_1x8x1.sh"

# Create results directory (:::MLLOG + timed run logs)
mkdir -p "${RESULTS_DIR}/logs"

cd "${PRIMUS_PATH}"

# Under multi-node SLURM (run_with_docker_slurm.sh), inherit rendezvous + node
# sizing from SLURM env so we can scale to N nodes without editing the config
# file. Single-node SLURM jobs (NNODES=1) fall through to the config defaults
# so torchrun doesn't try to do c10d rdzv against MASTER_ADDR=localhost.
if [[ -n "${SLURM_NNODES:-}" && "${SLURM_NNODES}" -gt 1 ]]; then
    NNODES="${SLURM_NNODES}"
    NODE_RANK="${SLURM_NODEID:-0}"
fi

TRAIN_EXP_LOG="${RESULTS_DIR}/train.mlperfposttrain.exp.log"
PRIMUS_CLI_LOG="${RESULTS_DIR}/logs/log_$(date +%Y%m%d_%H%M%S).txt"

echo "============================================"
echo "MLPerf Llama2-70B LoRA Post-Train (SFT)"
echo "============================================"
echo "Config:  ${EXP}"
echo "Data:    ${DATA_PATH}"
echo "MLLOG:   ${MLLOG_OUTPUT_FILE}"
echo "Run log: ${TRAIN_EXP_LOG}"
echo "GPUs:    ${GPUS_PER_NODE}"
echo "Nodes:   ${NNODES}"
echo "Rank:    ${NODE_RANK}"
echo "Master:  ${MASTER_ADDR}:${MASTER_PORT}"
echo "============================================"

# Start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# Launch through Primus CLI and keep the real exit code even though output is
# piped through tee. --log_file keeps primus-cli from mkdir logs/ in cwd.
set +e
./primus-cli direct --log_file "${PRIMUS_CLI_LOG}" -- \
    train posttrain \
    --config "${EXP}" \
    2>&1 | tee "${TRAIN_EXP_LOG}"
ret_code=${PIPESTATUS[0]}
set -e

# End timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# Report result (wall-clock seconds for MLPerf timing scripts)
result=$(( end - start ))
result_name="LLAMA2_70B_LORA"
echo "RESULT,$result_name,,$result,AMD,$start_fmt"

if [[ $ret_code != 0 ]]; then
    echo "Training failed with exit code: $ret_code"
    exit "$ret_code"
fi

exit 0
