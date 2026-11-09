#!/bin/bash

set -e

export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-}

readonly node_rank="${SLURM_NODEID:-0}"
readonly local_rank="${LOCAL_RANK:=${SLURM_LOCALID:=${OMPI_COMM_WORLD_LOCAL_RANK:-0}}}"

if [[ -z "${SEED}" ]]; then
    echo "SEED is not set!"
    exit 1
fi

# Create results directory
mkdir -p /results

cd /workspace/code

# Config env (EXP, batch, GEMM backend, FP4 dispatch) must be set even if
# docker exec only forwarded empty --env=NAME copies from the host.
if [[ -n "${DGXSYSTEM:-}" && -f "./config_${DGXSYSTEM}.sh" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "./config_${DGXSYSTEM}.sh"
    set +a
fi

: "${PRIMUS_PATH:=/workspace/Primus}"
if [[ ! -x "${PRIMUS_PATH}/primus-cli" ]]; then
    echo "primus-cli not found under PRIMUS_PATH=${PRIMUS_PATH}" >&2
    exit 1
fi
echo "============================================"
echo "MLPerf Flux1 Training (Primus diffusion backend)"
echo "============================================"
echo "Config:  ${EXP:-N/A}"
echo "Primus:  ${PRIMUS_PATH} ($(git -C "${PRIMUS_PATH}" rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout'))"
echo "Data:    ${DATASET_PATH:-${DATA_PATH:-/dataset/cc12m_preprocessed}}"
echo "GPUs:    ${GPUS_PER_NODE:-${DGXNGPU:-8}}"
echo "Nodes:   ${NNODES:-${DGXNNODES:-1}}"
echo "MBS/GBS: ${LOCAL_BATCH_SIZE:-?} / ${GLOBAL_BATCH_SIZE:-?}"
echo "GEMM:    ${FLUX_FP8_GEMM_BACKEND:-default}"
echo "MXFP4:   PASSES=${FLUX_FP4_PASSES:-off} BACKEND=${FLUX_FP4_GEMM_BACKEND:-} DISPATCH=${FLUX_FP4_DISPATCH:-} QKV_WGRAD_BF16=${FLUX_FP4_QKV_WGRAD_BF16:-0} H16_STOCK=${FLUX_FP4_H16_STOCK:-0} NORM_ROPE=${FLUX_FUSED_NORM_ROPE:-0}"
echo "Compile: ${TORCH_COMPILE_MODE:-default}"
echo "DP_rep:  ${DP_REPLICATE:-?}"
echo "Cache:   ${TORCHINDUCTOR_CACHE_SEED:-none}"
echo "NCCL:    plugin=${NCCL_NET_PLUGIN:-default} if=${NCCL_SOCKET_IFNAME:-default}"
echo "Seed:    ${SEED}"
echo "FP8 fwd: after step ${FLUX_FPROP_SWITCH_STEP:-off}"
echo "============================================"

# Start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# primus-cli owns the launch; it reads NNODES/NODE_RANK/MASTER_ADDR/MASTER_PORT.
if [[ -n "${SLURM_NNODES:-}" ]] && [[ "${SLURM_NNODES}" -gt 1 ]]; then
    export NNODES=${NNODES:-${SLURM_NNODES}}
    export NODE_RANK=${NODE_RANK:-${SLURM_NODEID:-0}}
    if [[ -z "${MASTER_PORT:-}" ]]; then
        if [[ -n "${SLURM_JOB_ID:-}" ]]; then
            export MASTER_PORT=$((20000 + (SLURM_JOB_ID % 20000)))
        else
            export MASTER_PORT=29501
        fi
    fi
    echo "Multi-node SLURM: ${NNODES} nodes, rank ${NODE_RANK}, master ${MASTER_ADDR}:${MASTER_PORT}"
fi

if [[ "${node_rank}" -eq 0 && "${local_rank}" -eq 0 ]]; then
    echo "SEED=${SEED}"
fi

if [[ "${MLPERF_CLEAR_CACHES}" == "true" ]]; then
    sync
    echo 3 > /proc/sys/vm/drop_caches || true
fi

# Training params come from $EXP plus the env exported by config_*.sh.
set +e
cd "${PRIMUS_PATH}"
# EXTRA_ARGS is a passthrough word list from the caller.
# shellcheck disable=SC2086
./primus-cli direct -- train pretrain --config "${EXP}" ${EXTRA_ARGS:-}
ret_code=$?
set -e

# End timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# Report result
result=$(( end - start ))
result_name="FLUX1"
echo "RESULT,$result_name,,$result,AMD,$start_fmt"

if [[ $ret_code != 0 ]]; then
    echo "Training failed with exit code: $ret_code"
    exit $ret_code
fi

exit 0
