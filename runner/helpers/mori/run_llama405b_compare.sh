#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

set -euo pipefail

BACKEND="${1:-}"
case "${BACKEND}" in
    rccl | mori) ;;
    *)
        echo "Usage: $0 {rccl|mori}" >&2
        exit 2
        ;;
esac

# Edit these values for the experiment you want to run.
NODES="${NODES:-smci355-ccs-aus-n04-33,smci355-ccs-aus-n05-21}"
IMAGE="${IMAGE:-unifiedtrainingdockers.azurecr.io/utd/nightly:primus_the_rock_rocm7.15_20260728}"
CONFIG="${CONFIG:-examples/torchtitan/configs/MI355X/llama3.1_405B-BF16-pretrain.yaml}"
MODEL_LAYERS="${MODEL_LAYERS:-64}" # Set 126 for the full 405B model.
TRAINING_STEPS="${TRAINING_STEPS:-21}"
# SEQ_LEN="${SEQ_LEN:-128}"
LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-1}"
MOCK_DATA="${MOCK_DATA:-True}"
# DISABLE_COMPILE="${DISABLE_COMPILE:-1}"
# DISABLE_PRIMUS_TURBO="${DISABLE_PRIMUS_TURBO:-1}"

# profile_freq=20, warmup=1, active=1 records iteration 19 and saves at 20.
PROFILE_STEP="${PROFILE_STEP:-19}"
PROFILER_WARMUP="${PROFILER_WARMUP:-1}"
PROFILER_ACTIVE="${PROFILER_ACTIVE:-1}"
PROFILE_FREQ="$((PROFILE_STEP + 1))"

PRIMUS_ROOT="${PRIMUS_ROOT:-/apps/tas/lorrirao/sdma_rccl_pytorch/primus}"
SHARED_DATA="${SHARED_DATA:-/apps/tas/lorrirao/sdma_rccl_pytorch/mori_multinode_data}"
HF_TOKEN_PATH="${HF_TOKEN_PATH:-/apps/tas/lorrirao/.cache/huggingface/token}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/apps/tas/lorrirao/sdma_rccl_pytorch/mori_perf_compare_405b}"

# Optional local tokenizer/assets override. Leave empty to use SHARED_DATA.
HF_ASSETS_HOST_DIR="${HF_ASSETS_HOST_DIR:-}"
HF_ASSETS_CONTAINER_DIR="${HF_ASSETS_CONTAINER_DIR:-/workspace/Primus/data/torchtitan/Llama-3.1-405B}"

MASTER_PORT="${MASTER_PORT:-29660}"
SOCKET_IFNAME="${SOCKET_IFNAME:-fenic}"
NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-1}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"

IFS="," read -r -a NODE_ARRAY <<<"${NODES}"
NNODES="${#NODE_ARRAY[@]}"
if ((NNODES != 2)); then
    echo "This comparison script expects exactly two nodes, got ${NNODES}." >&2
    exit 2
fi

quote_cmd() {
    printf "%q " "$@"
}

remote() {
    local node="$1"
    shift
    if [[ "${node}" == "$(hostname -s)" || "${node}" == "$(hostname -f)" ]]; then
        "$@"
    else
        ssh -o BatchMode=yes "${node}" "$(quote_cmd "$@")"
    fi
}

MASTER_ADDR="${MASTER_ADDR:-}"
if [[ -z "${MASTER_ADDR}" ]]; then
    MASTER_ADDR="$(
        remote "${NODE_ARRAY[0]}" bash -lc \
            "ip -o -4 addr show dev $(printf '%q' "${SOCKET_IFNAME}") scope global |
             awk 'NR==1{split(\$4,a,\"/\"); print a[1]}'"
    )"
fi
if [[ -z "${MASTER_ADDR}" ]]; then
    echo "Unable to determine MASTER_ADDR on ${NODE_ARRAY[0]}/${SOCKET_IFNAME}." >&2
    exit 1
fi

mkdir -p "${OUTPUT_ROOT}/${BACKEND}"

TRAIN_ARGS=(
    --
    train
    pretrain
    --config
    "${CONFIG}"
    --training.steps
    "${TRAINING_STEPS}"
    --training.mock_data
    "${MOCK_DATA}"
    --training.local_batch_size
    "${LOCAL_BATCH_SIZE}"
    --metrics.log_freq
    1
    --metrics.enable_tensorboard
    True
    --metrics.save_tb_folder
    tb
    --metrics.disable_color_printing
    True
    --profiling.enable_profiling
    True
    --profiling.save_traces_folder
    profile_traces
    --profiling.profile_freq
    "${PROFILE_FREQ}"
    --profiling.profiler_warmup
    "${PROFILER_WARMUP}"
    --profiling.profiler_active
    "${PROFILER_ACTIVE}"
    --job.dump_folder
    "/workspace/results/${BACKEND}"
)

[[ -n "${MODEL_LAYERS}" ]] && TRAIN_ARGS+=(--model.n_layers "${MODEL_LAYERS}")
[[ -n "${SEQ_LEN:-}" ]] && TRAIN_ARGS+=(--training.seq_len "${SEQ_LEN}")
[[ "${DISABLE_COMPILE:-0}" == "1" ]] && TRAIN_ARGS+=(--compile.enable False)
[[ "${DISABLE_PRIMUS_TURBO:-0}" == "1" ]] &&
    TRAIN_ARGS+=(--primus_turbo.enable_primus_turbo False)
if [[ -n "${EXTRA_TRAIN_ARGS}" ]]; then
    read -r -a extra_args <<<"${EXTRA_TRAIN_ARGS}"
    TRAIN_ARGS+=("${extra_args[@]}")
fi
printf -v TRAIN_ARGS_QUOTED "%q " "${TRAIN_ARGS[@]}"

container_name() {
    local rank="$1"
    echo "llama405_${BACKEND}_${USER}_${rank}"
}

cleanup() {
    local rank
    for rank in "${!NODE_ARRAY[@]}"; do
        remote "${NODE_ARRAY[rank]}" docker rm -f "$(container_name "${rank}")" \
            >/dev/null 2>&1 || true
    done
}
trap cleanup EXIT INT TERM

launch_rank() {
    local node="$1"
    local rank="$2"
    local name
    name="$(container_name "${rank}")"

    local docker_args=(
        docker run --rm
        --name "${name}"
        --device=/dev/kfd
        --device=/dev/dri
        --group-add video
        --cap-add SYS_PTRACE
        --security-opt seccomp=unconfined
        --privileged
        --ipc=host
        --network=host
        -v "${PRIMUS_ROOT}/primus:/workspace/Primus/primus:ro"
        -v "${PRIMUS_ROOT}/runner:/workspace/Primus/runner:ro"
        -v "${PRIMUS_ROOT}/examples:/workspace/Primus/examples:ro"
        -v "${SHARED_DATA}:/workspace/Primus/data"
        -v "${OUTPUT_ROOT}:/workspace/results"
        -v "${HF_TOKEN_PATH}:/run/hf_token:ro"
        -e HF_TOKEN_FILE=/run/hf_token
        -e NNODES="${NNODES}"
        -e NODE_RANK="${rank}"
        -e GPUS_PER_NODE=8
        -e MASTER_ADDR="${MASTER_ADDR}"
        -e MASTER_PORT="${MASTER_PORT}"
        -e NCCL_SOCKET_IFNAME="${SOCKET_IFNAME}"
        -e GLOO_SOCKET_IFNAME="${SOCKET_IFNAME}"
        -e NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX}"
    )
    if [[ -n "${HF_ASSETS_HOST_DIR}" ]]; then
        docker_args+=(-v "${HF_ASSETS_HOST_DIR}:${HF_ASSETS_CONTAINER_DIR}:ro")
    fi
    if [[ "${BACKEND}" == "mori" ]]; then
        docker_args+=(
            -e FSDP_ALL_GATHER_BACKEND=mori
            -e MORI_SOCKET_IFNAME="${SOCKET_IFNAME}"
            -e MORI_HIER_CUDA_GRAPH=0
            -e MORI_FSDP_COMPACT_WORKSPACE="${MORI_FSDP_COMPACT_WORKSPACE:-1}"
        )
    fi

    remote "${node}" docker rm -f "${name}" >/dev/null 2>&1 || true
    remote "${node}" \
        "${docker_args[@]}" \
        "${IMAGE}" \
        bash -lc \
        "export HF_TOKEN=\"\$(< /run/hf_token)\"; \
         cd /workspace/Primus; \
         bash runner/primus-cli direct \
           --log_file /workspace/results/${BACKEND}_node${rank}.log \
           ${TRAIN_ARGS_QUOTED}"
}

echo "Backend      : ${BACKEND}"
echo "Nodes        : ${NODE_ARRAY[*]}"
echo "Master       : ${MASTER_ADDR}:${MASTER_PORT}"
echo "Config       : ${CONFIG}"
echo "Layers       : ${MODEL_LAYERS:-config default}"
echo "Steps        : ${TRAINING_STEPS}"
echo "Profile step : ${PROFILE_STEP}"
echo "Output       : ${OUTPUT_ROOT}/${BACKEND}"

pids=()
for rank in "${!NODE_ARRAY[@]}"; do
    node="${NODE_ARRAY[rank]}"
    launch_rank "${node}" "${rank}" \
        >"${OUTPUT_ROOT}/${BACKEND}/launcher-node${rank}.log" 2>&1 &
    pids+=("$!")
    [[ "${rank}" == "0" ]] && sleep 1
done

status=0
for pid in "${pids[@]}"; do
    wait "${pid}" || status=$?
done
if [[ "${status}" -ne 0 ]]; then
    echo "${BACKEND} run failed. Logs: ${OUTPUT_ROOT}/${BACKEND}" >&2
    exit "${status}"
fi

echo "${BACKEND} run passed. Results: ${OUTPUT_ROOT}/${BACKEND}"
