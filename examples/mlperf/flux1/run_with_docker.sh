#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)

: "${DATA_ROOT:?Set DATA_ROOT to the directory containing the MLPerf datasets}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the output directory}"

export FLUX_CONFIG=${FLUX_CONFIG:-config_4n_gbs1024.sh}
[[ -f "$SCRIPT_DIR/$FLUX_CONFIG" ]] || { echo "Unknown FLUX_CONFIG: $FLUX_CONFIG" >&2; exit 2; }
source "$SCRIPT_DIR/$FLUX_CONFIG"

export DOCKER_IMAGE=${DOCKER_IMAGE:-zirui3/primus-v26.3-flux:v0.4}
export CONTAINER_NAME=${CONTAINER_NAME:-primus-mlperf-flux1-${SLURM_JOB_ID:-local}-${SLURM_PROCID:-0}}
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-${SLURM_NODEID:-0}}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export LAUNCH_MODE=${LAUNCH_MODE:-native}
if [[ -z "${CONFIG:-}" ]]; then
    if [[ "$LAUNCH_MODE" == "native" ]]; then
        CONFIG=examples/mlperf/flux1/flux.1_schnell_t2i-native.yaml
    else
        CONFIG=examples/mlperf/flux1/flux.1_schnell_t2i-pretrain.yaml
    fi
fi
export CONFIG
export PRIMUS_WORKSPACE=${PRIMUS_WORKSPACE:-/output/primus_workspace}
export DATASET_PATH=${DATASET_PATH:-/data/cc12m_preprocessed}
export EVAL_DATASET_PATH=${EVAL_DATASET_PATH:-/data/coco_preprocessed}
export EMPTY_ENCODINGS_PATH=${EMPTY_ENCODINGS_PATH:-/data/empty_encodings}
export OUTPUT_DIR=${OUTPUT_DIR:-/output/flux_mlperf}
export MLLOG_OUTPUT_FILE=${MLLOG_OUTPUT_FILE:-$OUTPUT_DIR/mlperf_compliance.log}
export PROFILE=${PROFILE:-false}
export PROFILE_RANK=${PROFILE_RANK:-0}
export PROFILE_WAIT_STEPS=${PROFILE_WAIT_STEPS:-30}
export PROFILE_WARMUP_STEPS=${PROFILE_WARMUP_STEPS:-2}
export PROFILE_ACTIVE_STEPS=${PROFILE_ACTIVE_STEPS:-5}
export PROFILE_OUTPUT_DIR=${PROFILE_OUTPUT_DIR:-$OUTPUT_DIR/torch_profile}
export PROFILE_WITH_STACK=${PROFILE_WITH_STACK:-false}

actual_gbs=$((NNODES * GPUS_PER_NODE * LOCAL_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
[[ "$actual_gbs" == "$GLOBAL_BATCH_SIZE" ]] || {
    echo "Invalid $FLUX_CONFIG: expected GBS=$GLOBAL_BATCH_SIZE, got $actual_gbs" >&2
    exit 2
}
echo "[flux1] config=$FLUX_CONFIG nnodes=$NNODES mbs=$LOCAL_BATCH_SIZE ga=$GRADIENT_ACCUMULATION_STEPS gbs=$actual_gbs"

if (( NNODES > 1 )) && [[ "${NCCL_IB_DISABLE:-0}" != "1" ]]; then
    LIBIONIC_ABI4_PATH=${LIBIONIC_ABI4_PATH:-/usr/lib/x86_64-linux-gnu/libionic.so.1.0.54.0-149.g3304be71}
    [[ -f "$LIBIONIC_ABI4_PATH" && -e /dev/infiniband ]] || {
        echo "ABI-4 libionic or /dev/infiniband is unavailable" >&2
        exit 1
    }
    export NCCL_IB_DISABLE=0
    export NCCL_IB_HCA=${NCCL_IB_HCA:-ionic_0:1,ionic_1:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_6:1,ionic_7:1}
    export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-1}
    export NCCL_IB_TC=${NCCL_IB_TC:-104}
    export NCCL_IB_FIFO_TC=${NCCL_IB_FIFO_TC:-192}
    export NCCL_IB_ROCE_VERSION_NUM=${NCCL_IB_ROCE_VERSION_NUM:-2}
    export NCCL_IB_USE_INLINE=${NCCL_IB_USE_INLINE:-1}
    export NCCL_IB_QPS_PER_CONNECTION=${NCCL_IB_QPS_PER_CONNECTION:-1}
    export NCCL_IB_RETRY_CNT=${NCCL_IB_RETRY_CNT:-20}
    export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-300}
    export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ens3}
    export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-ens3}
    export NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-librccl-anp.so}
    export NCCL_MAX_P2P_CHANNELS=${NCCL_MAX_P2P_CHANNELS:-56}
    export NCCL_GDR_FLUSH_DISABLE=${NCCL_GDR_FLUSH_DISABLE:-1}
    export NCCL_DMABUF_ENABLE=${NCCL_DMABUF_ENABLE:-0}
    export NCCL_IGNORE_CPU_AFFINITY=${NCCL_IGNORE_CPU_AFFINITY:-1}
    export NCCL_CROSS_NIC=${NCCL_CROSS_NIC:-0}
    export NET_OPTIONAL_RECV_COMPLETION=${NET_OPTIONAL_RECV_COMPLETION:-1}
    export RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=${RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING:-0}
fi

env_names=(
    FLUX_CONFIG LAUNCH_MODE NNODES NODE_RANK MASTER_ADDR MASTER_PORT GPUS_PER_NODE DP_REPLICATE CONFIG
    PRIMUS_WORKSPACE DATASET_PATH EVAL_DATASET_PATH EMPTY_ENCODINGS_PATH OUTPUT_DIR
    MLLOG_OUTPUT_FILE FLUX_FLOAT8_RECIPE FLUX_FP8_GEMM_BACKEND ATTENTION_BACKEND
    LOCAL_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS GLOBAL_BATCH_SIZE MAX_STEPS LR WARMUP_STEPS
    GRADIENT_CHECKPOINTING GRADIENT_CHECKPOINTING_RATIO COMPILE_TRANSFORMER_BLOCKS COMPILE_STRATEGY
    COMPILE_BACKEND COMPILE_FULLGRAPH COMPILE_DYNAMIC COMPILE_OUTPUT_HEAD TORCH_COMPILE_MODE
    FSDP2_RESHARD_AFTER_FORWARD FSDP2_REDUCE_DTYPE
    PROFILE PROFILE_RANK PROFILE_WAIT_STEPS PROFILE_WARMUP_STEPS PROFILE_ACTIVE_STEPS
    PROFILE_OUTPUT_DIR PROFILE_WITH_STACK FSDP2_HSDP_FP8_ALL_REDUCE FSDP2_HSDP_FP8_BLOCK_SIZE PIN_FLUX_T5_STACK
    FLUX_PERFORMANCE_MODE SAVE_STEPS SAVE_STRATEGY CHECKPOINT_KEEP_LATEST
    RESUME_FROM_CHECKPOINT MLPERF_ENABLE MLPERF_WARMUP_TRAIN_STEPS
    MLPERF_WARMUP_VALIDATION_STEPS MLPERF_CLEAR_CACHES TARGET_ACCURACY
    VAL_CHECK_INTERVAL SEED LOG_FREQ ENABLE_WANDB_LOGGER TORCHINDUCTOR_BENCHMARK_FUSION
    PRIMUS_FLUX_AITER_ATOMIC_FP32 PRIMUS_FLUX_REUSE_FP8_INPUT NCCL_IB_DISABLE
    NCCL_IB_HCA NCCL_IB_GID_INDEX NCCL_IB_TC NCCL_IB_FIFO_TC NCCL_IB_ROCE_VERSION_NUM
    NCCL_IB_USE_INLINE NCCL_IB_QPS_PER_CONNECTION NCCL_IB_RETRY_CNT NCCL_IB_TIMEOUT
    NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME NCCL_NET_PLUGIN NCCL_MAX_P2P_CHANNELS
    NCCL_GDR_FLUSH_DISABLE NCCL_DMABUF_ENABLE NCCL_NET_GDR_LEVEL NCCL_NET_GDR_READ
    NCCL_IGNORE_CPU_AFFINITY NCCL_CROSS_NIC NCCL_DEBUG NCCL_DEBUG_SUBSYS
    NET_OPTIONAL_RECV_COMPLETION RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING
)
docker_env_args=()
for name in "${env_names[@]}"; do
    [[ -v "$name" ]] && docker_env_args+=(--env "$name")
done

rdma_args=()
libionic_args=()
[[ -e /dev/infiniband ]] && rdma_args=(--device=/dev/infiniband)
[[ -n "${LIBIONIC_ABI4_PATH:-}" ]] && libionic_args=(-v "$LIBIONIC_ABI4_PATH:/usr/lib/x86_64-linux-gnu/libionic.so.1.1.54.0-187:ro")
mkdir -p "$OUTPUT_ROOT"

exec docker run --rm --init --privileged \
    --name "$CONTAINER_NAME" \
    --ulimit nofile=1048576:1048576 --ulimit memlock=-1:-1 \
    --device=/dev/kfd --device=/dev/dri "${rdma_args[@]}" --group-add video \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --ipc=host --network=host --shm-size=20G \
    -v "$REPO_ROOT:/workspace/Primus" \
    -v "$DATA_ROOT:/data" \
    -v "$OUTPUT_ROOT:/output" \
    -v /shared_nfs:/shared_nfs \
    "${libionic_args[@]}" \
    -w /workspace/Primus \
    "${docker_env_args[@]}" \
    "$DOCKER_IMAGE" bash -lc '
        set -euo pipefail
        mkdir -p "$OUTPUT_DIR"
        if [[ "$MLPERF_CLEAR_CACHES" == "true" ]]; then
            sync
            echo 3 > /proc/sys/vm/drop_caches
        fi
        if [[ "$LAUNCH_MODE" == "native" ]]; then
            torchrun \
              --nnodes="$NNODES" \
              --node_rank="$NODE_RANK" \
              --nproc_per_node="$GPUS_PER_NODE" \
              --master_addr="$MASTER_ADDR" \
              --master_port="$MASTER_PORT" \
              examples/diffusion/train_native.py --config "$CONFIG"
        elif [[ "$LAUNCH_MODE" == "primus" ]]; then
            ./primus-cli direct -- train pretrain --config "$CONFIG"
        else
            echo "Unknown LAUNCH_MODE: $LAUNCH_MODE" >&2
            exit 2
        fi
    '
