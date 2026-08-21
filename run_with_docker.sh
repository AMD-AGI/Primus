#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DOCKER_IMAGE=${DOCKER_IMAGE:-zirui3/primus-v26.3-flux:v0.4}
CONTAINER_NAME=${CONTAINER_NAME:-primus-flux-mlperf}
NOFILE_LIMIT=${NOFILE_LIMIT:-1048576}

if (( ${NNODES:-1} > 1 )) && [[ "${NCCL_IB_DISABLE:-0}" != "1" ]]; then
    LIBIONIC_ABI4_PATH=${LIBIONIC_ABI4_PATH:-/usr/lib/x86_64-linux-gnu/libionic.so.1.0.54.0-149.g3304be71}
    if [[ ! -f "$LIBIONIC_ABI4_PATH" || ! -e /dev/infiniband ]]; then
        echo "[run_with_docker] ABI-4 libionic or /dev/infiniband is unavailable" >&2
        exit 1
    fi
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

export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export DATASET_PATH=${DATASET_PATH:-/data/cc12m_preprocessed}
export EVAL_DATASET_PATH=${EVAL_DATASET_PATH:-/data/coco_preprocessed}
export EMPTY_ENCODINGS_PATH=${EMPTY_ENCODINGS_PATH:-/data/empty_encodings}
export OUTPUT_DIR=${OUTPUT_DIR:-/shared_nfs/zirui/runs/flux_rcp_compare/primus}
export SEED=${SEED:-1234}
export SAVE_STEPS=${SAVE_STEPS:-100}
export CHECKPOINT_KEEP_LATEST=${CHECKPOINT_KEEP_LATEST:-3}
export RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-latest}
export ENABLE_WANDB_LOGGER=${ENABLE_WANDB_LOGGER:-false}
export WANDB_PROJECT=${WANDB_PROJECT:-mlperf-flux-rcp}
export WANDB_RUN_ID=${WANDB_RUN_ID:-primus-flux}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-primus-flux}
export WANDB_RESUME=${WANDB_RESUME:-allow}
export WANDB_OFFLINE=${WANDB_OFFLINE:-false}
export WANDB_JOB_TYPE=${WANDB_JOB_TYPE:-train}

env_names=(
    CONFIG NNODES NODE_RANK MASTER_ADDR MASTER_PORT GPUS_PER_NODE DP_REPLICATE
    DATASET_PATH EVAL_DATASET_PATH EMPTY_ENCODINGS_PATH OUTPUT_DIR PRIMUS_WORKSPACE
    SAVE_STEPS SAVE_STRATEGY CHECKPOINT_KEEP_LATEST RESUME_FROM_CHECKPOINT DISABLE_CHECKPOINT
    ENABLE_WANDB_LOGGER WANDB_PROJECT WANDB_RUN_ID WANDB_RUN_NAME WANDB_RESUME
    WANDB_API_KEY WANDB_ENTITY WANDB_GROUP WANDB_TAGS WANDB_OFFLINE WANDB_JOB_TYPE
    LOCAL_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS MAX_STEPS LR WARMUP_STEPS SEED LOG_FREQ TARGET_ACCURACY
    VAL_CHECK_INTERVAL MLPERF_ENABLE MLPERF_CLEAR_CACHES ATTENTION_BACKEND
    FLUX_PERFORMANCE_MODE MLPERF_WARMUP_TRAIN_STEPS MLPERF_WARMUP_VALIDATION_STEPS
    FLUX_FLOAT8_RECIPE FLUX_FP8_GEMM_BACKEND
    GRADIENT_CHECKPOINTING GRADIENT_CHECKPOINTING_RATIO
    COMPILE_TRANSFORMER_BLOCKS COMPILE_STRATEGY COMPILE_BACKEND
    COMPILE_FULLGRAPH COMPILE_DYNAMIC COMPILE_OUTPUT_HEAD FSDP2_RESHARD_AFTER_FORWARD
    TORCH_COMPILE_MODE FSDP2_REDUCE_DTYPE
    PROFILE PROFILE_RANK PROFILE_WAIT_STEPS PROFILE_WARMUP_STEPS
    PROFILE_ACTIVE_STEPS PROFILE_OUTPUT_DIR PROFILE_WITH_STACK
    PIN_FLUX_T5_STACK
    RUN_TAG LOG_FILE SUMMARY_FILE MLLOG_OUTPUT_FILE RANK_LOG_DIR
    BENCH_SKIP_STEPS TORCHRUN_TEE
    NCCL_IB_DISABLE NCCL_IB_HCA NCCL_IB_GID_INDEX NCCL_IB_TC NCCL_IB_FIFO_TC NCCL_IB_ROCE_VERSION_NUM
    NCCL_IB_USE_INLINE NCCL_IB_QPS_PER_CONNECTION NCCL_IB_RETRY_CNT NCCL_IB_TIMEOUT
    NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME NCCL_NET_PLUGIN NCCL_MAX_P2P_CHANNELS
    NCCL_CROSS_NIC NCCL_PXN_DISABLE NCCL_GDR_FLUSH_DISABLE NCCL_NET_GDR_LEVEL NCCL_DMABUF_ENABLE
    NCCL_IGNORE_CPU_AFFINITY NCCL_DEBUG NCCL_DEBUG_SUBSYS NET_OPTIONAL_RECV_COMPLETION
    RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING USING_AINIC
)
docker_env_args=()
for name in "${env_names[@]}"; do
    if [[ -v "$name" ]]; then
        docker_env_args+=(--env "$name")
    fi
done

docker_tty_args=()
if [[ -t 0 && -t 1 ]]; then
    docker_tty_args=(-it)
fi

rdma_args=()
[[ -e /dev/infiniband ]] && rdma_args=(--device=/dev/infiniband)
libionic_args=()
if [[ -n "${LIBIONIC_ABI4_PATH:-}" ]]; then
    libionic_args=(-v "$LIBIONIC_ABI4_PATH:/usr/lib/x86_64-linux-gnu/libionic.so.1.1.54.0-187:ro")
fi

docker run "${docker_tty_args[@]}" --rm --init \
    --name "$CONTAINER_NAME" \
    --ulimit "nofile=${NOFILE_LIMIT}:${NOFILE_LIMIT}" --ulimit memlock=-1:-1 \
    --device=/dev/kfd --device=/dev/dri "${rdma_args[@]}" \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --ipc=host --network=host --privileged \
    --shm-size=20G \
    -v /shared_nfs:/shared_nfs \
    -v /shared_nfs/zirui/models:/models \
    -v /shared_nfs/zirui/data:/data \
    -v "$SCRIPT_DIR:/workspace/code" "${libionic_args[@]}" \
    -w /workspace/code \
    "${docker_env_args[@]}" \
    "$DOCKER_IMAGE" \
    bash local_runs/run_flux_mlperf.sh
