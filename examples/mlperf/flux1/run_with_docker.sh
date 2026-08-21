#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)

: "${DATA_ROOT:?Set DATA_ROOT to the directory containing the MLPerf datasets}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the output directory}"

export DOCKER_IMAGE=${DOCKER_IMAGE:-zirui3/primus-v26.3-flux:v0.4}
export CONTAINER_NAME=${CONTAINER_NAME:-primus-mlperf-flux1-${SLURM_JOB_ID:-local}-${SLURM_PROCID:-0}}
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-${SLURM_NODEID:-0}}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export CONFIG=${CONFIG:-examples/mlperf/flux1/flux.1_schnell_t2i-pretrain.yaml}
export PRIMUS_WORKSPACE=${PRIMUS_WORKSPACE:-/output/primus_workspace}
export DATASET_PATH=${DATASET_PATH:-/data/cc12m_preprocessed}
export EVAL_DATASET_PATH=${EVAL_DATASET_PATH:-/data/coco_preprocessed}
export EMPTY_ENCODINGS_PATH=${EMPTY_ENCODINGS_PATH:-/data/empty_encodings}
export OUTPUT_DIR=${OUTPUT_DIR:-/output/flux_mlperf}
export MLLOG_OUTPUT_FILE=${MLLOG_OUTPUT_FILE:-$OUTPUT_DIR/mlperf_compliance.log}
export FLUX_FLOAT8_RECIPE=${FLUX_FLOAT8_RECIPE:-tensorwise}
export FLUX_FP8_GEMM_BACKEND=${FLUX_FP8_GEMM_BACKEND:-selective_flydsl}
export ATTENTION_BACKEND=${ATTENTION_BACKEND:-flash_attn_aiter}
if (( NNODES == 4 )); then
    export DP_REPLICATE=${DP_REPLICATE:-4}
    export LOCAL_BATCH_SIZE=${LOCAL_BATCH_SIZE:-32}
    export LR=${LR:-0.00025}
    export WARMUP_STEPS=${WARMUP_STEPS:-800}
    export GRADIENT_CHECKPOINTING_RATIO=${GRADIENT_CHECKPOINTING_RATIO:-0}
    export FSDP2_REDUCE_DTYPE=${FSDP2_REDUCE_DTYPE:-bf16}
else
    export DP_REPLICATE=${DP_REPLICATE:-1}
    export LOCAL_BATCH_SIZE=${LOCAL_BATCH_SIZE:-64}
    export LR=${LR:-2e-4}
    export WARMUP_STEPS=${WARMUP_STEPS:-1600}
    export GRADIENT_CHECKPOINTING_RATIO=${GRADIENT_CHECKPOINTING_RATIO:-0.25}
    export FSDP2_REDUCE_DTYPE=${FSDP2_REDUCE_DTYPE:-fp32}
fi
export GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
export MAX_STEPS=${MAX_STEPS:-30000}
export COMPILE_TRANSFORMER_BLOCKS=${COMPILE_TRANSFORMER_BLOCKS:-true}
export COMPILE_STRATEGY=${COMPILE_STRATEGY:-per_block}
export COMPILE_BACKEND=${COMPILE_BACKEND:-inductor}
export COMPILE_FULLGRAPH=${COMPILE_FULLGRAPH:-true}
export COMPILE_DYNAMIC=${COMPILE_DYNAMIC:-false}
export FSDP2_RESHARD_AFTER_FORWARD=${FSDP2_RESHARD_AFTER_FORWARD:-false}
export FLUX_PERFORMANCE_MODE=${FLUX_PERFORMANCE_MODE:-nemo_mlperf}
export SAVE_STEPS=${SAVE_STEPS:-100}
export SAVE_STRATEGY=${SAVE_STRATEGY:-dtcp_full}
export CHECKPOINT_KEEP_LATEST=${CHECKPOINT_KEEP_LATEST:-3}
export RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-latest}
export MLPERF_ENABLE=${MLPERF_ENABLE:-true}
export MLPERF_CLEAR_CACHES=${MLPERF_CLEAR_CACHES:-true}
export TARGET_ACCURACY=${TARGET_ACCURACY:-0.586}
export VAL_CHECK_INTERVAL=${VAL_CHECK_INTERVAL:-262144}
export SEED=${SEED:-10007}

if [[ "$LOCAL_BATCH_SIZE" == "32" && "$FLUX_FP8_GEMM_BACKEND" == "selective_flydsl" ]]; then
    export TORCHINDUCTOR_BENCHMARK_FUSION=${TORCHINDUCTOR_BENCHMARK_FUSION:-1}
    export PRIMUS_FLUX_AITER_ATOMIC_FP32=${PRIMUS_FLUX_AITER_ATOMIC_FP32:-0}
    export PRIMUS_FLUX_REUSE_FP8_INPUT=${PRIMUS_FLUX_REUSE_FP8_INPUT:-1}
fi

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
    NNODES NODE_RANK MASTER_ADDR MASTER_PORT GPUS_PER_NODE DP_REPLICATE CONFIG
    PRIMUS_WORKSPACE DATASET_PATH EVAL_DATASET_PATH EMPTY_ENCODINGS_PATH OUTPUT_DIR
    MLLOG_OUTPUT_FILE FLUX_FLOAT8_RECIPE FLUX_FP8_GEMM_BACKEND ATTENTION_BACKEND
    LOCAL_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS MAX_STEPS LR WARMUP_STEPS
    GRADIENT_CHECKPOINTING_RATIO COMPILE_TRANSFORMER_BLOCKS COMPILE_STRATEGY
    COMPILE_BACKEND COMPILE_FULLGRAPH COMPILE_DYNAMIC FSDP2_RESHARD_AFTER_FORWARD
    FSDP2_REDUCE_DTYPE FLUX_PERFORMANCE_MODE SAVE_STEPS SAVE_STRATEGY
    CHECKPOINT_KEEP_LATEST RESUME_FROM_CHECKPOINT MLPERF_ENABLE MLPERF_CLEAR_CACHES
    TARGET_ACCURACY VAL_CHECK_INTERVAL SEED TORCHINDUCTOR_BENCHMARK_FUSION
    PRIMUS_FLUX_AITER_ATOMIC_FP32 PRIMUS_FLUX_REUSE_FP8_INPUT NCCL_IB_DISABLE
    NCCL_IB_HCA NCCL_IB_GID_INDEX NCCL_IB_TC NCCL_IB_FIFO_TC NCCL_IB_ROCE_VERSION_NUM
    NCCL_IB_USE_INLINE NCCL_IB_QPS_PER_CONNECTION NCCL_IB_RETRY_CNT NCCL_IB_TIMEOUT
    NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME NCCL_NET_PLUGIN NCCL_MAX_P2P_CHANNELS
    NCCL_GDR_FLUSH_DISABLE NCCL_DMABUF_ENABLE NCCL_IGNORE_CPU_AFFINITY NCCL_CROSS_NIC
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
        ./primus-cli direct -- train pretrain --config "$CONFIG"
    '
