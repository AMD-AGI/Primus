#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

print_usage() {
cat << EOF
Usage: bash $(basename "$0") [--help]

Environment variables (must set before running):

    EXP                              # Path to experiment config file (required)
    NNODES=1                         # Number of nodes (default: 1)
    NODE_RANK=0                      # Current node rank (default: 0)
    GPUS_PER_NODE=8                  # Number of GPUs per node (default: 8)
    MASTER_ADDR=localhost            # Master node address (default: localhost)
    MASTER_PORT=1234                 # Master node port (default: 1234)
    PRIMUS_HIPBLASLT_TUNING_STAGE=0  # HipBLASLt tuning stage: 0/1/2/3 (default: 0)
EOF
}

if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    print_usage
    exit 0
fi

HOSTNAME=$(hostname)

LOG_INFO() {
    if [ "$*" = "" ]; then echo ""; else echo "[NODE-$NODE_RANK($HOSTNAME)] [INFO] $*"; fi
}
LOG_INFO_RANK0() {
    if [ "$NODE_RANK" -eq 0 ]; then
        if [ "$*" = "" ]; then echo ""; else echo "[NODE-$NODE_RANK($HOSTNAME)] [INFO] $*"; fi
    fi
}
LOG_ERROR() { echo "[NODE-$NODE_RANK($HOSTNAME)] [ERROR] $*"; }

EXAMPLE_FAULT_TOLERANCE() {
    for arg in "$@"; do
        case "$arg" in
            --fault_tolerance.enable) echo "true"; return ;;
            --fault_tolerance.enable=*) echo "${arg#*=}"; return ;;
        esac
    done
    echo "false"
}

export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-1234}
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}

# Persistent kernel/JIT cache layout
export PRIMUS_CACHE_ROOT=${PRIMUS_CACHE_ROOT:-/workspace/cache_persist}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${PRIMUS_CACHE_ROOT}/triton}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-${PRIMUS_CACHE_ROOT}/torchinductor}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-${PRIMUS_CACHE_ROOT}/torch_extensions}
export PYTORCH_KERNEL_CACHE_PATH=${PYTORCH_KERNEL_CACHE_PATH:-${PRIMUS_CACHE_ROOT}/pytorch_kernel}
export MIOPEN_USER_DB_PATH=${MIOPEN_USER_DB_PATH:-${PRIMUS_CACHE_ROOT}/miopen_db}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_CUSTOM_CACHE_DIR:-${PRIMUS_CACHE_ROOT}/miopen_cache}
export MIOPEN_FIND_MODE=${MIOPEN_FIND_MODE:-2}
export MIOPEN_DISABLE_CACHE=${MIOPEN_DISABLE_CACHE:-0}
export AOTRITON_CACHE_DIR=${AOTRITON_CACHE_DIR:-${PRIMUS_CACHE_ROOT}/aotriton}
export AMD_COMGR_CACHE_DIR=${AMD_COMGR_CACHE_DIR:-${PRIMUS_CACHE_ROOT}/comgr}
export AMD_COMGR_CACHE=${AMD_COMGR_CACHE:-1}
export HIPBLASLT_TUNING_FILE=${HIPBLASLT_TUNING_FILE:-${PRIMUS_CACHE_ROOT}/hipblaslt_tuning.json}
export HF_HOME=${HF_HOME:-/workspace/hf_cache}
export HF_HUB_CACHE=${HF_HUB_CACHE:-${HF_HOME}/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/datasets}

mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" \
         "${TORCH_EXTENSIONS_DIR}" "${PYTORCH_KERNEL_CACHE_PATH}" \
         "${MIOPEN_USER_DB_PATH}" "${MIOPEN_CUSTOM_CACHE_DIR}" \
         "${AOTRITON_CACHE_DIR}" "${AMD_COMGR_CACHE_DIR}" \
         "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" 2>/dev/null || true

LOG_INFO_RANK0 "==========Training cluster info=========="
LOG_INFO_RANK0 "MASTER_ADDR: $MASTER_ADDR"
LOG_INFO_RANK0 "MASTER_PORT: $MASTER_PORT"
LOG_INFO_RANK0 "NNODES: $NNODES"
LOG_INFO_RANK0 "NODE_RANK: $NODE_RANK"
LOG_INFO_RANK0 "GPUS_PER_NODE: $GPUS_PER_NODE"

PRIMUS_PATH=$(realpath "$(dirname "$0")/..")
export DATA_PATH=${DATA_PATH:-"${PRIMUS_PATH}/data"}
export HF_HOME=${HF_HOME:-"${DATA_PATH}/huggingface"}

source "${PRIMUS_PATH}/runner/helpers/envs/path_utils.sh"

LOG_INFO_RANK0 "Pip installing required packages ..."
if [ "${BACKEND:-}" != "MaxText" ]; then
    pip install -r "$PRIMUS_PATH/requirements.txt"  --quiet
else
    pip install -r "$PRIMUS_PATH/requirements-jax.txt"  --quiet
fi


FAULT_TOLERANCE_VALUE=$(EXAMPLE_FAULT_TOLERANCE "$@")
LOG_INFO_RANK0 "FAULT_TOLERANCE_VALUE: $FAULT_TOLERANCE_VALUE"
if [[ "$FAULT_TOLERANCE_VALUE" == "true" ]] || [[ "$FAULT_TOLERANCE_VALUE" == "True" ]] || [[ "$FAULT_TOLERANCE_VALUE" == "1" ]]; then
    LOG_INFO_RANK0 "Installing requirements-torchft.txt ..."
    pip install -r "$PRIMUS_PATH/requirements-torchft.txt"  --quiet
fi

# -------------------- EXP Check --------------------
if [ -z "${EXP:-}" ]; then
    LOG_ERROR "EXP must be specified."
    exit 1
fi

if [ ! -f "${EXP}" ]; then
    LOG_ERROR "The specified EXP file does not exist: ${EXP}"
    exit 1
fi

TRAIN_LOG=${TRAIN_LOG:-"output/log_mp_pretrain_$(basename "$EXP" .yaml).txt"}

LOG_INFO_RANK0 "==========Training info=========="
LOG_INFO_RANK0 "EXP: $EXP"
LOG_INFO_RANK0 "BACKEND: $BACKEND"
LOG_INFO_RANK0 "TRAIN_LOG: $TRAIN_LOG"
LOG_INFO_RANK0 "PRIMUS_PATH: $PRIMUS_PATH"
LOG_INFO_RANK0 "DATA_PATH: $DATA_PATH"
LOG_INFO_RANK0 "HF_HOME: $HF_HOME"
LOG_INFO_RANK0 ""

# -------------------- NCCL and Communication Setup --------------------

HIP_VISIBLE_DEVICES=$(seq -s, 0 $((GPUS_PER_NODE - 1)))
export HIP_VISIBLE_DEVICES
ensure_rocm_ld_library_path

export NCCL_DEBUG=${NCCL_DEBUG:-}
export NCCL_CHECKS_DISABLE=1

if [ "$USING_AINIC" == "1" ]; then
    LOG_INFO_RANK0 "Using AINIC"
    export NCCL_IB_GID_INDEX=1
    export NCCL_IB_TC=${NCCL_IB_TC:-104}
    export NCCL_IB_FIFO_TC=${NCCL_IB_FIFO_TC:-192}
    export NET_OPTIONAL_RECV_COMPLETION=1
    export NCCL_IB_USE_INLINE=1
    export NCCL_GDR_FLUSH_DISABLE=1
    export NCCL_IGNORE_CPU_AFFINITY=1
    export ANP_HOME_DIR=${ANP_HOME_DIR:-"/workspace/amd-anp"}
    export RCCL_HOME_DIR=${RCCL_HOME_DIR:-"/workspace/rccl"}
    export MPI_HOME_DIR=${MPI_HOME_DIR:-"/opt/ompi"}
    export NCCL_MAX_P2P_CHANNELS=56
    export NCCL_DMABUF_ENABLE=0
    export NCCL_IB_QPS_PER_CONNECTION=1
    path_append_unique LD_LIBRARY_PATH \
        /usr/lib/x86_64-linux-gnu \
        /usr/lib/x86_64-linux-gnu/libibverbs \
        "${RCCL_HOME_DIR}/build/release" \
        "${ANP_HOME_DIR}/build" \
        "${MPI_HOME_DIR}/lib"
    if [ -n "${NCCL_NET_PLUGIN:-}" ]; then
        export NCCL_NET_PLUGIN
    elif [ -f "${ANP_HOME_DIR}/build/librccl-anp.so" ]; then
        export NCCL_NET_PLUGIN=librccl-anp.so
    elif [ -f "${ANP_HOME_DIR}/build/librccl-net.so" ]; then
        export NCCL_NET_PLUGIN=librccl-net.so
    fi
else
    export NCCL_IB_GID_INDEX=3
fi

export NCCL_CROSS_NIC=${NCCL_CROSS_NIC:-0}

if [ -z "${NCCL_IB_HCA}" ]; then
    NCCL_IB_HCA=$(bash "${PRIMUS_PATH}/examples/scripts/get_nccl_ib_hca.sh")
fi
export NCCL_IB_HCA

if [ -z "${IP_INTERFACE}" ]; then
    IP_INTERFACE=$(bash "${PRIMUS_PATH}/examples/scripts/get_ip_interface.sh")
fi
export IP_INTERFACE

export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-$IP_INTERFACE}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-$IP_INTERFACE}

LOG_INFO_RANK0 "==========NCCL and Network Settings=========="
LOG_INFO_RANK0 "NCCL_DEBUG: $NCCL_DEBUG"
LOG_INFO_RANK0 "NCCL_CHECKS_DISABLE: $NCCL_CHECKS_DISABLE"
LOG_INFO_RANK0 "NCCL_IB_GID_INDEX: $NCCL_IB_GID_INDEX"
LOG_INFO_RANK0 "NCCL_CROSS_NIC: $NCCL_CROSS_NIC"
LOG_INFO "NCCL_IB_HCA: $NCCL_IB_HCA"
LOG_INFO "NCCL_SOCKET_IFNAME: $NCCL_SOCKET_IFNAME"
LOG_INFO "GLOO_SOCKET_IFNAME: $GLOO_SOCKET_IFNAME"
LOG_INFO ""

# ----------------- AMD-specific GPU optimizations -----------------
export HSA_ENABLE_SDMA=1
export HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM:-0}
export RCCL_MSCCL_ENABLE=0
export RCCL_MSCCLPP_ENABLE=0
export RCCL_MSCCLPP_FORCE_ENABLE=0
export RCCL_MSCCLPP_THRESHOLD=$((1*1024*1024*1024))
export MSCCLPP_DISABLE_CHANNEL_CACHE=FALSE
if [ "${BACKEND:-}" != "MaxText" ]; then
    export TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=0
fi

LOG_INFO_RANK0 "==========AMD-specific GPU optimizations=========="
LOG_INFO_RANK0 "HSA_ENABLE_SDMA: $HSA_ENABLE_SDMA"
LOG_INFO_RANK0 "HSA_NO_SCRATCH_RECLAIM: $HSA_NO_SCRATCH_RECLAIM"
LOG_INFO_RANK0 "RCCL_MSCCL_ENABLE: $RCCL_MSCCL_ENABLE"
LOG_INFO_RANK0 ""

# ----------------- Performance tuning -----------------
export GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES:-2}
export ENABLE_NUMA_BINDING=${ENABLE_NUMA_BINDING:-0}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
if [ "${BACKEND:-}" != "MaxText" ]; then
    export TORCH_NCCL_HIGH_PRIORITY=${TORCH_NCCL_HIGH_PRIORITY:-1}
fi
export NCCL_PXN_DISABLE=${NCCL_PXN_DISABLE:-1}
export NCCL_P2P_NET_CHUNKSIZE=${NCCL_P2P_NET_CHUNKSIZE:-524288}
export NVTE_USE_CAST_TRANSPOSE_TRITON=${NVTE_USE_CAST_TRANSPOSE_TRITON:-1}
export NVTE_USE_OPTIMIZED_HIPIFIED_CAST_TRANSPOSE=${NVTE_USE_OPTIMIZED_HIPIFIED_CAST_TRANSPOSE:-0}
export NVTE_ROCM_ENABLE_MXFP8=1
export NVTE_CK_USES_BWD_V3=${NVTE_CK_USES_BWD_V3:-1}
export PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32=${PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32:-0}

if [ "${PRIMUS_DETERMINISTIC:-}" == "1" ]; then
    export NCCL_ALGO="Ring"
    export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
    export ROCBLAS_DEFAULT_ATOMICS_MODE=0
    export TORCH_COMPILE_DISABLE=1
fi

# nvte debug envs (honor outer env)
export NVTE_DEBUG=${NVTE_DEBUG:-0}
export NVTE_DEBUG_LEVEL=${NVTE_DEBUG_LEVEL:-0}
export NVTE_FUSED_ATTN_LOG_CONFIG=${NVTE_FUSED_ATTN_LOG_CONFIG:-0}
export PATCH_TE_FLASH_ATTN=${PATCH_TE_FLASH_ATTN:-0}

LOG_INFO_RANK0 "==========Performance tuning=========="
LOG_INFO_RANK0 "GPU_MAX_HW_QUEUES: $GPU_MAX_HW_QUEUES"
LOG_INFO_RANK0 "ENABLE_NUMA_BINDING: $ENABLE_NUMA_BINDING"
LOG_INFO_RANK0 "CUDA_DEVICE_MAX_CONNECTIONS: $CUDA_DEVICE_MAX_CONNECTIONS"
LOG_INFO_RANK0 "TORCH_NCCL_HIGH_PRIORITY: $TORCH_NCCL_HIGH_PRIORITY"
LOG_INFO_RANK0 "NCCL_PXN_DISABLE: $NCCL_PXN_DISABLE"
LOG_INFO_RANK0 ""

handle_hipblaslt_tuning() {
    local STAGE=${PRIMUS_HIPBLASLT_TUNING_STAGE:-0}
    local TUNE_LOG_PATH=${PRIMUS_PATH}/output/tune_hipblaslt/${MODEL}
    mkdir -p "$TUNE_LOG_PATH"
    case $STAGE in
        0)
            export TE_HIPBLASLT_TUNING_RUN_COUNT=${TE_HIPBLASLT_TUNING_RUN_COUNT:-10}
            export TE_HIPBLASLT_TUNING_ALGO_COUNT=${TE_HIPBLASLT_TUNING_ALGO_COUNT:-50}
            ;;
    esac
}

if [ "${PRIMUS_DETERMINISTIC:-}" != "1" ] && [ "${PRIMUS_HIPBLASLT_TUNING:-0}" = "1" ]; then
    handle_hipblaslt_tuning
else
    LOG_INFO "disable hipblaslt tuning by default to fix torch profiler issue in TE"
    export TE_HIPBLASLT_TUNING_RUN_COUNT=0
    export TE_HIPBLASLT_TUNING_ALGO_COUNT=0
fi

setup_pythonpath() {
    local site_packages
    site_packages=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
    export PYTHONPATH="${PRIMUS_PATH}:${site_packages}:${PYTHONPATH}"
}
setup_pythonpath

run_prepare_experiment() {
    PRIMUS_PATCH_ARGS_FILE=$(mktemp /tmp/primus_patch_args.XXXXXX.yaml)
    trap 'rm -f "$PRIMUS_PATCH_ARGS_FILE"' EXIT
    SCRIPT="$PRIMUS_PATH/examples/scripts/prepare_experiment.py"
    BACKEND_ARG=()
    if [[ -n "${BACKEND_PATH}" ]]; then
        BACKEND_ARG=(--backend_path "$BACKEND_PATH")
    fi
    if [[ -n "${BACKEND}" ]]; then
        BACKEND_ARG+=("--backend" "$BACKEND")
    fi
    if ! python3 "$SCRIPT" \
        --config "$EXP" \
        --data_path "$DATA_PATH" \
        --patch_args "$PRIMUS_PATCH_ARGS_FILE" \
        "${BACKEND_ARG[@]}"; then
        LOG_ERROR "$SCRIPT failed, aborting."
        exit 1
    fi
}
run_prepare_experiment

TRAIN_EXTRA_ARGS=""
TORCHRUN_EXTRA_ARGS=""

if [[ -f "$PRIMUS_PATCH_ARGS_FILE" ]]; then
    LOG_INFO_RANK0 "Loading patch args from $PRIMUS_PATCH_ARGS_FILE"
    source_yaml_args() {
        local file=$1
        local key=$2
        local collect=0
        local args=""
        while IFS= read -r line; do
            if [[ $collect -eq 0 && $line == "$key:"* ]]; then
                args="${line#*:}"
                collect=1
                continue
            fi
            if [[ $collect -eq 1 ]]; then
                if [[ $line =~ ^[[:space:]] ]]; then
                    args="${args} ${line}"
                else
                    break
                fi
            fi
        done < "$file"
        echo "$args"
    }
    TRAIN_EXTRA_ARGS=$(source_yaml_args "$PRIMUS_PATCH_ARGS_FILE" train_args)
    TORCHRUN_EXTRA_ARGS=$(source_yaml_args "$PRIMUS_PATCH_ARGS_FILE" torchrun_args)
    if [[ -n "$TRAIN_EXTRA_ARGS" ]]; then
        LOG_INFO_RANK0 "Patched TRAIN args: $TRAIN_EXTRA_ARGS"
    fi
    if [[ -n "$TORCHRUN_EXTRA_ARGS" ]]; then
        LOG_INFO_RANK0 "Patched TORCHRUN args: $TORCHRUN_EXTRA_ARGS"
    fi
fi


# -------------------- Launch Training --------------------
if [ "${BACKEND:-}" == "MaxText" ]; then
    CMD="python primus/cli/main.py train pretrain --config $EXP $TRAIN_EXTRA_ARGS $*"
else
    DISTRIBUTED_ARGS=(
        --nproc_per_node "${GPUS_PER_NODE}"
        --nnodes "${NNODES}"
        --node_rank "${NODE_RANK}"
        --master_addr "${MASTER_ADDR}"
        --master_port "${MASTER_PORT}"
    )
    if [[ "$ENABLE_NUMA_BINDING" == "1" ]]; then
        apt-get install numactl -y > /dev/null 2>&1
        NUMA_LAUNCHER="--no-python ./runner/helpers/numa_bind.sh python3"
    fi
    CMD="torchrun ${DISTRIBUTED_ARGS[*]} $TORCHRUN_EXTRA_ARGS ${NUMA_LAUNCHER} primus/cli/main.py train pretrain --config $EXP $TRAIN_EXTRA_ARGS $*"
fi
LOG_INFO "Launching distributed training with command: $CMD"

eval "$CMD" 2>&1 | tee "$TRAIN_LOG"
exit_code=${PIPESTATUS[0]}

LOG_INFO "primus launcher exited with code $exit_code"

if [[ $exit_code -ne 0 ]]; then
    if [[ $exit_code -ge 128 ]]; then
        signal=$((exit_code - 128))
        LOG_ERROR "primus launcher crashed due to signal $signal"
    else
        LOG_ERROR "primus launcher exited with code $exit_code"
    fi
fi

exit "$exit_code"
