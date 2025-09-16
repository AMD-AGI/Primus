#!/bin/bash
# shellcheck disable=SC2086
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# framework path
PRIMUS_PATH=$(realpath "$(dirname "$0")/../..")
export PRIMUS_PATH
export MEGATRON_PATH=${MEGATRON_PATH:-${PRIMUS_PATH}/third_party/Megatron-LM}
[[ ! -d "${MEGATRON_PATH}" || -z "$(ls -A "${MEGATRON_PATH}")" ]] && {
    echo "Error: MEGATRON_PATH (${MEGATRON_PATH}) does not exist or is empty"
    exit 1
}

# cluster envs
RUN_ENV="${RUN_ENV:-torchrun}"
if [ "$RUN_ENV" = "torchrun" ]; then
    export MASTER_ADDR=${MASTER_ADDR:-localhost}
    export MASTER_PORT=${MASTER_PORT:-1234}
    export NNODES=${NNODES:-1}
    export NODE_RANK=${NODE_RANK:-0}
    export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
elif [ "$RUN_ENV" = "slurm" ]; then
    node_list=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
    mapfile -t node_array <<<"$node_list"
    HEAD_NODE=${node_array[0]}

    export SLURM_MASTER_ADDR=$HEAD_NODE
    export SLURM_MASTER_PORT=1234
    export SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-8}
    export SLURM_WORLD_SIZE=$((SLURM_NNODES * SLURM_GPUS_ON_NODE))

    if [ "$SLURM_NODEID" = "0" ]; then
        echo "==========Slurm cluster info=========="
        echo "[SLURM-NODE-$SLURM_NODEID] NODELIST=${node_array[*]}"
        echo "[SLURM-NODE-$SLURM_NODEID] NODENAME=$SLURMD_NODENAME"
        echo "[SLURM-NODE-$SLURM_NODEID] SLURM_MASTER_ADDR=$SLURM_MASTER_ADDR"
        echo "[SLURM-NODE-$SLURM_NODEID] SLURM_MASTER_PORT=$SLURM_MASTER_PORT"
        echo "[SLURM-NODE-$SLURM_NODEID] SLURM_NNODES=$SLURM_NNODES"
        echo "[SLURM-NODE-$SLURM_NODEID] SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE"
        echo "[SLURM-NODE-$SLURM_NODEID] SLURM_WORLD_SIZE=$SLURM_WORLD_SIZE"
        echo "[SLURM-NODE-$SLURM_NODEID] SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
        echo "[SLURM-NODE-$SLURM_NODEID] SLURM_PROCID: $SLURM_PROCID"
        echo ""
    fi

    export MASTER_ADDR=${SLURM_MASTER_ADDR}
    export MASTER_PORT=${SLURM_MASTER_PORT}
    export NNODES=${SLURM_NNODES}
    export NODE_RANK=${SLURM_NODEID}
    export GPUS_PER_NODE=$((SLURM_WORLD_SIZE / SLURM_NNODES))
else
    echo "Error: Unknown RUN_ENV value: $RUN_ENV"
    exit 1
fi
gpus=$(seq -s, 0 $((GPUS_PER_NODE - 1)))
export HIP_VISIBLE_DEVICES=$gpus

if [ "$NODE_RANK" = "0" ]; then
    echo "==========Preflight experiment info=========="
    echo "[NODE-$NODE_RANK] PRIMUS_PATH: $PRIMUS_PATH"
    echo "[NODE-$NODE_RANK] MEGATRON_PATH: $MEGATRON_PATH"
    echo "[NODE-$NODE_RANK] RUN_ENV: $RUN_ENV"
    echo ""
fi

# Enable high-speed DMA transfers on AMD GPUs
export HSA_ENABLE_SDMA=1  # Enable system DMA (SDMA) engine for better GPU IO throughput

# Prevent scratch memory space from being reclaimed
export HSA_NO_SCRATCH_RECLAIM=1  # Helps stabilize large memory usage patterns (e.g. KV cache, MoE experts)

export NCCL_IB_GID_INDEX=3
export NCCL_CROSS_NIC=0
NCCL_IB_HCA=$(bash "${PRIMUS_PATH}"/examples/scripts/get_nccl_ib_hca.sh)
export NCCL_IB_HCA
export NCCL_IB_GDR_LEVEL=2
export NCCL_NET_GDR_LEVEL=2
IP_INTERFACE=$(bash "${PRIMUS_PATH}"/examples/scripts/get_ip_interface.sh)
export IP_INTERFACE
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-${IP_INTERFACE}}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-${IP_INTERFACE}}
export CUDA_DEVICE_MAX_CONNECTIONS=1 # Reducing to 1 ensures no PCIE traffic (even on single node)
export RCCL_MSCCL_ENABLE=0
export NCCL_CHECKS_DISABLE=1
export OMP_NUM_THREADS=1
export GPU_MAX_HW_QUEUES=2
export TORCH_NCCL_HIGH_PRIORITY=1
# VERSION, WARN, INFO, DEBUG
export NCCL_DEBUG=""

if [ "$NODE_RANK" = "0" ]; then
    echo "==========Preflight cluster info=========="
    echo "[NODE-$NODE_RANK] MASTER_ADDR: $MASTER_ADDR"
    echo "[NODE-$NODE_RANK] MASTER_PORT: $MASTER_PORT"
    echo "[NODE-$NODE_RANK] NCCL_IB_HCA: $NCCL_IB_HCA"
    echo "[NODE-$NODE_RANK] IP_INTERFACE: $IP_INTERFACE"
    echo "[NODE-$NODE_RANK] NNODES: $NNODES"
    echo "[NODE-$NODE_RANK] NODE_RANK: $NODE_RANK"
    echo "[NODE-$NODE_RANK] GPUS_PER_NODE: $GPUS_PER_NODE"
    echo "[NODE-$NODE_RANK] HIP_VISIBLE_DEVICES: $HIP_VISIBLE_DEVICES"
    echo ""
else
    echo "[NODE-$NODE_RANK] NCCL_IB_HCA: $NCCL_IB_HCA"
    echo "[NODE-$NODE_RANK] IP_INTERFACE: $IP_INTERFACE"
    echo "[NODE-$NODE_RANK] NODE_RANK: $NODE_RANK"
    echo "[NODE-$NODE_RANK] MASTER_PORT: $MASTER_PORT"
fi

PREFLIGHT_LOG=output/log_torchrun_preflight.txt
if [ "$NODE_RANK" = "0" ]; then
    echo "==========Preflight logging info=========="
    echo "[NODE-$NODE_RANK] PREFLIGHT_LOG: $PREFLIGHT_LOG"
    echo ""
fi


if [ "$RUN_ENV" = "torchrun" ]; then
    export PYTHONPATH=${MEGATRON_PATH}:${PRIMUS_PATH}:${PYTHONPATH}

    DISTRIBUTED_ARGS=(
        --nproc_per_node "${GPUS_PER_NODE}"
        --nnodes "${NNODES}"
        --node_rank "${NODE_RANK}"
        --master_addr "${MASTER_ADDR}"
        --master_port "${MASTER_PORT}"
    )

    pip install -qr requirements.txt && \
    apt install -y -qq libpango-1.0-0 libgdk-pixbuf2.0-0 libffi-dev libcairo2 && \
    torchrun "${DISTRIBUTED_ARGS[@]}" tools/preflight/preflight_perf_test.py \
        2>&1 | tee $PREFLIGHT_LOG

elif [ "$RUN_ENV" = "slurm" ]; then
    export DOCKER_IMAGE=${DOCKER_IMAGE:-"docker.io/rocm/megatron-lm:v25.7_py310"}

    bash "${PRIMUS_PATH}"/tools/docker/docker_podman_proxy.sh run --rm \
        --env SLURM_MASTER_ADDR=$SLURM_MASTER_ADDR \
        --env SLURM_MASTER_PORT=$SLURM_MASTER_PORT \
        --env SLURM_PROCID=$SLURM_PROCID \
        --env SLURM_WORLD_SIZE=$SLURM_WORLD_SIZE \
        --env SLURM_NODEID=$SLURM_NODEID \
        --env SLURM_NNODES=$SLURM_NNODES \
        --env MASTER_ADDR=${MASTER_ADDR} \
        --env MASTER_PORT=${MASTER_PORT} \
        --env NNODES=${NNODES} \
        --env NODE_RANK=${NODE_RANK} \
        --env GPUS_PER_NODE=${GPUS_PER_NODE} \
        --env HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES \
        --env OMP_NUM_THREADS=$OMP_NUM_THREADS \
        --env GPU_MAX_HW_QUEUES=$GPU_MAX_HW_QUEUES \
        --env TORCH_NCCL_HIGH_PRIORITY=$TORCH_NCCL_HIGH_PRIORITY \
        --env NCCL_DEBUG=$NCCL_DEBUG \
        --env NCCL_CHECKS_DISABLE=$NCCL_CHECKS_DISABLE \
        --env NCCL_IB_GDR_LEVEL=2 \
        --env NCCL_NET_GDR_LEVEL=2 \
        --env NCCL_IB_HCA=$NCCL_IB_HCA \
        --env NCCL_IB_GID_INDEX=$NCCL_IB_GID_INDEX \
        --env NCCL_CROSS_NIC=$NCCL_CROSS_NIC \
        --env HSA_ENABLE_SDMA=$HSA_ENABLE_SDMA \
        --env NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME \
        --env GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME \
        --env CUDA_DEVICE_MAX_CONNECTIONS=$CUDA_DEVICE_MAX_CONNECTIONS \
        --env RCCL_MSCCL_ENABLE=$RCCL_MSCCL_ENABLE \
        --ipc=host --network=host \
        --device=/dev/kfd --device=/dev/dri \
        --cap-add=SYS_PTRACE --cap-add=CAP_SYS_ADMIN \
        --security-opt seccomp=unconfined --group-add video \
        --privileged --device=/dev/infiniband \
        -v $MEGATRON_PATH:$MEGATRON_PATH \
        -v $PRIMUS_PATH:$PRIMUS_PATH \
        $DOCKER_IMAGE /bin/bash -c \
            "echo '[NODE-${NODE_RANK}]: begin, time=$(date +"%Y.%m.%d %H:%M:%S")' && \
            apt install -y -qq libpango-1.0-0 libgdk-pixbuf2.0-0 libffi-dev libcairo2 && \
            cd $PRIMUS_PATH && \
            pip install -qr requirements.txt && \
            PYTHONPATH=${MEGATRON_PATH}:${PRIMUS_PATH}:${PYTHONPATH} \
            torchrun \
                --nproc_per_node ${GPUS_PER_NODE} \
                --nnodes ${NNODES} \
                --node_rank ${NODE_RANK} \
                --master_addr ${MASTER_ADDR} \
                --master_port ${MASTER_PORT} \
                tools/preflight/preflight_perf_test.py \
                2>&1 | tee $PREFLIGHT_LOG && \
            echo '[NODE-${NODE_RANK}]: end time=$(date +"%Y.%m.%d %H:%M:%S")'"
        # --env NCCL_ALGO=Ring \
else
    echo "Error: Unknown RUN_ENV value: $RUN_ENV"
    exit 1
fi
