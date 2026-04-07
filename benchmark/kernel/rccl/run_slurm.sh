#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Slurm launcher for RCCL benchmarks inside a Docker container.
#
# Usage:
#   DOCKER_IMAGE=<image> sbatch run_slurm.sh
#   DOCKER_IMAGE=<image> NNODES=2 PARTITION=my-gpu sbatch run_slurm.sh
#   DOCKER_IMAGE=rocm/primus:v26.2 NNODES=2 sbatch -N2 -w smci355-ccs-aus-n04-[25,29] -p Compute-DCPT ./run_slurm.sh
#
# Environment variables (all optional except DOCKER_IMAGE):
#   DOCKER_IMAGE        Docker image to use (required)
#   NNODES              Number of nodes [default: 1]
#   PARTITION           Slurm partition [default: unset]
#   GPUS_PER_NODE       GPUs per node [default: 8]
#   MASTER_PORT         Port for torchrun rendezvous [default: 1234]
#   EXTRA_DOCKER_ARGS   Extra arguments passed to docker run
#
###############################################################################

#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --job-name=rccl-bench

set -euo pipefail

NNODES="${NNODES:-${SLURM_NNODES:-1}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-1234}"

SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
OUTPUT_DIR="${SCRIPT_DIR}"
echo "SCRIPT_DIR: ${SCRIPT_DIR}"

if [[ -z "${DOCKER_IMAGE:-}" ]]; then
    echo "[ERROR] DOCKER_IMAGE is not set. Export it before submitting the job."
    exit 1
fi

# Keep your original behavior (may be aggressive on shared nodes)
docker stop $(docker ps -q) >/dev/null 2>&1 || true

# Build sbatch overrides from env vars
SBATCH_OVERRIDES=()
if [[ -n "${PARTITION:-}" ]]; then
    SBATCH_OVERRIDES+=(-p "$PARTITION")
fi

echo "============================================"
echo " RCCL Benchmark - Slurm + Docker launcher"
echo "============================================"
echo "  DOCKER_IMAGE  : ${DOCKER_IMAGE}"
echo "  NNODES        : ${NNODES}"
echo "  GPUS_PER_NODE : ${GPUS_PER_NODE}"
echo "  MASTER_PORT   : ${MASTER_PORT}"
echo "  OUTPUT_DIR    : ${OUTPUT_DIR}"
echo "============================================"

# Pre-pull image + stop containers on allocated nodes
srun -N "${NNODES}" \
     --exclusive \
     --export=ALL \
     --ntasks-per-node=1 \
     "${SBATCH_OVERRIDES[@]}" \
     bash -c 'docker pull "${DOCKER_IMAGE}"; docker stop $(docker ps -q) >/dev/null 2>&1 || true'

# Run workload on allocated nodes
srun -N "${NNODES}" \
     --exclusive \
     --export=ALL \
     --ntasks-per-node=1 \
     "${SBATCH_OVERRIDES[@]}" \
     bash -c '

set -euo pipefail

# ---- Resolve master address from Slurm node list ----
readarray -t NODE_ARRAY < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
MASTER_ADDR="${NODE_ARRAY[0]}"
NODE_RANK="${SLURM_NODEID}"

# ---- Build short NODE_TAG like: n05-29_n05-33 ----
short_node() {
  # input: smci355-ccs-aus-n05-29  -> output: n05-29
  local h="$1"
  local n_part id_part
  n_part="$(echo "$h" | awk -F"-" "{print \$(NF-1)}")"  # n05
  id_part="$(echo "$h" | awk -F"-" "{print \$NF}")"     # 29
  echo "${n_part}-${id_part}"
}

NODE_TAG=""
for h in "${NODE_ARRAY[@]}"; do
  NODE_TAG+="$(short_node "$h")_"
done
NODE_TAG="${NODE_TAG%_}"   # trim trailing _

if [[ "$NODE_RANK" == "0" ]]; then
    echo "========== Slurm cluster info =========="
    echo "SLURM_NODELIST : ${NODE_ARRAY[*]}"
    echo "SLURM_NNODES   : ${SLURM_NNODES}"
    echo "MASTER_ADDR    : ${MASTER_ADDR}"
    echo "NODE_RANK      : ${NODE_RANK}"
    echo "NODE_TAG       : ${NODE_TAG}"
    echo ""
fi

SCRIPT_DIR='"${SCRIPT_DIR}"'
OUTPUT_DIR='"${OUTPUT_DIR}"'
DOCKER_IMAGE='"${DOCKER_IMAGE}"'
DOCKER_LOGIN='"${DOCKER_LOGIN:-}"'
NNODES='"${NNODES}"'
GPUS_PER_NODE='"${GPUS_PER_NODE}"'
MASTER_PORT='"${MASTER_PORT}"'
EXTRA_DOCKER_ARGS='"${EXTRA_DOCKER_ARGS:-}"'

docker run --rm \
    --network=host \
    --ipc=host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --privileged --device=/dev/infiniband \
    --cap-add=SYS_PTRACE \
    --cap-add=CAP_SYS_ADMIN \
    --security-opt seccomp=unconfined \
    --group-add video \
    -v "${SCRIPT_DIR}:${SCRIPT_DIR}" \
    -v "${HOME}:${HOME}" \
    -w "${SCRIPT_DIR}" \
    -e MASTER_ADDR="${MASTER_ADDR}" \
    -e MASTER_PORT="${MASTER_PORT}" \
    -e NNODES="${NNODES}" \
    -e NODE_RANK="${NODE_RANK}" \
    -e GPUS_PER_NODE="${GPUS_PER_NODE}" \
    -e NODE_TAG="${NODE_TAG}" \
    ${EXTRA_DOCKER_ARGS} \
    "${DOCKER_IMAGE}" \
    bash -cx "
        set -euo pipefail

        cd ${SCRIPT_DIR}
        ifconfig || true
        ibv_devices || true
        rocm-smi || true

        export TORCH_NCCL_HIGH_PRIORITY=1
        export NCCL_IB_HCA=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
        export NCCL_SOCKET_IFNAME=fenic
        export NCCL_DEBUG=TRACE
        export USING_AINIC=1

        # AINIC lib paths for different docker image
        export NCCL_NET_PLUGIN=/workspace/amd-anp/build/librccl-net.so

        # Set InfiniBand GID index for NCCL communication
        export NCCL_IB_GID_INDEX=1
        export NCCL_MAX_P2P_CHANNELS=56
        export NCCL_IB_TC=104
        export NCCL_IB_FIFO_TC=192
        export NET_OPTIONAL_RECV_COMPLETION=1
        export NCCL_IB_USE_INLINE=1
        export RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=0
        export NCCL_GDR_FLUSH_DISABLE=1
        export NCCL_DMABUF_ENABLE=0
        export NCCL_IGNORE_CPU_AFFINITY=1
        export NCCL_IB_QPS_PER_CONNECTION=1
        export NCCL_IB_RETRY_CNT=20
        export NCCL_IB_TIMEOUT=300

        PRIMUS_ROOT_PATH=\"${SCRIPT_DIR}/../../..\"
        MEGATRON_PATH=\"\${PRIMUS_ROOT_PATH}/third_party/Megatron-LM\"
        export PYTHONPATH=\"\${MEGATRON_PATH}:\${PYTHONPATH:-}\"

        # Final CSV names you want (no smci prefix, include both nodes)
        FINAL_ALLREDUCE=\"${OUTPUT_DIR}/allreduce_\${NODE_TAG}.csv\"
        FINAL_ALLGATHER=\"${OUTPUT_DIR}/allgather_\${NODE_TAG}.csv\"
        FINAL_REDUCESCATTER=\"${OUTPUT_DIR}/reducescatter_\${NODE_TAG}.csv\"

        # To avoid concurrent overwrite, only rank0 writes the final CSV.
        # Other ranks write to temporary per-rank files.
        if [[ \"\${NODE_RANK}\" == \"0\" ]]; then
          ALLREDUCE_CSV=\"\${FINAL_ALLREDUCE}\"
          ALLGATHER_CSV=\"\${FINAL_ALLGATHER}\"
          REDUCESCATTER_CSV=\"\${FINAL_REDUCESCATTER}\"
        else
          ALLREDUCE_CSV=\"/tmp/allreduce_\${NODE_TAG}_rank\${NODE_RANK}.csv\"
          ALLGATHER_CSV=\"/tmp/allgather_\${NODE_TAG}_rank\${NODE_RANK}.csv\"
          REDUCESCATTER_CSV=\"/tmp/reducescatter_\${NODE_TAG}_rank\${NODE_RANK}.csv\"
        fi

        echo \"[Node \${NODE_RANK}] CSV outputs:\"
        echo \"  allreduce     -> \${ALLREDUCE_CSV}\"
        echo \"  allgather     -> \${ALLGATHER_CSV}\"
        echo \"  reducescatter -> \${REDUCESCATTER_CSV}\"

        echo \"[Node \${NODE_RANK}] Starting RCCL benchmarks...\"
        torchrun --master_addr \"\${MASTER_ADDR}\" \
                 --master_port \"\${MASTER_PORT}\" \
                 --nnodes=\"\${NNODES}\" \
                 --node_rank=\"\${NODE_RANK}\" \
                 --nproc_per_node=\"\${GPUS_PER_NODE}\" \
            ./benchmark_allreduce.py \
                --allreduce-report-csv-path \"\${ALLREDUCE_CSV}\" \
                --allgather-report-csv-path \"\${ALLGATHER_CSV}\" \
                --reducescatter-report-csv-path \"\${REDUCESCATTER_CSV}\"

        if [[ \"\${NODE_RANK}\" == \"0\" ]]; then
          echo \"[Rank0] Final CSV written:\"
          echo \"  \${FINAL_ALLREDUCE}\"
          echo \"  \${FINAL_ALLGATHER}\"
          echo \"  \${FINAL_REDUCESCATTER}\"
        fi

        echo \"[Node \${NODE_RANK}] RCCL benchmarks complete.\"
    "
'

echo " Results written to: ${OUTPUT_DIR}/"
