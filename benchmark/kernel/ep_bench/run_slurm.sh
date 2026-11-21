#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Slurm launcher for MoRI EP (dispatch/combine) microbenchmarks inside a
# Docker container, WITHOUT torchrun.
#
# Both MoRI bench scripts spawn one process per GPU internally via
# torch.multiprocessing.spawn, so we only need ONE task per node. SLURM
# directly provides node rank / nnodes / master address; we forward those as
# RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT to the internode script.
#
# Usage:
#   DOCKER_IMAGE=primus-mori-ep:latest sbatch -N 1 -p <partition> run_slurm.sh
#   DOCKER_IMAGE=primus-mori-ep:latest NNODES=4 IBDEVICES=mlx5_0 \
#       sbatch -N 4 -p <partition> -w <node-list> run_slurm.sh
#
# Environment variables (all optional except DOCKER_IMAGE):
#   DOCKER_IMAGE        Docker image to use (required). Two options:
#                         - primus-mori-ep:latest  (built from
#                           docker/Dockerfile.mori in this directory)
#                         - ep-benchmarking:latest (MAD large_ep_benchmark)
#   NNODES              Number of nodes [default: SLURM_NNODES, then 1]
#   PARTITION           Slurm partition [default: unset; prefer -p on sbatch]
#   GPUS_PER_NODE       GPUs per node [default: 8]
#   MASTER_PORT         Port for distributed rendezvous [default: 2373]
#   IBDEVICES           InfiniBand HCA(s) for rocSHMEM / MoRI
#                       [default: mlx5_0]
#   LOG_DIR             Host directory for benchmark logs
#                       [default: ${SLURM_SUBMIT_DIR}/logs]
#   EXTRA_DOCKER_ARGS   Extra arguments passed to docker run
#
###############################################################################

#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --job-name=mori-ep-bench
#SBATCH --time=04:00:00

set -euo pipefail

NNODES="${NNODES:-${SLURM_NNODES:-1}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-2373}"
IBDEVICES="${IBDEVICES:-mlx5_0}"

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" && pwd)}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
mkdir -p "${LOG_DIR}"

if [[ -z "${DOCKER_IMAGE:-}" ]]; then
    echo "[ERROR] DOCKER_IMAGE is not set. Export it before submitting the job."
    echo "        Example: DOCKER_IMAGE=primus-mori-ep:latest sbatch -N 1 -p <partition> run_slurm.sh"
    echo "        (build the slim MoRI image via docker/Dockerfile.mori in this directory)"
    exit 1
fi

# Build sbatch overrides from env vars
SBATCH_OVERRIDES=()
if [[ -n "${PARTITION:-}" ]]; then
    SBATCH_OVERRIDES+=(-p "$PARTITION")
fi

echo "================================================"
echo " MoRI EP-Bench - Slurm + Docker launcher"
echo "================================================"
echo "  DOCKER_IMAGE  : ${DOCKER_IMAGE}"
echo "  NNODES        : ${NNODES}"
echo "  GPUS_PER_NODE : ${GPUS_PER_NODE}"
echo "  MASTER_PORT   : ${MASTER_PORT}"
echo "  IBDEVICES     : ${IBDEVICES}"
echo "  SCRIPT_DIR    : ${SCRIPT_DIR}"
echo "  LOG_DIR       : ${LOG_DIR}"
echo "================================================"

# Pre-pull image + stop any stale containers on allocated nodes
# shellcheck disable=SC2016
srun -N "${NNODES}" \
     --exclusive \
     --export=ALL \
     --ntasks-per-node=1 \
     "${SBATCH_OVERRIDES[@]}" \
     bash -c 'docker pull "${DOCKER_IMAGE}"; docker ps -q --filter "name=mori-ep-bench" | xargs -r docker stop >/dev/null 2>&1 || true; docker ps -aq --filter "name=mori-ep-bench" | xargs -r docker rm >/dev/null 2>&1 || true'

# Run workload on allocated nodes
# shellcheck disable=SC2016
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

if [[ "$NODE_RANK" == "0" ]]; then
    echo "========== Slurm cluster info =========="
    echo "SLURM_NODELIST : ${NODE_ARRAY[*]}"
    echo "SLURM_NNODES   : ${SLURM_NNODES}"
    echo "MASTER_ADDR    : ${MASTER_ADDR}"
    echo "NODE_RANK      : ${NODE_RANK}"
    echo ""
fi

SCRIPT_DIR='"${SCRIPT_DIR}"'
LOG_DIR='"${LOG_DIR}"'
DOCKER_IMAGE='"${DOCKER_IMAGE}"'
NNODES='"${NNODES}"'
GPUS_PER_NODE='"${GPUS_PER_NODE}"'
MASTER_PORT='"${MASTER_PORT}"'
IBDEVICES='"${IBDEVICES}"'
EXTRA_DOCKER_ARGS='"${EXTRA_DOCKER_ARGS:-}"'

CONTAINER_NAME="mori-ep-bench-${SLURM_JOB_ID:-manual}-${NODE_RANK}"

docker run --rm \
    --name "${CONTAINER_NAME}" \
    --network=host \
    --ipc=host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --device=/dev/infiniband \
    --privileged \
    --cap-add=SYS_PTRACE \
    --cap-add=CAP_SYS_ADMIN \
    --security-opt seccomp=unconfined \
    --group-add video \
    --ulimit memlock=-1:-1 \
    --shm-size 64G \
    -v "${SCRIPT_DIR}:${SCRIPT_DIR}" \
    -v "${HOME}:${HOME}" \
    -v "${LOG_DIR}:/run_logs" \
    -v /dev/infiniband:/dev/infiniband \
    -v /sys/class/infiniband:/sys/class/infiniband:ro \
    -v /sys/class/net:/sys/class/net:ro \
    -v /sys/bus/pci:/sys/bus/pci:ro \
    -w "${SCRIPT_DIR}" \
    -e MASTER_ADDR="${MASTER_ADDR}" \
    -e MASTER_PORT="${MASTER_PORT}" \
    -e NNODES="${NNODES}" \
    -e NODE_RANK="${NODE_RANK}" \
    -e GPUS_PER_NODE="${GPUS_PER_NODE}" \
    -e IBDEVICES="${IBDEVICES}" \
    -e SLURM_JOB_ID="${SLURM_JOB_ID:-}" \
    -e SLURM_JOB_NODELIST="${SLURM_JOB_NODELIST:-}" \
    ${EXTRA_DOCKER_ARGS} \
    --entrypoint /bin/bash \
    "${DOCKER_IMAGE}" \
    "${SCRIPT_DIR}/run_mori_bench.sh"
'

echo " Results written to: ${LOG_DIR}/"
