#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Slurm launcher for cluster RDMA perf tests (ib_write_bw between two nodes)
# inside a Docker container.
#
# Ported from:
#   ROCm/dist-inf-cookbook : cluster-sphere/cluster-rdma-tests/slurm_scripts/
#                            rdma_perf_tests.slurm
# but re-styled to match the Primus kernel-benchmark launchers
# (benchmark/kernel/rccl/run_slurm.sh, benchmark/kernel/ep_bench/run_slurm.sh):
#   - env-driven configuration with sensible defaults
#   - separate pre-pull and run srun stages
#   - minimal default bind-mounts; site-specific mounts go through
#     EXTRA_DOCKER_ARGS
#   - no hardcoded site-local paths (/it-share, libionic.*, etc.)
#
# Each node runs exactly one container; rank 0 hosts the ib_write_bw server,
# rank 1 connects as the client.
#
# Usage:
#   sbatch -N 2 -p <partition> run_slurm.sh
#   IBDEVICES=rdma3 LOG_PATH=/shared/logs sbatch -N 2 -p <partition> run_slurm.sh
#
# Environment variables (all optional):
#   DOCKER_IMAGE        Docker image with linux-rdma/perftest preinstalled
#                       [default: lmsysorg/sglang:v0.5.7-rocm700-mi35x]
#   NNODES              Number of nodes [default: SLURM_NNODES, then 2]
#   PARTITION           Slurm partition [default: unset; prefer -p on sbatch]
#   IBDEVICES           InfiniBand HCA passed to ib_write_bw -d
#                       [default: rdma0]
#   LOG_PATH            Host directory for benchmark logs and sbatch out/err
#                       [default: ${SLURM_SUBMIT_DIR}/logs]
#   CONTAINER_NAME      Docker container name [default: primus-rdma-tests]
#   MASTER_PORT         Reserved rendezvous port (currently unused by perftest;
#                       kept for parity with other Primus launchers)
#                       [default: 39566]
#   BARRIER_PORT        TCP port used by socket_barrier.py [default: 5000]
#   IB_WRITE_BW_PORT    Data port passed to ib_write_bw -p [default: 2000]
#   EXTRA_DOCKER_ARGS   Extra args appended to `docker run` (e.g. site-local
#                       bind mounts like /it-share, libionic.*)
#
###############################################################################

#SBATCH --job-name=primus-rdma-perf
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=primus_rdma_perf_%j.out
#SBATCH --error=primus_rdma_perf_%j.err

set -euo pipefail

DOCKER_IMAGE="${DOCKER_IMAGE:-lmsysorg/sglang:v0.5.7-rocm700-mi35x}"
NNODES="${NNODES:-${SLURM_NNODES:-2}}"
IBDEVICES="${IBDEVICES:-rdma0}"
CONTAINER_NAME="${CONTAINER_NAME:-primus-rdma-tests}"
MASTER_PORT="${MASTER_PORT:-39566}"
BARRIER_PORT="${BARRIER_PORT:-5000}"
IB_WRITE_BW_PORT="${IB_WRITE_BW_PORT:-2000}"

SCRIPT_DIR="$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" && pwd)"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}"
LOG_PATH="${LOG_PATH:-${SUBMIT_DIR}/logs}"
mkdir -p "${LOG_PATH}"

RDMA_TESTS_REPO="${SCRIPT_DIR}"
RDMA_TESTS_PATH="/root/rdma-tests-repo"

# ---------------------------------------------------------------------------
# Resolve master node + per-node IB-routable IPs from the Slurm allocation.
# We use `hostname -I` per node (same approach as upstream) because it returns
# the IP the IB stack will actually use; `getent` would give us the management
# IP which is often on a separate VLAN.
# ---------------------------------------------------------------------------
readarray -t NODE_ARRAY < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
MASTER_NODE="${NODE_ARRAY[0]}"

IPS=()
for NODE in "${NODE_ARRAY[@]}"; do
    IP="$(srun --nodes=1 --ntasks=1 -w "${NODE}" --export=ALL bash -c 'hostname -I' | awk 'NR==1{print $1}')"
    IPS+=("${IP}")
done
MASTER_ADDR="${IPS[0]}"
IPADDRS="$(IFS=,; echo "${IPS[*]}")"

# Build sbatch overrides from env vars
SBATCH_OVERRIDES=()
if [[ -n "${PARTITION:-}" ]]; then
    SBATCH_OVERRIDES+=(-p "${PARTITION}")
fi

echo "================================================"
echo " Primus Cluster RDMA Perf Tests - Slurm launcher"
echo "================================================"
echo "  SLURM_JOB_ID     : ${SLURM_JOB_ID:-N/A}"
echo "  SLURM_JOB_NODES  : ${NODE_ARRAY[*]}"
echo "  NNODES           : ${NNODES}"
echo "  DOCKER_IMAGE     : ${DOCKER_IMAGE}"
echo "  CONTAINER_NAME   : ${CONTAINER_NAME}"
echo "  IBDEVICES        : ${IBDEVICES}"
echo "  MASTER_NODE      : ${MASTER_NODE}"
echo "  MASTER_ADDR      : ${MASTER_ADDR}"
echo "  MASTER_PORT      : ${MASTER_PORT}"
echo "  BARRIER_PORT     : ${BARRIER_PORT}"
echo "  IB_WRITE_BW_PORT : ${IB_WRITE_BW_PORT}"
echo "  IPADDRS          : ${IPADDRS}"
echo "  SCRIPT_DIR       : ${SCRIPT_DIR}"
echo "  RDMA_TESTS_REPO  : ${RDMA_TESTS_REPO}"
echo "  RDMA_TESTS_PATH  : ${RDMA_TESTS_PATH}"
echo "  LOG_PATH         : ${LOG_PATH}"
echo "================================================"

export DOCKER_IMAGE NNODES IBDEVICES CONTAINER_NAME
export MASTER_ADDR MASTER_PORT BARRIER_PORT IB_WRITE_BW_PORT
export IPADDRS LOG_PATH RDMA_TESTS_REPO RDMA_TESTS_PATH
export EXTRA_DOCKER_ARGS="${EXTRA_DOCKER_ARGS:-}"

# ---------------------------------------------------------------------------
# Stage A: pre-pull image + clean up any stale container of the same name.
# ---------------------------------------------------------------------------
# shellcheck disable=SC2016
srun -N "${NNODES}" \
     --ntasks-per-node=1 \
     --export=ALL \
     "${SBATCH_OVERRIDES[@]}" \
     bash -c '
        echo "[$(hostname)] Pulling ${DOCKER_IMAGE} ...";
        docker pull "${DOCKER_IMAGE}";
        docker ps -q --filter "name=${CONTAINER_NAME}" | xargs -r docker stop >/dev/null 2>&1 || true;
        docker ps -aq --filter "name=${CONTAINER_NAME}" | xargs -r docker rm >/dev/null 2>&1 || true;
     '

# ---------------------------------------------------------------------------
# Stage B: run the ib_write_bw server (rank 0) / client (rank 1) on each node.
# ---------------------------------------------------------------------------
# shellcheck disable=SC2016
srun -N "${NNODES}" \
     --ntasks-per-node=1 \
     --export=ALL \
     "${SBATCH_OVERRIDES[@]}" \
     bash -c '
        set -euo pipefail
        NODE_RANK="${SLURM_NODEID:-${SLURM_PROCID:-0}}"
        echo "[$(hostname)] NODE_RANK=${NODE_RANK} starting container ${CONTAINER_NAME}";

        # shellcheck disable=SC2086
        docker run --rm \
            --name "${CONTAINER_NAME}" \
            --device /dev/dri \
            --device /dev/kfd \
            --device /dev/infiniband \
            --network host \
            --ipc host \
            --group-add video \
            --cap-add SYS_PTRACE \
            --privileged \
            --security-opt seccomp=unconfined \
            --ulimit memlock=-1:-1 \
            --shm-size 64G \
            -v "${RDMA_TESTS_REPO}:${RDMA_TESTS_PATH}" \
            -v "${LOG_PATH}:/run_logs" \
            -v /sys:/sys \
            -v /dev/infiniband:/dev/infiniband \
            -v /sys/class/infiniband:/sys/class/infiniband:ro \
            -v /sys/class/net:/sys/class/net:ro \
            -v /sys/bus/pci:/sys/bus/pci:ro \
            -e SLURM_JOB_ID="${SLURM_JOB_ID:-}" \
            -e SLURM_JOB_NODELIST="${SLURM_JOB_NODELIST:-}" \
            -e NODE_RANK="${NODE_RANK}" \
            -e NNODES="${NNODES}" \
            -e MASTER_ADDR="${MASTER_ADDR}" \
            -e MASTER_PORT="${MASTER_PORT}" \
            -e IPADDRS="${IPADDRS}" \
            -e IBDEVICES="${IBDEVICES}" \
            -e BARRIER_PORT="${BARRIER_PORT}" \
            -e IB_WRITE_BW_PORT="${IB_WRITE_BW_PORT}" \
            -e RDMA_TESTS_PATH="${RDMA_TESTS_PATH}" \
            ${EXTRA_DOCKER_ARGS} \
            "${DOCKER_IMAGE}" \
            bash -c "${RDMA_TESTS_PATH}/run_rdma_tests.sh"
     '

# ---------------------------------------------------------------------------
# Stage C: defensive cleanup (matches upstream tail).
# ---------------------------------------------------------------------------
# shellcheck disable=SC2016
srun -N "${NNODES}" \
     --ntasks-per-node=1 \
     --export=ALL \
     "${SBATCH_OVERRIDES[@]}" \
     bash -c '
        docker ps -q --filter "name=${CONTAINER_NAME}" | xargs -r docker stop >/dev/null 2>&1 || true;
        docker ps -aq --filter "name=${CONTAINER_NAME}" | xargs -r docker rm >/dev/null 2>&1 || true;
     '

echo " Results written to: ${LOG_PATH}/"
