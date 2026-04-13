#!/bin/bash
###############################################################################
# Slurm launcher for NCCL benchmarks on GB200 via enroot/pyxis.
#
# Usage:
#   sbatch -w slurm-compute-node-[1-2] run_bench_gb200.sh
###############################################################################

#SBATCH --job-name="rccl-bench"
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --partition=batch
#SBATCH --gres=gpu:4
#SBATCH --output="rccl-gb200-bench-%j.out"
#SBATCH --qos="perf"
#SBATCH --account=perf

set -euo pipefail

GPUS_PER_NODE=4
DOCKER_IMAGE="nvcr.io#nvidia/nemo:25.11.01"
MASTER_PORT=29500
SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
OUTPUT_DIR="${SCRIPT_DIR}"

MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
export MASTER_ADDR

# ---- Build NODE_TAG from Slurm node list ----
readarray -t NODE_ARRAY < <(scontrol show hostnames "$SLURM_NODELIST")
NODE_TAG=$(IFS=_; echo "${NODE_ARRAY[*]}")

echo "============================================"
echo " RCCL Benchmark - GB200 Slurm + enroot"
echo "============================================"
echo "  DOCKER_IMAGE  : ${DOCKER_IMAGE}"
echo "  NNODES        : ${SLURM_JOB_NUM_NODES}"
echo "  GPUS_PER_NODE : ${GPUS_PER_NODE}"
echo "  MASTER_ADDR   : ${MASTER_ADDR}"
echo "  MASTER_PORT   : ${MASTER_PORT}"
echo "  NODE_TAG      : ${NODE_TAG}"
echo "  OUTPUT_DIR    : ${OUTPUT_DIR}"
echo "============================================"

ALLREDUCE_CSV="${OUTPUT_DIR}/allreduce_${NODE_TAG}.csv"
ALLGATHER_CSV="${OUTPUT_DIR}/allgather_${NODE_TAG}.csv"
REDUCESCATTER_CSV="${OUTPUT_DIR}/reducescatter_${NODE_TAG}.csv"

srun --container-image="${DOCKER_IMAGE}" \
     --container-mounts="${SCRIPT_DIR}:${SCRIPT_DIR},${HOME}:${HOME}" \
     --container-workdir="${SCRIPT_DIR}" \
     --container-env=NCCL_MNNVL_ENABLE=1 \
     --container-env=TORCH_NCCL_AVOID_RECORD_STREAMS=0 \
     --container-env=NCCL_NVLS_ENABLE=0 \
     --container-env=NCCL_DEBUG=INFO \
     --container-env=TORCH_NCCL_HIGH_PRIORITY=1 \
     --container-env=GLOO_SOCKET_IFNAME=enp50s0 \
     --container-env=PYTHONPATH="${SCRIPT_DIR}" \
     --network=host \
     torchrun \
       --nnodes="${SLURM_JOB_NUM_NODES}" \
       --nproc_per_node="${GPUS_PER_NODE}" \
       --rdzv_id="${SLURM_JOB_ID}" \
       --rdzv_backend=c10d \
       --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
       "${SCRIPT_DIR}/benchmark_allreduce.py" \
         --allreduce-report-csv-path "${ALLREDUCE_CSV}" \
         --allgather-report-csv-path "${ALLGATHER_CSV}" \
         --reducescatter-report-csv-path "${REDUCESCATTER_CSV}"

echo "Benchmark complete. CSV outputs:"
echo "  ${ALLREDUCE_CSV}"
echo "  ${ALLGATHER_CSV}"
echo "  ${REDUCESCATTER_CSV}"
