#!/bin/bash
sbatch -w  GPUA7DD,GPUA817 run_benchmark_gb200.sh
joyzhang@GPUA7DD:/mnt/shared/joyzhang/Primus_Lorri/Primus/benchmark/kernel/rccl$ cat run_benchmark_b200.sh 
#!/bin/bash
###############################################################################
# Slurm launcher for NCCL benchmarks on GB200 via enroot/pyxis.
#
# Fixed from original (2026-03-03):
#  1. --container-env takes variable NAMES only (not NAME=VALUE) -- export first
#  2. NCCL_SOCKET_IFNAME=bond0 (bond0 has IP; ens110f0 is down with no IP)
#  3. --ntasks-per-node=1 not --ntasks=8 (torchrun manages GPU procs internally)
#  4. Container image: pre-imported .sqsh (NeMo 25.11.01 has aufs whiteout bug)
#  5. Removed --network=host (not a valid pyxis flag, silently ignored)
#  6. Added MELLANOX_VISIBLE_DEVICES=all (required for IB passthrough in container)
#
# Usage:
#   sbatch run_benchmark_gb200.sh
###############################################################################
#SBATCH --job-name=rccl-bench
#SBATCH --partition=b200-cirra
#SBATCH --account=dcgpu-perf
#SBATCH --qos=dcgpu-prod
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_b200:8
#SBATCH --time=04:00:00
#SBATCH --output="rccl-b200-bench-%j.out"

set -euo pipefail

GPUS_PER_NODE=8
# Pre-imported sqsh avoids enroot aufs whiteout errors on ext4 /data filesystem.
# The sqsh is pre-imported on: GPUA806, GPUA817, GPUA818, GPUA863, GPUA7DD, GPUA81E
# If running on other nodes, root must run: enroot import --output /scratch/enroot/data/nvcr+nvidia+nemo+25.11.01.sqsh docker://nvcr.io#nvidia/nemo:25.11.01
DOCKER_IMAGE="/scratch/enroot/data/nvcr+nvidia+nemo+25.11.01.sqsh"
MASTER_PORT=29500
SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
OUTPUT_DIR="${SCRIPT_DIR}"

export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)

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

# ---- Export env vars BEFORE srun ----
# pyxis --container-env only accepts variable NAMES (not NAME=VALUE).
# Variables must be exported in the host environment first.
export NCCL_MNNVL_ENABLE=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=0
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=INFO
export TORCH_NCCL_HIGH_PRIORITY=1
export GLOO_SOCKET_IFNAME=bond0
export NCCL_SOCKET_IFNAME=bond0
export MELLANOX_VISIBLE_DEVICES=all
export UCX_NET_DEVICES=all
export NCCL_IB_HCA=ibs
export PYTHONPATH="${SCRIPT_DIR}"

srun --container-image="${DOCKER_IMAGE}" \
     --container-mounts="${SCRIPT_DIR}:${SCRIPT_DIR},${HOME}:${HOME},/dev/infiniband:/dev/infiniband" \
     --container-workdir="${SCRIPT_DIR}" \
     --container-env=NCCL_MNNVL_ENABLE,TORCH_NCCL_AVOID_RECORD_STREAMS,NCCL_NVLS_ENABLE,NCCL_DEBUG,TORCH_NCCL_HIGH_PRIORITY,GLOO_SOCKET_IFNAME,NCCL_SOCKET_IFNAME,PYTHONPATH,MELLANOX_VISIBLE_DEVICES,UCX_NET_DEVICES,NCCL_IB_HCA \
     torchrun \
       --nnodes=${SLURM_JOB_NUM_NODES} \
       --nproc_per_node=${GPUS_PER_NODE} \
       --rdzv_id=${SLURM_JOB_ID} \
       --rdzv_backend=c10d \
       --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
       --rdzv_conf=timeout=300 \
       ${SCRIPT_DIR}/benchmark_allreduce.py \
         --allreduce-report-csv-path "${ALLREDUCE_CSV}" \
         --allgather-report-csv-path "${ALLGATHER_CSV}" \
         --reducescatter-report-csv-path "${REDUCESCATTER_CSV}"

echo "Benchmark complete. CSV outputs:"
echo "  ${ALLREDUCE_CSV}"
echo "  ${ALLGATHER_CSV}"
echo "  ${REDUCESCATTER_CSV}"
