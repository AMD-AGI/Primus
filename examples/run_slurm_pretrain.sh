#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

if [[ "$1" == "--help" || "$1" == "-h" ]]; then
cat <<EOF
Usage: run_slurm_pretrain.sh

Launches a Primus distributed pretraining task on a Slurm cluster using Docker.

Requirements:
  - Slurm job scheduler with 'srun'
  - Docker or Podman runtime (for container execution)

Optional Environment Variables:
  NNODES          Number of nodes to use [default: 1]
  MASTER_PORT     Master port [default: 12345]
  LOG_DIR         Directory for log output [default: ./output]

Example:
  export DATA_PATH=/mnt/data
  export EXP=examples/megatron/exp_pretrain.yaml
  NNODES=2 bash run_slurm_pretrain.sh
EOF
exit 0
fi

export MASTER_PORT=${MASTER_PORT:-12345}
export NNODES=${NNODES:-1}

SCRIPT_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")

export LOG_DIR=${LOG_DIR:-"./output"}
LOG_FILE="${LOG_DIR}/log_slurm_pretrain.txt"
mkdir -p "$LOG_DIR"

srun -N "${NNODES}" \
     --exclusive \
     --ntasks-per-node=1 \
     --cpus-per-task="${CPUS_PER_TASK:-256}" \
     bash -c "
          readarray -t node_array < <(scontrol show hostnames \"\$SLURM_JOB_NODELIST\")
          if [ \"\$SLURM_NODEID\" = \"0\" ]; then
              echo \"========== Slurm cluster info ==========\"
              echo \"SLURM_NODELIST: \${node_array[*]}\"
              echo \"SLURM_NNODES: \${SLURM_NNODES}\"
              echo \"SLURM_GPUS_ON_NODE: \${SLURM_GPUS_ON_NODE}\"
              echo \"\"
          fi
          export MASTER_ADDR=\${node_array[0]}
          export MASTER_PORT=\${MASTER_PORT}
          export NNODES=\${SLURM_NNODES}
          export NODE_RANK=\${SLURM_PROCID}
          export GPUS_PER_NODE=\${SLURM_GPUS_ON_NODE}
          export HSA_NO_SCRATCH_RECLAIM=\${HSA_NO_SCRATCH_RECLAIM}
          export NVTE_CK_USES_BWD_V3=\${NVTE_CK_USES_BWD_V3}
          export NCCL_IB_HCA=\${NCCL_IB_HCA}
          export NCCL_PXN_DISABLE=\${NCCL_PXN_DISABLE}
          export NCCL_P2P_NET_CHUNKSIZE=\${NCCL_P2P_NET_CHUNKSIZE}
          export GPU_MAX_HW_QUEUES=\${GPU_MAX_HW_QUEUES}
          export GLOO_SOCKET_IFNAME=\${GLOO_SOCKET_IFNAME}
          export NCCL_SOCKET_IFNAME=\${NCCL_SOCKET_IFNAME}
          export REBUILD_BNXT=\${REBUILD_BNXT}
          export MEGATRON_PATH=\${MEGATRON_PATH}
          export TORCHTITAN_PATH=\${TORCHTITAN_PATH}
          export BACKEND_PATH=\${BACKEND_PATH}
          export PATH_TO_BNXT_TAR_PACKAGE=\${PATH_TO_BNXT_TAR_PACKAGE}
          export RCCL_HOME_DIR=\${RCCL_HOME_DIR}
          export ANP_HOME_DIR=\${ANP_HOME_DIR}
          export USING_AINIC=\${USING_AINIC}
          bash ${SCRIPT_DIR}/run_local_pretrain.sh \"\$@\" 2>&1 | tee ${LOG_FILE}
     " bash "$@"
