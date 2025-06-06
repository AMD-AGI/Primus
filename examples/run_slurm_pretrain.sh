#!/bin/bash
# shellcheck disable=SC2086
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#################################################################################

if [[ "$1" == "--help" || "$1" == "-h" ]]; then
cat <<EOF
Usage: run_slurm_pretrain.sh

Launches a Primus distributed pretraining task on a Slurm cluster using Docker.

Requirements:
  - Slurm job scheduler with 'srun'
  - Docker or Podman runtime (for container execution)

Optional Environment Variables:
  NUM_NODES       Number of nodes to use [default: 1]
  MASTER_PORT     Master port [default: 12345]
  LOG_DIR         Directory for log output [default: ./output]

Example:
  export DATA_PATH=/mnt/data
  export EXP=examples/megatron/exp_pretrain.yaml
  NUM_NODES=2 bash run_slurm_pretrain.sh
EOF
exit 0
fi

export MASTER_PORT=${MASTER_PORT:-12345}
export NUM_NODES=${NUM_NODES:-1}

SCRIPT_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")

export LOG_DIR=${LOG_DIR:-"./output"}
LOG_FILE="${LOG_DIR}/log_slurm_pretrain.txt"
mkdir -p "$LOG_DIR"

srun -N ${NUM_NODES} \
     --exclusive \
     --ntasks-per-node=1 \
     --cpus-per-task=256 \
     --reservation=gpu-40_gpu-41_gpu-43_gpu-44_gpu-46_gpu-47_gpu-50_gpu-55_reservation \
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
          bash ${SCRIPT_DIR}/run_local_pretrain.sh 2>&1 | tee ${LOG_FILE}
     "
