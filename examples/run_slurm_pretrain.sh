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
  export BACKEND=MaxText
  NNODES=2 bash run_slurm_pretrain.sh
EOF
exit 0
fi

export MASTER_PORT=${MASTER_PORT:-12345}
export NNODES=${NNODES:-1}

SCRIPT_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")

# -------------------- Unique Output Directory Per Run --------------------
# Extract model name from EXP config file path (e.g., deepseek_v2_lite-pretrain.yaml -> deepseek_v2_lite-pretrain)
MODEL_NAME=$(basename "${EXP:-unknown}" .yaml)
# Export TIMESTAMP so all nodes use the same value (prevents multi-node race condition)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export TIMESTAMP

# Set PRIMUS environment variables for output paths
BASE_LOG_DIR=${LOG_DIR:-"./output"}
export PRIMUS_WORKSPACE="${BASE_LOG_DIR}"
export PRIMUS_EXP_NAME="${MODEL_NAME}_${TIMESTAMP}"
export LOG_DIR="${PRIMUS_WORKSPACE}/${PRIMUS_EXP_NAME}"
# Clear work_group and user_name to simplify path: workspace/exp_name
export PRIMUS_TEAM=""
export PRIMUS_USER=""

LOG_FILE="${LOG_DIR}/log_slurm_pretrain.txt"
mkdir -p "$LOG_DIR"

srun -N "${NNODES}" \
     --exclusive \
     --export ALL \
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
          # Log TIMESTAMP on each node to verify consistency across nodes
          echo \"[Node \$SLURM_NODEID] TIMESTAMP=\${TIMESTAMP}\"
          export MASTER_ADDR=\${node_array[0]}
          export MASTER_PORT=\${MASTER_PORT}
          export NNODES=\${SLURM_NNODES}
          export NODE_RANK=\${SLURM_PROCID}
          export GPUS_PER_NODE=\${SLURM_GPUS_ON_NODE}
          bash ${SCRIPT_DIR}/run_local_pretrain.sh \"\$@\" 2>&1 | tee ${LOG_FILE}
     " bash "$@"
