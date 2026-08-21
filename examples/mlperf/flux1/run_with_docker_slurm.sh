#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

: "${SLURM_JOB_ID:?Run inside a Slurm/Spur allocation}"
: "${SLURM_NNODES:?Missing SLURM_NNODES}"
: "${DATA_ROOT:?Set DATA_ROOT}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"

export NNODES=${NNODES:-$SLURM_NNODES}
if [[ -z "${MASTER_ADDR:-}" ]]; then
    if command -v scontrol >/dev/null; then
        MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
    else
        MASTER_ADDR=${SPUR_PEER_NODES:?Set MASTER_ADDR}
        MASTER_ADDR=${MASTER_ADDR%%,*}
        MASTER_ADDR=${MASTER_ADDR%%:*}
    fi
fi
export MASTER_ADDR MASTER_PORT=${MASTER_PORT:-29500}

srun_args=(--nodes="$NNODES" --ntasks="$NNODES" --ntasks-per-node=1)
command -v spur >/dev/null && srun_args+=(--jobid="$SLURM_JOB_ID" --overlap)

exec srun "${srun_args[@]}" bash "$SCRIPT_DIR/run_with_docker.sh"
