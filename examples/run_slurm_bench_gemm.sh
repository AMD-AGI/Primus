#!/bin/bash

set -euo pipefail

export NNODES="${NNODES:-1}"

SLURM_ARGS=(--nodes="$NNODES")

if [[ -n "${RESERVATION:-}" ]]; then
    SLURM_ARGS+=(--reservation="$RESERVATION")
fi

if [[ -n "${PARTITION:-}" ]]; then
    SLURM_ARGS+=(--partition="$PARTITION")
fi

PRIMUS_PATH=$(realpath "$(dirname "$0")/..")
bash "${PRIMUS_PATH}"/bin/primus-cli slurm srun "${SLURM_ARGS[@]}" -- benchmark gemm "$@"
