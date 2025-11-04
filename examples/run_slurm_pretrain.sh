#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

set -euo pipefail

PRIMUS_PATH=$(realpath "$(dirname "$0")/..")

export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-1234}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}

export NNODES="${NNODES:-1}"
export EXP="${EXP:-examples/megatron/exp_pretrain.yaml}"
export DATA_PATH="${DATA_PATH:-./data}"

SLURM_ARGS=(--nodes="$NNODES")

if [[ -n "${RESERVATION:-}" ]]; then
    SLURM_ARGS+=(--reservation="$RESERVATION")
fi

if [[ -n "${TIME:-}" ]]; then
    SLURM_ARGS+=(--time="$TIME")
fi

bash "${PRIMUS_PATH}"/runner/primus-cli slurm srun "${SLURM_ARGS[@]}" \
                -- container --mount "$DATA_PATH" \
                -- --env MASTER_ADDR="$MASTER_ADDR" \
                   --env MASTER_PORT="$MASTER_PORT" \
                   --env GPUS_PER_NODE="$GPUS_PER_NODE" \
                -- train pretrain --config "$EXP" --data_path "$DATA_PATH" "$@"
