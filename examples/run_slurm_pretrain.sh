#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

set -euo pipefail

export NNODES="${NNODES:-1}"

# -------------------- EXP Check --------------------
if [ -z "${EXP:-}" ]; then
    echo "[ERROR] EXP must be specified (e.g., examples/megatron/exp_pretrain.yaml)." \
         "Primus will use the configuration in EXP to train the model."
    exit 1
fi

# Ensure EXP file exists, otherwise exit with error
if [ ! -f "${EXP}" ]; then
    echo "[ERROR] The specified EXP file does not exist: ${EXP}" \
         "Primus will use the configuration in EXP to train the model."
    exit 1
fi

# -------------------- DATA_PATH Check --------------------
if [ -z "${DATA_PATH:-}" ]; then
    DATA_PATH="$(pwd)/data"
    echo "[WARNING] DATA_PATH not specified. Defaulting to: ${DATA_PATH}"

    if [ ! -d "${DATA_PATH}" ]; then
        echo "[WARNING] DATA_PATH does not exist. Creating: ${DATA_PATH}"
        mkdir -p "${DATA_PATH}"
    fi
fi

# Slurm Launch
PRIMUS_PATH=$(realpath "$(dirname "$0")/..")
export DATA_PATH

SLURM_ARGS=(--nodes="$NNODES")
if [[ -n "${RESERVATION:-}" ]]; then
    SLURM_ARGS+=(--reservation="$RESERVATION")
fi

if [[ -n "${PARTITION:-}" ]]; then
    SLURM_ARGS+=(--partition="$PARTITION")
fi

if [[ -n "${TIME:-}" ]]; then
    SLURM_ARGS+=(--time="$TIME")
fi

bash "${PRIMUS_PATH}"/bin/primus-cli slurm srun "${SLURM_ARGS[@]}" \
                -- container --mount "$DATA_PATH" \
                -- train pretrain --config "$EXP" --data_path "$DATA_PATH" "$@"
