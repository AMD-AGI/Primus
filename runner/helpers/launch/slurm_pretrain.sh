#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Shared SLURM pretrain launcher for the packaged example/benchmark scripts.
#
# This is NOT a user-facing entry point: launch training with
#   bash ./runner/primus-cli slurm srun -N <N> -- container -- train pretrain --config <exp.yaml>
#
# It exists because the customer/MoE package scripts and the benchmark runners
# all need the same srun + container assembly (exclusive whole nodes, one task
# per node, dataset volume, env passthrough) on top of an `EXP` / `NNODES`
# environment contract. Keeping that in one place avoids duplicating a dozen
# srun flags in every packaged script.

if [[ "$1" == "--help" || "$1" == "-h" ]]; then
cat <<EOF
Usage: bash runner/helpers/launch/slurm_pretrain.sh [PRIMUS_ARGS...]

Launch distributed Primus pretraining on SLURM via primus-cli slurm + container.

Optional Environment Variables:
  EXP             Experiment config YAML
  NNODES          Number of nodes [default: 1]
  MASTER_PORT     Master port [default: 12345]
  LOG_DIR         Log output directory [default: ./output]
  DATA_PATH       Dataset directory [default: ./data]
  DOCKER_IMAGE    Container image
  SLURM_TIME      srun --time value
  SLURM_NODELIST  srun --nodelist value
  SLURM_PARTITION srun --partition value
  CPUS_PER_TASK   srun --cpus-per-task [default: 128]

Example:
  export DATA_PATH=/mnt/data
  export EXP=examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml
  NNODES=2 bash runner/helpers/launch/slurm_pretrain.sh
EOF
exit 0
fi

SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
PRIMUS_PATH=$(realpath "${SCRIPT_DIR}/../../..")

if [[ -z "${EXP:-}" ]]; then
    export EXP="${PRIMUS_PATH}/examples/megatron/exp_pretrain.yaml"
fi

export MASTER_PORT=${MASTER_PORT:-12345}
export NNODES=${NNODES:-1}
export LOG_DIR=${LOG_DIR:-"./output"}
LOG_FILE="${LOG_DIR}/log_slurm_pretrain_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p "$LOG_DIR"

DATA_PATH=${DATA_PATH:-"$(pwd)/data"}
export DATA_PATH
mkdir -p "$DATA_PATH"

if [[ "${BACKEND:-}" == "MaxText" ]]; then
    export DOCKER_IMAGE=${DOCKER_IMAGE:-"docker.io/rocm/jax-training:maxtext-v26.2"}
else
    export DOCKER_IMAGE=${DOCKER_IMAGE:-"docker.io/rocm/primus:v26.4"}
fi

CONTAINER_CLEAN_ARGS=()
case "${CLEAN_DOCKER_CONTAINER:-0}" in
    1 | true | True | TRUE | yes | on)
        CONTAINER_CLEAN_ARGS=(--clean)
        ;;
esac

bash "$PRIMUS_PATH/runner/primus-cli" slurm \
    srun -N "${NNODES}" \
         --exclusive \
         --export ALL \
         --ntasks-per-node=1 \
         ${SLURM_TIME:+--time="${SLURM_TIME}"} \
         ${SLURM_NODELIST:+--nodelist="${SLURM_NODELIST}"} \
         ${SLURM_PARTITION:+--partition="${SLURM_PARTITION}"} \
         --cpus-per-task="${CPUS_PER_TASK:-128}" \
    -- container \
    "${CONTAINER_CLEAN_ARGS[@]}" \
    --image "$DOCKER_IMAGE" \
    --volume "$DATA_PATH:$DATA_PATH" \
    --env DATA_PATH \
    --env EXP \
    --env BACKEND \
    --env TRAIN_LOG \
    --env PRIMUS_SKIP_PIP \
    --env USING_UEP \
    --env REBUILD_UEP \
    --env USING_AINIC \
    --env PATCH_TE_FLASH_ATTN \
    --env REBUILD_PRIMUS_TURBO \
    --env REBUILD_BNXT \
    --env TOKENIZED_DATA_PATH \
    --env TOKENIZED_TRAIN_DATA_PATH \
    --env TOKENIZED_EVAL_DATA_PATH \
    -- \
    train pretrain --config "$EXP" "$@" 2>&1 | tee "$LOG_FILE"
# PIPESTATUS[0] is primus-cli's exit code. Without it the pipeline reports
# tee's status, so a failed run would look successful to SLURM and to the
# benchmark runners that wrap this script.
exit_code=${PIPESTATUS[0]}
exit "$exit_code"
