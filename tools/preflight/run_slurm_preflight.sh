#!/bin/bash
# shellcheck disable=SC2086
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

export RUN_ENV=slurm
export NUM_NODES=${NUM_NODES:-2}

SCRIPT_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")

     # --nodelist=smc300x-ccs-aus-a16-10 \
     # --nodelist=gpu-[21,26,42,45] \
srun -N ${NUM_NODES} \
     --exclusive \
     --ntasks-per-node=1 \
     --cpus-per-task=128 \
     bash ${SCRIPT_DIR}/run_preflight.sh 2>&1 | tee output/log_slurm_preflight.txt
