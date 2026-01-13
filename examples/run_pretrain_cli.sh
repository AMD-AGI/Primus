#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# shellcheck disable=SC2086,SC2048

PRIMUS_PATH=$(realpath "$(dirname "$0")/..")

EXP=${EXP:-"examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml"}

export PATCH_TE_FLASH_ATTN=${PATCH_TE_FLASH_ATTN:-0}
export REBUILD_PRIMUS_TURBO=${REBUILD_PRIMUS_TURBO:-0}
export REBUILD_BNXT=${REBUILD_BNXT:-0}
export USING_AINIC=${USING_AINIC:-0}

# Scenario 1: Use default config (Llama3.1 8B BF16)
bash "$PRIMUS_PATH/runner/primus-cli" direct \
    -- train pretrain --config "$EXP" "$@"
