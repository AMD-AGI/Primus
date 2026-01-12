#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# shellcheck disable=SC2086,SC2048

PRIMUS_PATH=$(realpath "$(dirname "$0")/..")

EXP=${EXP:-"examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml"}

ENV_ARGS=()
PATCH_ARGS=()

USING_AINIC=${USING_AINIC:-0}
if [ "$USING_AINIC" == "1" ]; then
    ENV_ARGS=("--env ./runner/helpers/env/env_ainic.sh")
fi

PATCH_TE_FLASH_ATTN=${PATCH_TE_FLASH_ATTN:-0}
if [[ "$PATCH_TE_FLASH_ATTN" == "1" ]]; then
    PATCH_ARGS+=("--patch ./runner/helpers/patch_te_flash_attn_max_version.sh")
fi


# Scenario 1: Use default config (Llama3.1 8B BF16)
bash "$PRIMUS_PATH/runner/primus-cli" direct \
    "${ENV_ARGS[@]}" \
    "${PATCH_ARGS[@]}" \
    -- train pretrain --config "$EXP" "$@"
