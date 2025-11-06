#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1. Load base environment ---
# shellcheck source=runner/helpers/envs/base_env.sh
source "${SCRIPT_DIR}/base_env.sh"

# 2. Detect GPU type ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_MODEL=$(bash "${SCRIPT_DIR}/detect_gpu.sh")
GPU_CONFIG_FILE=""
LOG_INFO_RANK0 "Detected GPU model: ${GPU_MODEL}"

# 3. Load device-specific env ---
case "$GPU_MODEL" in
    *MI300*)
        GPU_CONFIG_FILE="${SCRIPT_DIR}/MI300X.sh"
        ;;
    *MI325*)
        GPU_CONFIG_FILE="${SCRIPT_DIR}/MI324X.sh"
        ;;
    *MI355*)
        GPU_CONFIG_FILE="${SCRIPT_DIR}/MI355X.sh"
        ;;
    # *MI450*)
    #     source "${SCRIPT_DIR}/MI450.sh"
    #     ;;
    *)
        LOG_INFO_RANK0 "[WARN] Unknown GPU model: ${GPU_MODEL}, using base env only."
        ;;
esac

# 4. Source the GPU-specific configuration
if [[ -f "$GPU_CONFIG_FILE" ]]; then
    LOG_INFO_RANK0 "Loading GPU configuration: $GPU_CONFIG_FILE"
    # shellcheck disable=SC1090
    source "$GPU_CONFIG_FILE"
else
    LOG_ERROR "GPU configuration file not found: $GPU_CONFIG_FILE"
    LOG_ERROR "Please create $ENV_CONFIG_DIR/default.sh or specify GPU_MODEL manually"
    exit 1
fi
