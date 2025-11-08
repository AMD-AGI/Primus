#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1. Load base environment ---
# shellcheck source=runner/helpers/envs/base_env.sh
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/base_env.sh"

# 2. Detect GPU type ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_MODEL=$(bash "${SCRIPT_DIR}/detect_gpu.sh")
GPU_CONFIG_FILE=""
LOG_INFO_RANK0 "Detected GPU model: ${GPU_MODEL}"

# 3. Load device-specific env ---
GPU_CONFIG_FILE="${SCRIPT_DIR}/${GPU_MODEL}.sh"
if [[ -f "$GPU_CONFIG_FILE" ]]; then
    LOG_INFO_RANK0 "Loading GPU configuration: $GPU_CONFIG_FILE"
    # shellcheck disable=SC1090
    source "$GPU_CONFIG_FILE"
else
    LOG_INFO_RANK0 "[WARN] GPU configuration file not found: ${GPU_CONFIG_FILE}, using base env only."
fi
