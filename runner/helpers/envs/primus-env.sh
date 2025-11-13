#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# =============================================================================
# Primus Environment Setup - Layered Configuration Loading
# =============================================================================
# Load order:
#   1. base_env.sh           - Base configuration (logging, cluster info, pythonpath)
#   2. common_network.sh     - Network and NCCL settings
#   3. perf_tuning.sh        - Performance tuning and optimizations
#   4. <GPU_MODEL>.sh        - GPU-specific overrides (e.g., MI300X.sh, MI325X.sh)
#
# Environment Variables:
#   PRIMUS_DEBUG=1           - Enable debug mode (set -x, verbose output)
#   PRIMUS_SKIP_VALIDATION=1 - Skip configuration validation (not recommended)
# =============================================================================

# Enable debug mode if requested
if [[ "${PRIMUS_DEBUG:-0}" == "1" ]]; then
    set -x
    echo "[DEBUG] Primus debug mode enabled"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1. Load base environment (logging, cluster info, pythonpath)
# shellcheck source=runner/helpers/envs/base_env.sh
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/base_env.sh"

LOG_INFO_RANK0 ""
LOG_INFO_RANK0 "=== Loading Primus Environment Configuration ==="

# 2. Load common network configuration
# shellcheck source=runner/helpers/envs/common_network.sh
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_network.sh"

# 3. Load performance tuning configuration
# shellcheck source=runner/helpers/envs/perf_tuning.sh
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/perf_tuning.sh"

# 4. Detect GPU model and load device-specific configuration
GPU_MODEL=$(bash "${SCRIPT_DIR}/detect_gpu.sh")
LOG_INFO_RANK0 "Detected GPU model: ${GPU_MODEL}"

GPU_CONFIG_FILE="${SCRIPT_DIR}/${GPU_MODEL}.sh"
if [[ -f "$GPU_CONFIG_FILE" ]]; then
    LOG_INFO_RANK0 "Loading GPU-specific configuration: $GPU_CONFIG_FILE"
    # shellcheck disable=SC1090
    source "$GPU_CONFIG_FILE"
else
    LOG_WARN "GPU configuration file not found: ${GPU_CONFIG_FILE}, using common settings only."
fi

# 5. Load validation library and validate configuration (unless explicitly skipped)
if [[ "${PRIMUS_SKIP_VALIDATION:-0}" != "1" ]]; then
    LOG_INFO_RANK0 ""
    LOG_INFO_RANK0 "=== Validating Configuration ==="

    # Load validation library (requires common.sh which is already loaded by base_env.sh)
    VALIDATION_LIB="${SCRIPT_DIR}/../../lib/validation.sh"
    if [[ -f "$VALIDATION_LIB" ]]; then
        # shellcheck disable=SC1090
        source "$VALIDATION_LIB"
    else
        LOG_WARN "Validation library not found: $VALIDATION_LIB"
        LOG_WARN "Skipping validation..."
    fi

    # Run validation if the function is available
    if declare -f validate_distributed_params >/dev/null 2>&1; then
        if validate_distributed_params; then
            LOG_INFO_RANK0 "✓ Configuration validation passed"
        else
            LOG_ERROR "✗ Configuration validation failed"
            LOG_ERROR "Set PRIMUS_SKIP_VALIDATION=1 to skip validation (not recommended)"
            exit 1
        fi
    else
        LOG_WARN "validate_distributed_params function not found, skipping validation"
    fi
fi

LOG_INFO_RANK0 ""
LOG_INFO_RANK0 "=== Environment Configuration Complete ==="
LOG_INFO_RANK0 ""
