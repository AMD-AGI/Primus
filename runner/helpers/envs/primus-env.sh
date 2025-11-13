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
#   1. base_env.sh    - All base configurations (cluster, network, performance, pythonpath)
#   2. <GPU_MODEL>.sh - GPU-specific overrides (e.g., MI300X.sh, MI325X.sh)
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

# 1. Load base environment (includes all configurations)
# shellcheck source=runner/helpers/envs/base_env.sh
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/base_env.sh"

LOG_INFO_RANK0 ""
LOG_INFO_RANK0 "=== Loading Primus Environment Configuration ==="

# 2. Detect GPU model and load device-specific configuration

# GPU detection function
detect_gpu_model() {
    local gpu_model
    gpu_model="unknown"

    # Check if rocm-smi is available
    if ! command -v rocm-smi &> /dev/null; then
        echo "Error: rocm-smi not found. Is ROCm installed?" >&2
        echo "unknown"
        return 1
    fi

    # Get product name from rocm-smi
    local product_name
    product_name=$(rocm-smi --showproductname 2>/dev/null | grep -i "Card series" | head -n1 | awk '{print $NF}')

    # If that doesn't work, try alternative method
    if [[ -z "$product_name" ]]; then
        product_name=$(rocm-smi --showproductname 2>/dev/null | grep -oP 'MI\d+[A-Z]*' | head -n1)
    fi

    # Extract model identifier (MI300, MI355, etc.)
    if [[ "$product_name" =~ MI([0-9]+)([A-Z]*) ]]; then
        gpu_model="MI${BASH_REMATCH[1]}${BASH_REMATCH[2]}"
    fi

    echo "$gpu_model"
}

GPU_MODEL=$(detect_gpu_model)
if [[ "$GPU_MODEL" == "unknown" ]]; then
    LOG_WARN "Unable to detect GPU model. Using default configuration."
fi
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
