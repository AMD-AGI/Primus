#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Detect GPU Model Script
# Uses rocm-smi to detect AMD GPU model (MI300, MI355, etc.)
#

detect_gpu_model() {
    local gpu_model="unknown"

    # Check if rocm-smi is available
    if ! command -v rocm-smi &> /dev/null; then
        echo "Error: rocm-smi not found. Is ROCm installed?" >&2
        return 1
    fi

    # Get product name from rocm-smi
    # local product_name=$(rocm-smi --showproductname 2>/dev/null | grep -i "Card series" | head -n1 | awk '{print $NF}')

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

# Execute detection
GPU_MODEL=$(detect_gpu_model)

# Output result
echo "$GPU_MODEL"

# Exit with error if detection failed
if [[ "$GPU_MODEL" == "unknown" ]]; then
    echo "Warning: Unable to detect GPU model. Using default configuration." >&2
    exit 1
fi
