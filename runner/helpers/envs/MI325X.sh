#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# AMD MI325X GPU-specific optimizations
# Note: Common settings are in base_env.sh. This file only contains MI325X-specific overrides.
#

LOG_INFO_RANK0 "Loading MI325X-specific optimizations..."

# ----------------- MI325X-specific GPU settings -----------------
# MI325X has 256GB HBM3e (enhanced), disable XNACK for performance
# export HSA_XNACK=${HSA_XNACK:-0}

# Optimize memory allocation for larger models compared to MI300X
# export GPU_MAX_HEAP_SIZE=${GPU_MAX_HEAP_SIZE:-100}

# MI325X-specific memory optimizations
# export HSA_KERNARG_POOL_SIZE=${HSA_KERNARG_POOL_SIZE:-12582912}

# ----------------- MI325X RCCL optimizations -----------------
# MI325X may benefit from different RCCL settings
# Override common_network.sh settings if needed for MI325X

# Uncomment to enable MSCCLPP for MI325X if tested and verified
# export RCCL_MSCCLPP_ENABLE=1
# export RCCL_MSCCLPP_FORCE_ENABLE=1

# log_exported_vars "MI325X-specific optimizations" \
#     HSA_XNACK GPU_MAX_HEAP_SIZE HSA_KERNARG_POOL_SIZE
