#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# AMD MI300X GPU-specific optimizations
# Note: Common settings are in base_env.sh. This file only contains MI300X-specific overrides.
#

LOG_INFO_RANK0 "Loading MI300X-specific optimizations..."

# ----------------- MI300X-specific GPU settings -----------------
# MI300X has 192GB HBM3, disable XNACK for performance
# export HSA_XNACK=${HSA_XNACK:-0}

# Optimize memory allocation for large models
# export GPU_MAX_HEAP_SIZE=${GPU_MAX_HEAP_SIZE:-100}

# MI300X-specific memory optimizations
# Increase HSA kernarg pool size for large model workloads (12MB)
# export HSA_KERNARG_POOL_SIZE=${HSA_KERNARG_POOL_SIZE:-12582912}

# ----------------- MI300X RCCL optimizations -----------------
# MI300X works well with MSCCLPP disabled (already set in common_network.sh)
# Override here only if needed for specific MI300X workloads

# Uncomment to enable MSCCLPP for MI300X if tested and verified
# export RCCL_MSCCLPP_ENABLE=1
# export RCCL_MSCCLPP_FORCE_ENABLE=1

# log_exported_vars "MI300X-specific optimizations" \
#     HSA_XNACK GPU_MAX_HEAP_SIZE HSA_KERNARG_POOL_SIZE
