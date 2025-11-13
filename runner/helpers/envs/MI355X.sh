#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# AMD MI355X GPU-specific optimizations
# Note: MI355X is an APU with integrated CPU and GPU, using unified memory architecture.
# Common settings are in base_env.sh. This file only contains MI355X-specific overrides.
#

LOG_INFO_RANK0 "Loading MI355X-specific optimizations..."

# ----------------- MI355X-specific GPU settings -----------------
# MI355X has 128GB unified memory (HBM + DDR)
# Enable XNACK for unified memory support (different from discrete GPUs)
# export HSA_XNACK=${HSA_XNACK:-1}

# APU-specific: Enable interrupt-driven mode for better power efficiency
# export HSA_ENABLE_INTERRUPT=${HSA_ENABLE_INTERRUPT:-1}

# Optimize memory allocation for unified memory architecture
# export GPU_MAX_HEAP_SIZE=${GPU_MAX_HEAP_SIZE:-100}

# MI355X memory pool settings
# export HSA_KERNARG_POOL_SIZE=${HSA_KERNARG_POOL_SIZE:-8388608}  # 8MB (smaller than discrete GPUs)

# ----------------- MI355X RCCL optimizations -----------------
# APU may have different interconnect characteristics
# Keep common_network.sh settings unless testing shows otherwise

# log_exported_vars "MI355X-specific optimizations" \
#     HSA_XNACK HSA_ENABLE_INTERRUPT GPU_MAX_HEAP_SIZE HSA_KERNARG_POOL_SIZE
