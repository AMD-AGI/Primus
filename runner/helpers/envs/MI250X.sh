#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# AMD MI250X GPU-specific optimizations
#

LOG_INFO_RANK0 "Loading MI250X-specific optimizations..."

# ----------------- AMD MI250X GPU optimizations -----------------
export HSA_ENABLE_SDMA=1
export HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM:-1}  # Enable for MI250X

# RCCL settings for MI250X
export RCCL_MSCCL_ENABLE=0
export RCCL_MSCCLPP_ENABLE=0
export RCCL_MSCCLPP_FORCE_ENABLE=0
export RCCL_MSCCLPP_THRESHOLD=$((512*1024*1024))  # 512 MB for MI250X

export MSCCLPP_DISABLE_CHANNEL_CACHE=FALSE
export TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=0

# ----------------- Performance tuning for MI250X -----------------
export GPU_MAX_HW_QUEUES=2
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export TORCH_NCCL_HIGH_PRIORITY=1

# ----------------- Transformer Engine optimizations -----------------
export NVTE_USE_CAST_TRANSPOSE_TRITON=0  # Disable for MI250X
export NVTE_USE_OPTIMIZED_HIPIFIED_CAST_TRANSPOSE=1  # Use optimized version
export NVTE_CK_USES_BWD_V3=${NVTE_CK_USES_BWD_V3:-0}

export NVTE_DEBUG=0
export NVTE_DEBUG_LEVEL=0
export NVTE_FUSED_ATTN_LOG_CONFIG=0
export PATCH_TE_FLASH_ATTN=${PATCH_TE_FLASH_ATTN:-1}  # Enable patch for MI250X

# MI250X has 128GB HBM2e (64GB per GCD)
export HSA_XNACK=0
export GPU_MAX_HEAP_SIZE=90  # Conservative for MI250X

log_exported_vars "MI250X-specific optimizations" \
    HSA_ENABLE_SDMA HSA_NO_SCRATCH_RECLAIM HSA_XNACK GPU_MAX_HEAP_SIZE \
    RCCL_MSCCLPP_THRESHOLD GPU_MAX_HW_QUEUES \
    NVTE_USE_CAST_TRANSPOSE_TRITON NVTE_USE_OPTIMIZED_HIPIFIED_CAST_TRANSPOSE \
    PATCH_TE_FLASH_ATTN
