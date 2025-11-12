#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Default GPU optimizations (fallback for unknown models)
#

LOG_INFO_RANK0 "Loading default GPU optimizations..."
LOG_WARN "Using default configuration. GPU model not recognized or not specified."

# ----------------- Default AMD GPU optimizations -----------------
export HSA_ENABLE_SDMA=1
export HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM:-0}

# Conservative RCCL settings
export RCCL_MSCCL_ENABLE=0
export RCCL_MSCCLPP_ENABLE=0
export RCCL_MSCCLPP_FORCE_ENABLE=0
export RCCL_MSCCLPP_THRESHOLD=$((1*1024*1024*1024))

export MSCCLPP_DISABLE_CHANNEL_CACHE=FALSE
export TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=0

# ----------------- Default performance tuning -----------------
export GPU_MAX_HW_QUEUES=2
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export TORCH_NCCL_HIGH_PRIORITY=1

# ----------------- Default Transformer Engine settings -----------------
export NVTE_USE_CAST_TRANSPOSE_TRITON=1
export NVTE_USE_OPTIMIZED_HIPIFIED_CAST_TRANSPOSE=0
export NVTE_CK_USES_BWD_V3=${NVTE_CK_USES_BWD_V3:-0}

export NVTE_DEBUG=0
export NVTE_DEBUG_LEVEL=0
export NVTE_FUSED_ATTN_LOG_CONFIG=0
export PATCH_TE_FLASH_ATTN=${PATCH_TE_FLASH_ATTN:-0}

# Conservative memory settings
export GPU_MAX_HEAP_SIZE=80

log_exported_vars "Default GPU optimizations" \
    HSA_ENABLE_SDMA HSA_NO_SCRATCH_RECLAIM GPU_MAX_HEAP_SIZE \
    GPU_MAX_HW_QUEUES CUDA_DEVICE_MAX_CONNECTIONS TORCH_NCCL_HIGH_PRIORITY
