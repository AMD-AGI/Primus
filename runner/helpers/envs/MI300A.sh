#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# AMD MI300A GPU-specific optimizations
# MI300A is an APU with integrated CPU and GPU
#

LOG_INFO_RANK0 "Loading MI300A-specific optimizations..."

# ----------------- AMD MI300A GPU optimizations -----------------
export HSA_ENABLE_SDMA=1
export HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM:-0}

# MI300A-specific: APU optimizations
export HSA_ENABLE_INTERRUPT=1  # Enable interrupt-driven mode for APU

# RCCL settings
export RCCL_MSCCL_ENABLE=0
export RCCL_MSCCLPP_ENABLE=0
export RCCL_MSCCLPP_FORCE_ENABLE=0
export RCCL_MSCCLPP_THRESHOLD=$((1*1024*1024*1024))

export MSCCLPP_DISABLE_CHANNEL_CACHE=FALSE
export TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=0

# ----------------- Performance tuning for MI300A -----------------
export GPU_MAX_HW_QUEUES=2
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export TORCH_NCCL_HIGH_PRIORITY=1

# ----------------- Transformer Engine optimizations -----------------
export NVTE_USE_CAST_TRANSPOSE_TRITON=1
export NVTE_USE_OPTIMIZED_HIPIFIED_CAST_TRANSPOSE=0
export NVTE_CK_USES_BWD_V3=${NVTE_CK_USES_BWD_V3:-0}

export NVTE_DEBUG=0
export NVTE_DEBUG_LEVEL=0
export NVTE_FUSED_ATTN_LOG_CONFIG=0
export PATCH_TE_FLASH_ATTN=${PATCH_TE_FLASH_ATTN:-0}

# MI300A has 128GB unified memory (HBM + DDR)
export HSA_XNACK=1  # Enable XNACK for unified memory
export GPU_MAX_HEAP_SIZE=100

log_exported_vars "MI300A-specific optimizations" \
    HSA_ENABLE_SDMA HSA_NO_SCRATCH_RECLAIM HSA_ENABLE_INTERRUPT HSA_XNACK GPU_MAX_HEAP_SIZE \
    RCCL_MSCCL_ENABLE RCCL_MSCCLPP_ENABLE GPU_MAX_HW_QUEUES CUDA_DEVICE_MAX_CONNECTIONS \
    NVTE_USE_CAST_TRANSPOSE_TRITON NVTE_CK_USES_BWD_V3
