#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# AMD MI300X GPU-specific optimizations
#

LOG_INFO_RANK0 "Loading MI300X-specific optimizations..."

# ----------------- AMD MI300X GPU optimizations -----------------
# Enable system DMA engine (SDMA) on AMD GPUs for better IO throughput
export HSA_ENABLE_SDMA=1

# Prevent scratch memory from being reclaimed to stabilize large memory usage patterns
# NOTE: Must disable scratch reclaim to avoid MoE training crash on AMD GPUs
export HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM:-0}

# Disable MSCCL (RCCL multi-connection feature) for better stability
export RCCL_MSCCL_ENABLE=0
export RCCL_MSCCLPP_ENABLE=0
export RCCL_MSCCLPP_FORCE_ENABLE=0
export RCCL_MSCCLPP_THRESHOLD=$((1*1024*1024*1024)) # default 1 GB

# MSCCLPP settings
export MSCCLPP_DISABLE_CHANNEL_CACHE=FALSE

# PyTorch tensor register allocator hook
export TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=0

# ----------------- Performance tuning for MI300X -----------------
# Limit GPU hardware queues to 2 for performance stability
export GPU_MAX_HW_QUEUES=2

# Limit max CUDA device connections to reduce PCIe traffic
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

# Prioritize NCCL communication for PyTorch for higher throughput
export TORCH_NCCL_HIGH_PRIORITY=1

# ----------------- Transformer Engine optimizations for MI300X -----------------
# Optimize nvte fp8 cast transpose
export NVTE_USE_CAST_TRANSPOSE_TRITON=1
export NVTE_USE_OPTIMIZED_HIPIFIED_CAST_TRANSPOSE=0

# Note: Disable v3 due to accuracy issues on MI300X
export NVTE_CK_USES_BWD_V3=${NVTE_CK_USES_BWD_V3:-0}

# NVTE debug envs
export NVTE_DEBUG=0 # 0, 1
export NVTE_DEBUG_LEVEL=0 # 0, 1, 2
export NVTE_FUSED_ATTN_LOG_CONFIG=0 # 0, 1
export PATCH_TE_FLASH_ATTN=${PATCH_TE_FLASH_ATTN:-0}

# MI300X has 192GB HBM3, optimize for large models
export HSA_XNACK=0  # Disable XNACK for performance
export GPU_MAX_HEAP_SIZE=100  # Percentage of GPU memory

log_exported_vars "MI300X-specific optimizations" \
    HSA_ENABLE_SDMA HSA_NO_SCRATCH_RECLAIM HSA_XNACK GPU_MAX_HEAP_SIZE \
    RCCL_MSCCL_ENABLE RCCL_MSCCLPP_ENABLE RCCL_MSCCLPP_FORCE_ENABLE RCCL_MSCCLPP_THRESHOLD \
    MSCCLPP_DISABLE_CHANNEL_CACHE TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK \
    GPU_MAX_HW_QUEUES CUDA_DEVICE_MAX_CONNECTIONS TORCH_NCCL_HIGH_PRIORITY \
    NVTE_USE_CAST_TRANSPOSE_TRITON NVTE_USE_OPTIMIZED_HIPIFIED_CAST_TRANSPOSE \
    NVTE_CK_USES_BWD_V3 NVTE_DEBUG NVTE_DEBUG_LEVEL NVTE_FUSED_ATTN_LOG_CONFIG PATCH_TE_FLASH_ATTN
