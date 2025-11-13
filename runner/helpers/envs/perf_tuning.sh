#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# =============================================================================
# Performance Tuning Configuration
# =============================================================================
# This file contains all performance-related settings including:
#   - AMD-specific GPU optimizations (HSA, RCCL)
#   - General performance tuning (GPU queues, NUMA, CUDA connections)
#   - NCCL performance settings (PXN, P2P)
#   - Transformer Engine optimizations (NVTE)
# =============================================================================

# Dependency check: ensure base_env.sh has been loaded
if ! declare -f log_exported_vars >/dev/null 2>&1; then
    echo "[ERROR] log_exported_vars function not found. base_env.sh must be loaded first." >&2
    exit 1
fi

# ----------------- AMD-specific GPU optimizations -----------------
# Enable system DMA engine (SDMA) on AMD GPUs for better IO throughput
export HSA_ENABLE_SDMA=${HSA_ENABLE_SDMA:-1}

# Prevent scratch memory from being reclaimed to stabilize large memory usage
# NOTE: Must disable scratch reclaim to avoid MoE training crash on AMD GPUs
# Setting this to 0 prevents core dumps when using Mixture-of-Experts (MoE) models
export HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM:-0}

log_exported_vars "AMD GPU Optimizations" \
    HSA_ENABLE_SDMA HSA_NO_SCRATCH_RECLAIM

# ----------------- General Performance Tuning -----------------
# Limit GPU hardware queues to 2 for performance stability
export GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES:-2}

# Increase HSA kernarg pool size to 12MB for models with many kernels (optional, can be set by GPU-specific configs)
# export HSA_KERNARG_POOL_SIZE=${HSA_KERNARG_POOL_SIZE:-12582912}

# Enable NUMA binding for better memory locality (may increase stability for large models)
export ENABLE_NUMA_BINDING=${ENABLE_NUMA_BINDING:-0}

# Limit max CUDA device connections to reduce PCIe traffic
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

# Prioritize NCCL communication for PyTorch for higher throughput
export TORCH_NCCL_HIGH_PRIORITY=${TORCH_NCCL_HIGH_PRIORITY:-1}

# ----------------- NCCL Performance Settings -----------------
# In multi-node training, PXN can be enabled to improve inter-node all-to-all
# communication efficiency, but it will increase GPU memory usage.
# Default: disable PXN for NCCL
export NCCL_PXN_DISABLE=${NCCL_PXN_DISABLE:-1}
export NCCL_P2P_NET_CHUNKSIZE=${NCCL_P2P_NET_CHUNKSIZE:-524288}

log_exported_vars "General Performance Tuning" \
    GPU_MAX_HW_QUEUES ENABLE_NUMA_BINDING CUDA_DEVICE_MAX_CONNECTIONS \
    TORCH_NCCL_HIGH_PRIORITY NCCL_PXN_DISABLE NCCL_P2P_NET_CHUNKSIZE

# ----------------- Transformer Engine Optimizations -----------------
# Optimize NVTE fp8 cast transpose
export NVTE_USE_CAST_TRANSPOSE_TRITON=${NVTE_USE_CAST_TRANSPOSE_TRITON:-1}
export NVTE_USE_OPTIMIZED_HIPIFIED_CAST_TRANSPOSE=${NVTE_USE_OPTIMIZED_HIPIFIED_CAST_TRANSPOSE:-0}

# Note: Disable v3 due to accuracy issues. Will fix after TE version 2.1.
export NVTE_CK_USES_BWD_V3=${NVTE_CK_USES_BWD_V3:-0}

# Note: Disable fp32 atomic if you find any accuracy issue
export PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32=${PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32:-0}

# NVTE debug envs
export NVTE_DEBUG=${NVTE_DEBUG:-0}              # 0, 1
export NVTE_DEBUG_LEVEL=${NVTE_DEBUG_LEVEL:-0}  # 0, 1, 2
export NVTE_FUSED_ATTN_LOG_CONFIG=${NVTE_FUSED_ATTN_LOG_CONFIG:-0}  # 0, 1
export PATCH_TE_FLASH_ATTN=${PATCH_TE_FLASH_ATTN:-0}

log_exported_vars "Transformer Engine Optimizations" \
    NVTE_USE_CAST_TRANSPOSE_TRITON NVTE_USE_OPTIMIZED_HIPIFIED_CAST_TRANSPOSE \
    NVTE_CK_USES_BWD_V3 PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32 \
    NVTE_DEBUG NVTE_DEBUG_LEVEL NVTE_FUSED_ATTN_LOG_CONFIG PATCH_TE_FLASH_ATTN
