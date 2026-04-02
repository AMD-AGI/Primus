#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# =============================================================================
# NVIDIA GPU-specific environment overrides
# Loaded by primus-env.sh when nvidia-smi is detected (no rocm-smi present).
# Overrides AMD/ROCm env vars set in base_env.sh.
# =============================================================================

# Use CUDA device numbering; clear AMD counterpart
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPUS_PER_NODE - 1)))
unset HIP_VISIBLE_DEVICES

# Unset AMD/ROCm-specific variables that are not applicable on NVIDIA
unset HSA_ENABLE_SDMA
unset HSA_NO_SCRATCH_RECLAIM
unset GPU_MAX_HW_QUEUES
unset NVTE_ROCM_ENABLE_MXFP8
unset PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32
unset NVTE_CK_USES_BWD_V3
unset NVTE_CK_USES_FWD_V3
unset NVTE_CK_IS_V3_ATOMIC_FP32
unset NVTE_CK_HOW_V3_BF16_CVT
unset NVTE_FUSED_ATTN_CK
unset NVTE_FUSED_ATTN_AOTRITON
unset HIP_FORCE_DEV_KERNARG
unset HSA_FORCE_FINE_GRAIN_PCIE

# NVIDIA performance settings
export CUDA_DEVICE_MAX_CONNECTIONS=1

# NCCL: use loopback for single-node bootstrap (override if multi-node)
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo}

LOG_INFO_RANK0 "NVIDIA GPU environment configured (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
