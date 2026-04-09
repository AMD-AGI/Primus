#!/bin/bash
###############################################################################
# MLPerf Training v6.0 — MI300X runtime environment variables for FLUX.1
#
# Source: mlperf-training-6-0/flux1/nemo/config_MI300X_01x08x16.sh
#         mlperf-training-6-0/flux1/nemo/config_common.sh
#
# These are runtime tunables (TE, CK, RCCL, HSA, HipBLASLt, etc.) that cannot
# be expressed in the Primus YAML config.  Source this file before launching:
#
#   source examples/diffusion/mlperf_flux1/MI300X_env_vars.sh
#   EXP=examples/megatron_bridge/configs/MI300X/flux_12b_pretrain_mlperf_flux1.yaml \
#     bash examples/run_pretrain.sh
###############################################################################

# ── Transformer Engine / Composable Kernel tunables ──────────────────────────
export NVTE_RS_STRIDED_ATOMIC=2
export NVTE_FP8_DPA_BWD=1
export NVTE_FUSED_ATTN=1
export NVTE_FUSED_ATTN_CK=1
export NVTE_FUSED_ATTN_AOTRITON=1
export NVTE_DEBUG=0
export NVTE_DEBUG_LEVEL=0
export NVTE_USE_HIPBLASLT=1
export NVTE_USE_CAST_TRANSPOSE_TRITON=1
export NVTE_USE_OPTIMIZED_HIPIFIED_CAST_TRANSPOSE=0
export USE_TE_SWIGLU=1

export NVTE_CK_USES_BWD_V3=1
export NVTE_CK_USES_FWD_V3=1
export NVTE_CK_IS_V3_ATOMIC_FP32=0

export CK_FUSED_ATTN_LOG_CONFIG=0
export NVTE_LOG_CK_CONFIG=0
export NVTE_LOG_FUSED_ATTN_CONFIG=0
export CHECK_FOR_NAN_IN_GRAD=0

export FUSED_SOFTMAX=0
export RMSNORM_CAST=0
export ENABLE_TRANSPOSE_CACHE=0

# ── HipBLASLt ────────────────────────────────────────────────────────────────
export USE_HIPBLASLT=1
export TORCH_BLAS_PREFER_HIPBLASLT=1

# ── HSA / ROCm ───────────────────────────────────────────────────────────────
export HSA_NO_SCRATCH_RECLAIM=1

# ── CUDA graph disabled for MI300X FLUX ──────────────────────────────────────
export USE_CUDA_GRAPH=False

# Walltime
export WALLTIME_RUNANDTIME=200
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
