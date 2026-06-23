#!/bin/bash
###############################################################################
# DeepSeek-V4 Pro — PAPER-precision FP8 training (ue8m0 / mxfp8), CK-free.
#
# Goal: match the paper's FP8 precision map as closely as this gfx950 build
# allows, using the **TransformerEngine** path (no Composable-Kernel grouped
# GEMM), which is the only backend with mxfp8 (ue8m0 microscaling) support.
#
# Precision map (paper §4.x / techblog §9.6) — what runs in what:
#   - MoE expert weight GEMMs       : FP8 (TEGroupedMLP -> TE general_grouped_gemm)
#   - Attention QKV/O + dense proj  : FP8 (TELinear, under fp8 autocast)
#   - Attention CORE (QK^T, softmax*V): BF16  (kept high-precision in training,
#                                       same as V3 — fp8 attention is inference-only)
#   - mHC Hyper-Connection params   : FP32  (Sinkhorn stability, §9.3)
#   - Embedding / head / RMSNorm / router : BF16 (+ fp32 reductions / states)
#   - Optimizer states (Muon / AdamW): FP32
#   - Scale format                  : ue8m0  (E8M0 microscale) via fp8_recipe=mxfp8
#
# Backend = TE only, **no CK**:
#   - TURBO_USE_GROUPED_MLP=False  -> experts route to TEGroupedMLP (hipBLASLt),
#                                     NOT PrimusTurboGroupedMLP (ck_grouped_gemm).
#   - USE_TURBO_DEEPEP=False       -> Megatron MoEFlexTokenDispatcher (no turbo).
#   - projections via turbo linear (use_turbo_parallel_linear=true in the EXP
#     yaml) OR TELinear -- both are Tensile/hipBLASLt F8, NOT CK. Only the turbo
#     GROUPED-GEMM uses CK, and that is what TURBO_USE_GROUPED_MLP=False avoids.
#   - NVTE_ROCM_ENABLE_MXFP8=1     -> unlock TE's MXFP8 path on gfx950 (required).
#
# KNOWN HARD BLOCKER on this build (verified, isolated 3-line TE GEMM test):
#   TE's ROCm MXFP8 kernel asserts GEMM K % 128 == 0
#   (transformer_engine/common/gemm/rocm_gemm.hip:1529). V4-Flash has MULTIPLE
#   non-128-aligned GEMM dims (observed K=224 and K=32 so far), so ue8m0/mxfp8
#   errors out and padding a single dim does NOT fix it -- V4's shapes are
#   structurally incompatible with TE-ROCm MXFP8 here. => mxfp8 is currently
#   NOT runnable; use FP8_RECIPE=tensorwise (default-overridable below) to get
#   the SAME paper fp8 LAYOUT (all weight GEMMs fp8, no CK) with a per-tensor
#   scale instead of the ue8m0 microscale. mxfp8 is kept as the default to
#   document the paper target; flip to tensorwise to actually train.
#
# Usage:
#   ./run_deepseek_v4_pro_fp8_paper.sh                       # paper mxfp8 (ue8m0)
#   FP8_RECIPE=tensorwise ./run_deepseek_v4_pro_fp8_paper.sh # working fallback
#   PRECISION_TYPE=BF16 ./run_deepseek_v4_pro_fp8_paper.sh   # A/B vs bf16
###############################################################################
set -euo pipefail

# ---- TE mxfp8 (ue8m0) ------------------------------------------------------
export NVTE_ROCM_ENABLE_MXFP8=${NVTE_ROCM_ENABLE_MXFP8:-1}
export FP8=${FP8:-e4m3}
export FP8_RECIPE=${FP8_RECIPE:-mxfp8}            # paper ue8m0; tensorwise to fall back

# ---- CK-free: route experts/dispatcher/linear off PrimusTurbo onto TE -------
export TURBO_USE_GROUPED_MLP=${TURBO_USE_GROUPED_MLP:-False}   # -> TEGroupedMLP (no ck_grouped_gemm)
export USE_TURBO_DEEPEP=${USE_TURBO_DEEPEP:-False}             # -> Megatron dispatcher
# Reuse the FP8 EXP yaml (use_turbo_parallel_linear=true -> turbo Tensile linear,
# which is NOT CK). The only CK path (turbo grouped-GEMM) is disabled above.
export EXP=${EXP:-examples/megatron/configs/MI355X/deepseek_v4_flash-FP8-pretrain.yaml}

# ---- Single-node TE distributed init needs a real NIC (not the cluster nic) -
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}

# Everything else (model widths, Muon, recompute, launch) comes from the Pro
# Muon runner; we only override the precision/backend knobs above.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_deepseek_v4_pro_muon.sh"
