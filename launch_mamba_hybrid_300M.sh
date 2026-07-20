#!/bin/bash
###############################################################################
# Launch the 75% Hybrid Mamba2 + MLA 300M pretrain (matched to the FLA
# `train_mamba2_hybrid_300M.log` schedule: 4768 iter × 1024 batch × 2048 seq
# ≈ 10B tokens on 8 GPUs).
#
# Full FLA-parity stack — same as launch_hybrid_faflag.sh (the GDN hybrid).
# An early run without these flags reproduced the iter-1 bit-perfect parity
# with FLA, then drifted +2.58 nats by iter 100 (the warm-up spike we already
# debugged for GDN).  Enabling the fusion / data flags closes that gap and
# also unlocks ~25 % more throughput.
#
#   PRIMUS_FLA_MLA_ATTN=1  → MLA core_attention → flash_attn_func directly
#                             (~30 ms/MLA block on MI300X with flash-attn 2.8.3)
#   PRIMUS_FUSED_CE=1      → FLA FusedLinearCrossEntropyLoss (chunked, big mem
#                             + speed win regardless of mixer)
#   PRIMUS_FLA_SWIGLU=1    → FLA Triton SwiGLU in MLP (~20 ms/iter)
#   PRIMUS_FLA_NORM=1      → WrappedTorchNorm → FLA Triton RMSNorm (saves
#                             ~30 ms/iter AND drops peak memory ~5 GB/rank,
#                             unblocking the 98.8 % rocm pressure we hit on
#                             the first run)
#   PRIMUS_FLA_DATA=1      → drive Megatron from FLA's DistributedSampler
#                             token order so the loss curves are directly
#                             comparable iter-by-iter (no batch-order noise)
#   PRIMUS_FLA_CACHE_DIR   → FLA's fineweb-edu cache that PRIMUS_FLA_DATA
#                             reads from
#
# (PRIMUS_FLA_CONV is intentionally NOT exported — that flag targets GDN's
# fused conv1d; Mamba2 has its own causal_conv1d path in upstream Megatron.)
#
# Run inside the rocm/primus:v26.2 container:
#
#   cd /home/vanbhati@amd.com/Primus && bash launch_mamba_hybrid_300M.sh
###############################################################################

set -euo pipefail

# NOTE (2026-05-27): These FLA flags are now ALSO declared in the EXP YAML
# (examples/megatron/configs/MI300X/zebra_llama_300M_mamba_hybrid-pretrain.yaml,
# under `overrides:` as use_fla_fused_swiglu / use_fla_fused_rmsnorm / etc.).
# The patch primus.backends.megatron.patches.fla_runtime_patches resolves
# each knob (env > YAML > default) onto args.* at phase="before_train".
# All consumers read getattr(get_args(), 'field', default) directly.
# Env vars set here win over the YAML (backward compat).
# Comment them out to let the YAML be the sole source of truth.
export PRIMUS_FLA_MLA_ATTN=1
export PRIMUS_FUSED_CE=1
export PRIMUS_FLA_SWIGLU=1
export PRIMUS_FLA_NORM=1
export PRIMUS_FLA_DATA=1
export PRIMUS_FLA_CACHE_DIR=${PRIMUS_FLA_CACHE_DIR:-/home/vanbhati@amd.com/flash-linear-attention/legacy/training/data/HuggingFaceFW/fineweb-edu/sample-10BT/train}

EXP=${EXP:-examples/megatron/configs/MI300X/zebra_llama_300M_mamba_hybrid-pretrain.yaml}
LOG=${LOG:-primus_mamba_hybrid_300M.log}

echo "==========[launch_mamba_hybrid_300M.sh]=========="
echo "PRIMUS_FLA_MLA_ATTN  = ${PRIMUS_FLA_MLA_ATTN}"
echo "PRIMUS_FUSED_CE      = ${PRIMUS_FUSED_CE}"
echo "PRIMUS_FLA_SWIGLU    = ${PRIMUS_FLA_SWIGLU}"
echo "PRIMUS_FLA_NORM      = ${PRIMUS_FLA_NORM}"
echo "PRIMUS_FLA_DATA      = ${PRIMUS_FLA_DATA}"
echo "PRIMUS_FLA_CACHE_DIR = ${PRIMUS_FLA_CACHE_DIR}"
echo "EXP                  = ${EXP}"
echo "LOG                  = ${LOG}"
echo "=================================================="

if [ ! -d "${PRIMUS_FLA_CACHE_DIR}" ]; then
    echo "WARNING: PRIMUS_FLA_CACHE_DIR=${PRIMUS_FLA_CACHE_DIR} does not exist."
    echo "         FLAOrderGPTDataset will silently fall back to Megatron's GPTDataset"
    echo "         shuffler, and the loss curve will diverge from FLA's after iter 1."
fi

EXP="${EXP}" bash examples/run_pretrain.sh 2>&1 | tee "${LOG}"
