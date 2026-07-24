#!/bin/bash
###############################################################################
# Foolproof launcher for the 75% Hybrid GDN run with the FULL FLA-parity stack
# enabled.  Run this *inside the container*:
#
#   cd /workspace/Primus && bash launch_hybrid_faflag.sh
#
# FLA-parity flags exported (matched to GDN_FLA_PARITY.md §A/B/D and the
# pure-KDA config that hit 1.46 s/iter on MI300X):
#
#   PRIMUS_FLA_MLA_ATTN=1  → MLA `core_attention` calls `flash_attn_func`
#                             directly, skipping TE's CK fallback (~30 ms/MLA
#                             block on MI300X with flash-attn 2.8.3 > 2.8.1).
#   PRIMUS_FUSED_CE=1      → FLA `FusedLinearCrossEntropyLoss` (chunked, no
#                             full logits tensor — huge memory + speed win).
#   PRIMUS_FLA_SWIGLU=1    → FLA's Triton SwiGLU instead of Megatron's naive
#                             `silu * x` (saves ~20 ms/iter).
#   PRIMUS_FLA_NORM=1      → (a) WrappedTorchNorm → FLA's Triton RMSNorm
#                             (used by GDN, MLP pre-norm, AND MLA's LoRA
#                             q_layernorm/kv_layernorm — fixes the +0.12 loss
#                             gap vs FLA), (b) GDN out_norm → FLA's
#                             FusedRMSNormGated (RMSNorm + sigmoid + mul in
#                             one Triton kernel — saves ~50 ms/iter), and
#                             (c) HybridStack fuses each GDN block's mixer-out
#                             with the next MLP's pre-MLP layernorm (~30 ms).
#   PRIMUS_FLA_CONV=1      → FLA's Triton causal_conv1d for GDN (matches FLA's
#                             default; small speed win and bit-exact gradient).
#   PRIMUS_FLA_DATA=1      → drive Megatron from FLA's `DistributedSampler`
#                             token order so the loss curves are directly
#                             comparable (no batch-order noise).
#
# Combined, on the 75% GDN+MLA hybrid these match FLA's 1.47 s/iter and
# loss-curve from iter 100 onward (iter-1 was already bit-perfect).
###############################################################################

set -euo pipefail

export PRIMUS_FLA_MLA_ATTN=1
export PRIMUS_FUSED_CE=1
export PRIMUS_FLA_SWIGLU=1
export PRIMUS_FLA_NORM=1
export PRIMUS_FLA_CONV=1
export PRIMUS_FLA_DATA=1
# PRIMUS_FLA_DATA is a no-op without PRIMUS_FLA_CACHE_DIR — the trainer's
# FLAOrderGPTDataset is guarded by `fla_data_flag == "1" and fla_cache`
# (primus/modules/trainer/megatron/trainer.py).  Point this at the FLA
# fineweb-edu cache so Primus consumes tokens in the exact same order as
# FLA's HF DistributedSampler — eliminates the dataloader-ordering bias
# that drives the +0.02 late-training loss gap and the +2.4 warm-up bump.
export PRIMUS_FLA_CACHE_DIR=${PRIMUS_FLA_CACHE_DIR:-/home/vanbhati@amd.com/flash-linear-attention/legacy/training/data/HuggingFaceFW/fineweb-edu/sample-10BT/train}

EXP=${EXP:-examples/megatron/configs/MI300X/zebra_llama_300M_gdn_hybrid-pretrain.yaml}
LOG=${LOG:-primus_gdn_hybrid_300M_faflag.log}

echo "==========[launch_hybrid_faflag.sh]=========="
echo "PRIMUS_FLA_MLA_ATTN  = ${PRIMUS_FLA_MLA_ATTN}"
echo "PRIMUS_FUSED_CE      = ${PRIMUS_FUSED_CE}"
echo "PRIMUS_FLA_SWIGLU    = ${PRIMUS_FLA_SWIGLU}"
echo "PRIMUS_FLA_NORM      = ${PRIMUS_FLA_NORM}"
echo "PRIMUS_FLA_CONV      = ${PRIMUS_FLA_CONV}"
echo "PRIMUS_FLA_DATA      = ${PRIMUS_FLA_DATA}"
echo "PRIMUS_FLA_CACHE_DIR = ${PRIMUS_FLA_CACHE_DIR}"
echo "EXP                  = ${EXP}"
echo "LOG                  = ${LOG}"
echo "============================================="

# Sanity check: warn loudly if the FLA cache dir is missing
if [ ! -d "${PRIMUS_FLA_CACHE_DIR}" ]; then
    echo "WARNING: PRIMUS_FLA_CACHE_DIR=${PRIMUS_FLA_CACHE_DIR} does not exist."
    echo "         FLAOrderGPTDataset will silently fall back to Megatron's GPTDataset"
    echo "         shuffler, and you'll see the +0.02 late-training loss gap reappear."
fi

EXP="${EXP}" bash examples/run_pretrain.sh 2>&1 | tee "${LOG}"
