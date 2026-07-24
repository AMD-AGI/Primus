#!/bin/bash
# ===========================================================================
# megatron_patch.sh — Apply Primus modifications to vendored Megatron-LM
#
# This script applies all Megatron-LM submodule changes that Primus needs
# for FLA-parity training of the hybrid linear-attention recipes:
#
#   * Gated DeltaNet (GDN)        — docs/zebra_llama/README_GDN.md
#   * Kimi Delta Attention (KDA)  — docs/zebra_llama/README_KDA.md
#
# Both architectures consume the same six patches below. All KDA-specific
# code lives in primus/ (kimi_delta_attention.py,
# kimi_delta_attention_layer.py, hybrid_mamba_mla_layer_specs.py
# ::kda_hybrid_stack_spec_no_te, and the use_fla_triton_kda /
# use_fla_kda_in_kernel_gate / use_fla_fused_norm_gated fields registered
# in patches/gdn_config_patches.py) — no additional
# megatron_patches/*.patch is required for KDA.
#
# Patch sources live in ./megatron_patches/ and are applied with `git apply`
# inside the third_party/Megatron-LM submodule.
#
#   01-mamba_model-fused-ce.patch
#       Wires FLA's FusedLinearCrossEntropyLoss / FusedCrossEntropyLoss into
#       MambaModel so we never materialize the (b*s, vocab) logits tensor.
#       Selected by env var PRIMUS_FUSED_CE (0=off, 1=fused-linear-CE [default],
#       2=fused-CE matching FLA exactly).
#
#   02-optimizer-torch-fused-adam.patch
#       Adds an opt-in path to use torch.optim.AdamW(fused=True) instead of
#       TE/Apex FusedAdam. Set PRIMUS_TORCH_OPTIM=1 to match FLA's optimizer
#       exactly for bit-level reproducibility experiments.
#
#   03-mlp-fla-swiglu.patch
#       Replaces Megatron's naive SwiGLU (silu + multiply, 2 kernels) with
#       FLA's Triton-fused swiglu (1 fwd + 1 bwd kernel). Saves ~20 ms/step.
#       Toggle: PRIMUS_FLA_SWIGLU=1 (default), 0=disable.
#
#   04-torch_norm-fla-rmsnorm.patch
#       Routes WrappedTorchNorm's RMSNorm path through fla.modules.RMSNorm
#       to match FLA's normalization kernels exactly when PRIMUS_FLA_NORM=1.
#
#   05-transformer_config-hybrid-init.patch
#       Hybrid models (GDN, Mamba) now use uniform init_method_normal for
#       the output layer (matching FLA's initializer_range), instead of
#       the depth-scaled init that's appropriate only for pure transformers.
#
#   06-pretrain_mamba-fla-data.patch
#       Adds an opt-in FLA-order dataset shim activated by
#       PRIMUS_FLA_DATA=1 + PRIMUS_FLA_CACHE_DIR=<path>; uses
#       tools/fla_order_dataset.py to feed the exact same token order as
#       FLA's HuggingFace DistributedSampler.
#
# Usage:
#   bash megatron_patch.sh           # apply all patches
#   bash megatron_patch.sh --check   # dry-run (does not modify files)
#   bash megatron_patch.sh --revert  # undo all patches
#
# Runtime toggles (no re-patching needed):
#   PRIMUS_FUSED_CE      0=off, 1=FusedLinearCE [default], 2=FusedCE-match-FLA
#   PRIMUS_FUSED_CE_CHUNKS  Number of chunks for FusedLinearCE (default 32)
#   PRIMUS_FLA_SWIGLU    1=FLA Triton SwiGLU [default], 0=Megatron native
#   PRIMUS_FLA_NORM      1=FLA fused RMSNorm, 0=torch.nn.RMSNorm [default]
#   PRIMUS_FLA_CONV      1=FLA Triton causal_conv1d, 0=Tri-Dao CUDA [default]
#   PRIMUS_TORCH_OPTIM   1=torch AdamW(fused=True), 0=TE/Apex FusedAdam
#   PRIMUS_FLA_DATA      1=use FLA-order dataset shim (also need
#                          PRIMUS_FLA_CACHE_DIR=<HF dataset cache path>)
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEGATRON_DIR="${SCRIPT_DIR}/third_party/Megatron-LM"
PATCH_DIR="${SCRIPT_DIR}/megatron_patches"

if [[ ! -d "$MEGATRON_DIR" ]]; then
    echo "ERROR: Megatron-LM directory not found: $MEGATRON_DIR"
    exit 1
fi
if [[ ! -d "$PATCH_DIR" ]]; then
    echo "ERROR: Patch directory not found: $PATCH_DIR"
    exit 1
fi

# Patches are applied in numeric order (file names start with NN-).
PATCHES=(
    "01-mamba_model-fused-ce.patch"
    "02-optimizer-torch-fused-adam.patch"
    "03-mlp-fla-swiglu.patch"
    "04-torch_norm-fla-rmsnorm.patch"
    "05-transformer_config-hybrid-init.patch"
    "06-pretrain_mamba-fla-data.patch"
)

apply_one() {
    local patch_file="$1"
    local action="$2"
    local p="${PATCH_DIR}/${patch_file}"
    if [[ ! -f "$p" ]]; then
        echo "  ! ${patch_file} — patch file missing"
        return 1
    fi

    case "$action" in
        check)
            if (cd "$MEGATRON_DIR" && git apply --check "$p") 2>/dev/null; then
                echo "  ✓ ${patch_file} — can be applied"
                return 0
            elif (cd "$MEGATRON_DIR" && git apply --check --reverse "$p") 2>/dev/null; then
                echo "  · ${patch_file} — already applied"
                return 0
            else
                echo "  ✗ ${patch_file} — cannot apply (context mismatch)"
                return 1
            fi
            ;;
        apply)
            if (cd "$MEGATRON_DIR" && git apply --check --reverse "$p") 2>/dev/null; then
                echo "  · ${patch_file} — already applied, skipped"
                return 0
            fi
            if (cd "$MEGATRON_DIR" && git apply "$p"); then
                echo "  ✓ ${patch_file} — applied"
                return 0
            else
                echo "  ✗ ${patch_file} — failed to apply"
                return 1
            fi
            ;;
        revert)
            if (cd "$MEGATRON_DIR" && git apply --check "$p") 2>/dev/null; then
                echo "  · ${patch_file} — not currently applied, skipped"
                return 0
            fi
            if (cd "$MEGATRON_DIR" && git apply --reverse "$p"); then
                echo "  ✓ ${patch_file} — reverted"
                return 0
            else
                echo "  ✗ ${patch_file} — failed to revert"
                return 1
            fi
            ;;
    esac
}

run_all() {
    local action="$1"
    local ok=0 fail=0
    local patches=("${PATCHES[@]}")

    if [[ "$action" == "revert" ]]; then
        # Reverse order for revert.
        local reversed=()
        for ((i=${#patches[@]}-1; i>=0; i--)); do
            reversed+=("${patches[i]}")
        done
        patches=("${reversed[@]}")
    fi

    for p in "${patches[@]}"; do
        if apply_one "$p" "$action"; then
            ok=$((ok + 1))
        else
            fail=$((fail + 1))
        fi
    done
    echo ""
    echo "Done: ${ok} ${action}'d, ${fail} skipped/failed."
}

MODE="${1:---apply}"
case "$MODE" in
    --apply|-a)
        echo "Applying Megatron-LM patches from ${PATCH_DIR} ..."
        run_all apply
        ;;
    --check|-c)
        echo "Dry-run check ..."
        run_all check
        ;;
    --revert|-r)
        echo "Reverting Megatron-LM patches ..."
        run_all revert
        ;;
    *)
        echo "Usage: bash megatron_patch.sh [--apply|--check|--revert]"
        exit 1
        ;;
esac
