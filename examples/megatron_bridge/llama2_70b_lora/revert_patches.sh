#!/bin/bash
# Revert Megatron-Bridge / Megatron-LM patches for Llama2-70B LoRA MLPerf.
#
# Idempotent: patches that are not applied are skipped.
#
# Usage:
#   bash examples/megatron_bridge/llama2_70b_lora/revert_patches.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRIMUS_ROOT="${PRIMUS_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
MEGATRON_BRIDGE="${PRIMUS_ROOT}/third_party/Megatron-Bridge"
MEGATRON_LM="${MEGATRON_BRIDGE}/3rdparty/Megatron-LM"
PATCH_DIR="${PRIMUS_ROOT}/primus/recipes/patches"

PATCHES_REVERTED=0
PATCHES_NOT_APPLIED=0
PATCHES_FAILED=0

revert_one() {
    local patch_name="$1"
    local target_dir="$2"
    local required="${3:-1}"

    local patch_path="${PATCH_DIR}/${patch_name}"
    if [[ ! -f "${patch_path}" ]]; then
        echo "[llama2-patches][ERROR] Missing patch file: ${patch_path}" >&2
        PATCHES_FAILED=$((PATCHES_FAILED + 1))
        return 1
    fi

    cd "${target_dir}"

    if git apply --reverse --check "${patch_path}" 2>/dev/null; then
        git apply --reverse "${patch_path}"
        echo "[llama2-patches] Reverted: ${patch_name} (${target_dir})"
        PATCHES_REVERTED=$((PATCHES_REVERTED + 1))
        return 0
    fi

    echo "[llama2-patches] Not applied (skip revert): ${patch_name} (${target_dir})"
    PATCHES_NOT_APPLIED=$((PATCHES_NOT_APPLIED + 1))
    if [[ "${required}" == "0" ]]; then
        return 0
    fi
    return 0
}

git config --global --add safe.directory "${PRIMUS_ROOT}" 2>/dev/null || true
git config --global --add safe.directory "${MEGATRON_BRIDGE}" 2>/dev/null || true
git config --global --add safe.directory "${MEGATRON_LM}" 2>/dev/null || true

echo "[llama2-patches] Reverting 5 MLPerf patches under ${PATCH_DIR}"

revert_one megatron_nemo_lora_only.patch "${MEGATRON_BRIDGE}" 1
revert_one megatron_bridge_validation_consumed_samples.patch "${MEGATRON_BRIDGE}" 0
revert_one megatron_bridge_deterministic_eval.patch "${MEGATRON_BRIDGE}" 1
revert_one sft_attention_mask_cache.patch "${MEGATRON_BRIDGE}" 1
revert_one megatron_lm_mxfp4_recipe.patch "${MEGATRON_LM}" 1

if [[ "${PATCHES_FAILED}" -gt 0 ]]; then
    echo "[llama2-patches][ERROR] ${PATCHES_FAILED} patch revert(s) failed" >&2
    exit 1
fi

echo "[llama2-patches] Revert summary: ${PATCHES_REVERTED} reverted, ${PATCHES_NOT_APPLIED} not applied"
if [[ "${PATCHES_REVERTED}" -eq 0 ]]; then
    echo "[llama2-patches] No patches to revert — submodules already clean"
else
    echo "[llama2-patches] Megatron-Bridge / Megatron-LM restored after Llama2-70B LoRA MLPerf run"
fi
