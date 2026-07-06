#!/bin/bash
# Apply Megatron-Bridge / Megatron-LM patches for Llama2-70B LoRA MLPerf.
#
# Idempotent: already-applied patches are detected and skipped (not re-applied).
#
# Usage (from Primus repo root or inside container):
#   bash examples/megatron_bridge/llama2_70b_lora/apply_patches.sh
#
# Optional:
#   PRIMUS_ROOT=/workspace/Primus
#   RESET_BEFORE_PATCH=1   # git reset --hard submodules before applying (destructive)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRIMUS_ROOT="${PRIMUS_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
MEGATRON_BRIDGE="${PRIMUS_ROOT}/third_party/Megatron-Bridge"
MEGATRON_LM="${MEGATRON_BRIDGE}/3rdparty/Megatron-LM"
PATCH_DIR="${PRIMUS_ROOT}/primus/recipes/patches"

PATCHES_APPLIED=0
PATCHES_ALREADY=0
PATCHES_FAILED=0

apply_one() {
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
        echo "[llama2-patches] Already applied: ${patch_name} (${target_dir})"
        PATCHES_ALREADY=$((PATCHES_ALREADY + 1))
        return 0
    fi

    if git apply --check "${patch_path}" 2>/dev/null; then
        git apply "${patch_path}"
        echo "[llama2-patches] Applied: ${patch_name} (${target_dir})"
        PATCHES_APPLIED=$((PATCHES_APPLIED + 1))
        return 0
    fi

    if [[ "${required}" == "0" ]]; then
        echo "[llama2-patches][WARN] Skipped ${patch_name} (not applicable; optional patch)" >&2
        return 0
    fi

    echo "[llama2-patches][ERROR] ${patch_name} does not apply in ${target_dir} (HEAD $(git rev-parse --short HEAD))" >&2
    PATCHES_FAILED=$((PATCHES_FAILED + 1))
    return 1
}

git config --global --add safe.directory "${PRIMUS_ROOT}" 2>/dev/null || true
git config --global --add safe.directory "${MEGATRON_BRIDGE}" 2>/dev/null || true
git config --global --add safe.directory "${MEGATRON_LM}" 2>/dev/null || true

if [[ "${RESET_BEFORE_PATCH:-0}" == "1" ]]; then
    echo "[llama2-patches] RESET_BEFORE_PATCH=1: resetting Megatron-Bridge and Megatron-LM to submodule HEAD"
    git -C "${MEGATRON_BRIDGE}" reset --hard HEAD
    git -C "${MEGATRON_LM}" reset --hard HEAD
fi

echo "[llama2-patches] Checking 5 MLPerf patches under ${PATCH_DIR}"

apply_one megatron_nemo_lora_only.patch "${MEGATRON_BRIDGE}" 1
apply_one megatron_bridge_validation_consumed_samples.patch "${MEGATRON_BRIDGE}" 0
apply_one megatron_bridge_deterministic_eval.patch "${MEGATRON_BRIDGE}" 1
apply_one sft_attention_mask_cache.patch "${MEGATRON_BRIDGE}" 1
apply_one megatron_lm_mxfp4_recipe.patch "${MEGATRON_LM}" 1

if ! grep -q 'class ResettableDataIterator' "${MEGATRON_BRIDGE}/src/megatron/bridge/data/loaders.py"; then
    echo "[llama2-patches][ERROR] ResettableDataIterator not found after patching" >&2
    exit 1
fi

if [[ "${PATCHES_FAILED}" -gt 0 ]]; then
    echo "[llama2-patches][ERROR] ${PATCHES_FAILED} patch(es) failed" >&2
    exit 1
fi

echo "[llama2-patches] Summary: ${PATCHES_APPLIED} applied, ${PATCHES_ALREADY} already applied"
if [[ "${PATCHES_APPLIED}" -eq 0 && "${PATCHES_ALREADY}" -gt 0 ]]; then
    echo "[llama2-patches] All patches already applied — no changes made"
else
    echo "[llama2-patches] All 5 Llama2-70B LoRA MLPerf patches ready"
fi
