#!/usr/bin/env bash
set -euo pipefail

# Native Primus reproduction of HyLo-Llama-4MLA12GDN-8K-SFT.
# Required inputs:
#   HYLO_LLAMA32_1B_MEGATRON_CKPT  dense 1B teacher/backbone
#   HYLO_LLAMA31_8B_MEGATRON_CKPT  dense 8B stage-2 teacher
# Optional inputs (generated below when unset):
#   HYLO_GDN_UPCYCLED_CKPT         all-GDN from-teacher initialization
#   HYLO_MLA_UPCYCLED_CKPT         all-MLA from-teacher initialization
#   HYLO_STAGE2_BASE_CKPT           target 4MLA12GDN from-teacher initialization
# Other optional settings: HYLO_SFT_DATASET, HYLO_OUTPUT_ROOT,
#   HYLO_ILD_TRAIN_ITERS, HYLO_SFT_TRAIN_ITERS, PRIMUS_IMAGE.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${PRIMUS_IMAGE:-rocm/primus:v26.8}"
OUT="${HYLO_OUTPUT_ROOT:-${ROOT}/output/hylo-native}"
mkdir -p "${OUT}"

: "${HYLO_LLAMA32_1B_MEGATRON_CKPT:?set the converted Llama-3.2-1B checkpoint}"
: "${HYLO_LLAMA31_8B_MEGATRON_CKPT:?set the converted Llama-3.1-8B checkpoint}"

export HYLO_GDN_UPCYCLED_CKPT="${HYLO_GDN_UPCYCLED_CKPT:-${OUT}/gdn-upcycled}"
export HYLO_MLA_UPCYCLED_CKPT="${HYLO_MLA_UPCYCLED_CKPT:-${OUT}/mla-upcycled}"
export HYLO_STAGE2_BASE_CKPT="${HYLO_STAGE2_BASE_CKPT:-${OUT}/4mla12gdn-upcycled}"
RUN_GROUP="${PRIMUS_TEAM:-amd}"
RUN_USER="${PRIMUS_USER:-root}"
export HYLO_GDN_ILD_OUTPUT="${OUT}/${RUN_GROUP}/${RUN_USER}/gdn-ild/checkpoints"
export HYLO_MLA_ILD_OUTPUT="${OUT}/${RUN_GROUP}/${RUN_USER}/mla-ild/checkpoints"
export HYLO_STAGE2_INIT_CKPT="${OUT}/stage2-init"
export HYLO_SFT_OUTPUT="${OUT}/${RUN_GROUP}/${RUN_USER}/4mla12gdn-8k-sft/checkpoints"

run_primus() {
  local config="$1"
  local exp_name="$2"
  "${ROOT}/primus-cli" container \
    --image "${IMAGE}" \
    --volume "${ROOT}:${ROOT}" \
    --env "PRIMUS_WORKSPACE=${OUT}" \
    --env "PRIMUS_EXP_NAME=${exp_name}" \
    --env "HYLO_LLAMA32_1B_MEGATRON_CKPT=${HYLO_LLAMA32_1B_MEGATRON_CKPT}" \
    --env "HYLO_LLAMA31_8B_MEGATRON_CKPT=${HYLO_LLAMA31_8B_MEGATRON_CKPT}" \
    --env "HYLO_GDN_UPCYCLED_CKPT=${HYLO_GDN_UPCYCLED_CKPT}" \
    --env "HYLO_MLA_UPCYCLED_CKPT=${HYLO_MLA_UPCYCLED_CKPT}" \
    --env "HYLO_STAGE2_INIT_CKPT=${HYLO_STAGE2_INIT_CKPT}" \
    --env "HYLO_SFT_DATASET=${HYLO_SFT_DATASET:-JunxiongWang/sftdatasetv3}" \
    --env "HYLO_TOKENIZER_MODEL=${HYLO_TOKENIZER_MODEL:-meta-llama/Llama-3.2-1B-Instruct}" \
    --env "HYLO_GDN_ILD_OUTPUT=${HYLO_GDN_ILD_OUTPUT}" \
    --env "HYLO_MLA_ILD_OUTPUT=${HYLO_MLA_ILD_OUTPUT}" \
    --env "HYLO_SFT_OUTPUT=${HYLO_SFT_OUTPUT}" \
    --env "HYLO_ILD_TRAIN_ITERS=${HYLO_ILD_TRAIN_ITERS:-1000}" \
    --env "HYLO_SFT_TRAIN_ITERS=${HYLO_SFT_TRAIN_ITERS:-1000}" \
    -- \
    train posttrain --config "${config}"
}

run_constructor() {
  local pattern="$1"
  local exp_name="$2"
  "${ROOT}/primus-cli" container \
    --image "${IMAGE}" \
    --volume "${ROOT}:${ROOT}" \
    --env "PRIMUS_WORKSPACE=${OUT}" \
    --env "PRIMUS_EXP_NAME=${exp_name}" \
    --env "HYLO_CONSTRUCTOR_PATTERN=${pattern}" \
    --env "HYLO_TOKENIZER_MODEL=${HYLO_TOKENIZER_MODEL:-meta-llama/Llama-3.2-1B-Instruct}" \
    -- \
    train pretrain --config \
    "${ROOT}/examples/megatron/configs/MI300X/hylo_1B-constructor.yaml"
}

upcycle() {
  local constructor="$1"
  local pattern="$2"
  local output="$3"
  docker run --rm \
    -v "${ROOT}:${ROOT}" -w "${ROOT}" \
    "${IMAGE}" \
    python tools/hybrid/upcycle_dense_to_hybrid.py \
      --dense-checkpoint "${HYLO_LLAMA32_1B_MEGATRON_CKPT}" \
      --hybrid-init-checkpoint "${constructor}" \
      --hybrid-pattern "${pattern}" \
      --output-dir "${output}"
}

GDN_PATTERN="M-M-M-M-M-M-M-M-M-M-M-M-M-M-M-M-"
MLA_PATTERN="*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-"
TARGET_PATTERN="M-*-M-M-M-*-M-M-M-M-*-M-M-M-*-M-"

if [[ ! -f "${HYLO_GDN_UPCYCLED_CKPT}/latest_checkpointed_iteration.txt" ]]; then
  run_constructor "${GDN_PATTERN}" "gdn-constructor"
  upcycle "${OUT}/${RUN_GROUP}/${RUN_USER}/gdn-constructor/checkpoints" \
    "${GDN_PATTERN}" "${HYLO_GDN_UPCYCLED_CKPT}"
fi
if [[ ! -f "${HYLO_MLA_UPCYCLED_CKPT}/latest_checkpointed_iteration.txt" ]]; then
  run_constructor "${MLA_PATTERN}" "mla-constructor"
  upcycle "${OUT}/${RUN_GROUP}/${RUN_USER}/mla-constructor/checkpoints" \
    "${MLA_PATTERN}" "${HYLO_MLA_UPCYCLED_CKPT}"
fi
if [[ ! -f "${HYLO_STAGE2_BASE_CKPT}/latest_checkpointed_iteration.txt" ]]; then
  run_constructor "${TARGET_PATTERN}" "4mla12gdn-constructor"
  upcycle "${OUT}/${RUN_GROUP}/${RUN_USER}/4mla12gdn-constructor/checkpoints" \
    "${TARGET_PATTERN}" "${HYLO_STAGE2_BASE_CKPT}"
fi

run_primus "${ROOT}/examples/megatron/configs/MI300X/hylo_gdn_1B-enhanced-ild.yaml" "gdn-ild"
run_primus "${ROOT}/examples/megatron/configs/MI300X/hylo_mla_1B-enhanced-ild.yaml" "mla-ild"

docker run --rm \
  -v "${ROOT}:${ROOT}" -w "${ROOT}" \
  "${IMAGE}" \
  python tools/hybrid/assemble_hylo_ild.py \
    --hybrid-base-checkpoint "${HYLO_STAGE2_BASE_CKPT}" \
    --gdn-ild-checkpoint "${HYLO_GDN_ILD_OUTPUT}" \
    --mla-ild-checkpoint "${HYLO_MLA_ILD_OUTPUT}" \
    --output-dir "${HYLO_STAGE2_INIT_CKPT}"

run_primus "${ROOT}/examples/megatron/configs/MI300X/hylo_4MLA12GDN_1B-8K-sft.yaml" \
  "4mla12gdn-8k-sft"
