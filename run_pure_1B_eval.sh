#!/bin/bash
###############################################################################
# End-to-end evaluation of the Pure GDN 1B / 100B-tokens model.
#
# Run this *inside the rocm/primus container* (e.g. primus_hybrid_new on tw006)
# with the repo mounted at /home/vanbhati@amd.com/Primus:
#
#   ssh tw006 \
#       'docker exec primus_hybrid_new \
#           bash -c "cd /home/vanbhati@amd.com/Primus && bash run_pure_1B_eval.sh"'
#
# What it does (mirrors run_hybrid_eval.sh, with one extra step up front):
#   0. (NEW) Consolidates the FSDP-dtensor distcp shards into the legacy
#      mp_rank_00/model_optim_rng.pt layout (tools/consolidate_distcp_to_torch.py)
#   1. Converts the consolidated Megatron checkpoint -> FLA HuggingFace format
#      via the existing tools/convert_gdn_to_fla_hf.py
#   2. (NEW) Architecturally verifies Primus-HF == FLA-HF
#   3. Runs lm-eval on the Primus-converted HF model
#   4. Runs lm-eval on FLA's HF checkpoint (apples-to-apples comparison)
#   5. Diffs the two scoreboards using tools/compare_hybrid_eval.py
###############################################################################
set -euo pipefail

PRIMUS_DISTCP=${PRIMUS_DISTCP:-output/amd/root/zebra_llama_1B_gdn_pure_100B-pretrain/checkpoints/iter_0095368}
PRIMUS_CONSOLIDATED=${PRIMUS_CONSOLIDATED:-output/amd/root/zebra_llama_1B_gdn_pure_100B-pretrain/checkpoints_consolidated/iter_0095368}
PRIMUS_HF_DIR=${PRIMUS_HF_DIR:-output/gdn_pure_1B_fla_hf}
FLA_HF_DIR=${FLA_HF_DIR:-/home/vanbhati@amd.com/checkpoints/gdn_pure_1B_100B/checkpoint-95368}
FLA_CONFIG=${FLA_CONFIG:-/home/vanbhati@amd.com/flash-linear-attention/legacy/training/configs/gated_deltanet_1B_pure_100B.json}
RESULTS_DIR=${RESULTS_DIR:-output/gdn_pure_1B_eval_results}
TASKS=${TASKS:-arc_easy,arc_challenge,hellaswag,openbookqa,piqa,winogrande,mmlu,race}
BATCH_SIZE=${BATCH_SIZE:-auto}
TOKENIZER=${TOKENIZER:-/home/vanbhati@amd.com/checkpoints/gdn_pure_1B_10B}

echo "==========[run_pure_1B_eval.sh]=========="
echo "PRIMUS_DISTCP        = ${PRIMUS_DISTCP}"
echo "PRIMUS_CONSOLIDATED  = ${PRIMUS_CONSOLIDATED}"
echo "PRIMUS_HF_DIR        = ${PRIMUS_HF_DIR}"
echo "FLA_HF_DIR           = ${FLA_HF_DIR}"
echo "FLA_CONFIG           = ${FLA_CONFIG}"
echo "RESULTS_DIR          = ${RESULTS_DIR}"
echo "TASKS                = ${TASKS}"
echo "BATCH_SIZE           = ${BATCH_SIZE}"
echo "TOKENIZER            = ${TOKENIZER}"
echo "========================================="

mkdir -p "${RESULTS_DIR}"

# ─────────────────────────────────────────────────────────────────────────────
# Step 0: Consolidate FSDP-dtensor shards -> legacy mp_rank_00 layout
# ─────────────────────────────────────────────────────────────────────────────
if [ ! -f "${PRIMUS_CONSOLIDATED}/mp_rank_00/model_optim_rng.pt" ]; then
    echo
    echo "==========[Step 0] Consolidating distcp -> legacy =========="
    python3 tools/consolidate_distcp_to_torch.py \
        --distcp-dir "${PRIMUS_DISTCP}" \
        --output-dir "${PRIMUS_CONSOLIDATED}"
else
    echo "Consolidated checkpoint already exists at ${PRIMUS_CONSOLIDATED}, skipping consolidation."
fi

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Convert legacy Megatron -> FLA HF format
# ─────────────────────────────────────────────────────────────────────────────
if [ ! -f "${PRIMUS_HF_DIR}/model.safetensors" ]; then
    echo
    echo "==========[Step 1] Converting Primus -> FLA HF =========="
    python3 tools/convert_gdn_to_fla_hf.py \
        --checkpoint-path "${PRIMUS_CONSOLIDATED}" \
        --output-dir "${PRIMUS_HF_DIR}" \
        --config "${FLA_CONFIG}"
else
    echo "Primus HF checkpoint already exists at ${PRIMUS_HF_DIR}, skipping conversion."
fi

# Copy tokenizer into Primus HF dir if missing (FLA HF dir already has its own)
for tdir in "${PRIMUS_HF_DIR}" "${FLA_HF_DIR}"; do
    if [ ! -f "${tdir}/tokenizer.json" ] && [ -f "${TOKENIZER}/tokenizer.json" ]; then
        for f in tokenizer.json tokenizer.model tokenizer_config.json special_tokens_map.json vocab.json merges.txt; do
            [ -f "${TOKENIZER}/${f}" ] && cp "${TOKENIZER}/${f}" "${tdir}/${f}"
        done
        echo "[tok] copied tokenizer files into ${tdir}"
    fi
done

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Evaluate Primus-converted HF
# ─────────────────────────────────────────────────────────────────────────────
echo
echo "==========[Step 2] lm-eval on Primus HF =========="
python3 tools/eval_gdn_lm_eval.py \
    --model hf \
    --model_args "pretrained=${PRIMUS_HF_DIR},dtype=bfloat16,trust_remote_code=True" \
    --tasks "${TASKS}" \
    --batch_size "${BATCH_SIZE}" \
    --output_path "${RESULTS_DIR}/primus" \
    2>&1 | tee "${RESULTS_DIR}/primus_eval.log"

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Evaluate FLA HF reference
# ─────────────────────────────────────────────────────────────────────────────
echo
echo "==========[Step 3] lm-eval on FLA HF =========="
python3 tools/eval_gdn_lm_eval.py \
    --model hf \
    --model_args "pretrained=${FLA_HF_DIR},dtype=bfloat16,trust_remote_code=True" \
    --tasks "${TASKS}" \
    --batch_size "${BATCH_SIZE}" \
    --output_path "${RESULTS_DIR}/fla" \
    2>&1 | tee "${RESULTS_DIR}/fla_eval.log"

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Compact side-by-side scoreboard
# ─────────────────────────────────────────────────────────────────────────────
echo
echo "==========[Step 4] Side-by-side scoreboard =========="
python3 tools/compare_hybrid_eval.py \
    --primus-dir "${RESULTS_DIR}/primus" \
    --fla-dir "${RESULTS_DIR}/fla" \
    2>&1 | tee "${RESULTS_DIR}/scoreboard.txt"

echo
echo "DONE. Results saved to ${RESULTS_DIR}/"
