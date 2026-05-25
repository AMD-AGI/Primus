#!/bin/bash
###############################################################################
# End-to-end evaluation of the 75% Hybrid GDN+MLA 300M model.
#
# Run this *inside the rocm/primus container* with the repo mounted at
# /home/vanbhati@amd.com/Primus:
#
#   cd /home/vanbhati@amd.com/Primus && bash run_hybrid_eval.sh
#
# What it does:
#   1. Converts the Primus Megatron checkpoint → FLA HuggingFace format
#      (uses tools/convert_gdn_hybrid_to_fla_hf.py — handles the 3 MLA + 9 GDN
#      sublayer mix and FLA's nn.Sequential(Linear→RMSNorm→Linear) LoRA packing)
#   2. Runs lm-eval on the Primus-converted HF model
#   3. Runs lm-eval on FLA's HF checkpoint (apples-to-apples comparison)
#   4. Diffs the two scoreboards
###############################################################################
set -euo pipefail

PRIMUS_CKPT=${PRIMUS_CKPT:-output/amd/root/zebra_llama_300M_gdn_hybrid-pretrain/checkpoints/iter_0004768}
PRIMUS_HF_DIR=${PRIMUS_HF_DIR:-output/gdn_hybrid_300M_fla_hf}
FLA_HF_DIR=${FLA_HF_DIR:-/home/vanbhati@amd.com/checkpoints/gdn_hybrid_300M_10B/checkpoint-4768}
FLA_CONFIG=${FLA_CONFIG:-/home/vanbhati@amd.com/flash-linear-attention/legacy/training/configs/gated_deltanet_300M_hybrid.json}
RESULTS_DIR=${RESULTS_DIR:-output/gdn_hybrid_300M_eval_results}
TASKS=${TASKS:-arc_easy,arc_challenge,hellaswag,openbookqa,piqa,winogrande,mmlu,race}
BATCH_SIZE=${BATCH_SIZE:-auto}
TOKENIZER=${TOKENIZER:-/home/vanbhati@amd.com/checkpoints/gdn_pure_300M_10B}

echo "==========[run_hybrid_eval.sh]=========="
echo "PRIMUS_CKPT     = ${PRIMUS_CKPT}"
echo "PRIMUS_HF_DIR   = ${PRIMUS_HF_DIR}"
echo "FLA_HF_DIR      = ${FLA_HF_DIR}"
echo "FLA_CONFIG      = ${FLA_CONFIG}"
echo "RESULTS_DIR     = ${RESULTS_DIR}"
echo "TASKS           = ${TASKS}"
echo "BATCH_SIZE      = ${BATCH_SIZE}"
echo "TOKENIZER       = ${TOKENIZER}"
echo "========================================="

mkdir -p "${RESULTS_DIR}"

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Convert Primus checkpoint to FLA HF format
# ─────────────────────────────────────────────────────────────────────────────
if [ ! -f "${PRIMUS_HF_DIR}/model.safetensors" ]; then
    echo
    echo "==========[Step 1] Converting Primus → FLA HF =========="
    python tools/convert_gdn_hybrid_to_fla_hf.py \
        --checkpoint-path "${PRIMUS_CKPT}" \
        --output-dir "${PRIMUS_HF_DIR}" \
        --config "${FLA_CONFIG}"
else
    echo "Primus HF checkpoint already exists at ${PRIMUS_HF_DIR}, skipping conversion."
fi

# Copy FLA tokenizer into both eval dirs so lm-eval can load them with --tokenizer
# (the FLA hybrid config doesn't ship a tokenizer; it shares Llama-3.2 tokenizer)
for tdir in "${PRIMUS_HF_DIR}" "${FLA_HF_DIR}"; do
    if [ ! -f "${tdir}/tokenizer.json" ] && [ -f "${TOKENIZER}/tokenizer.json" ]; then
        cp "${TOKENIZER}/tokenizer.json" \
           "${TOKENIZER}/tokenizer_config.json" \
           "${TOKENIZER}/special_tokens_map.json" \
           "${tdir}/" 2>/dev/null || true
    fi
done

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Evaluate Primus-converted HF
# ─────────────────────────────────────────────────────────────────────────────
echo
echo "==========[Step 2] lm-eval on Primus HF =========="
python tools/eval_gdn_lm_eval.py \
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
python tools/eval_gdn_lm_eval.py \
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
python tools/compare_hybrid_eval.py \
    --primus-dir "${RESULTS_DIR}/primus" \
    --fla-dir "${RESULTS_DIR}/fla" \
    2>&1 | tee "${RESULTS_DIR}/scoreboard.txt"

echo
echo "DONE. Results saved to ${RESULTS_DIR}/"
