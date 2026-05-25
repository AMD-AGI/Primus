#!/bin/bash
###############################################################################
# End-to-end evaluation of the 75% Hybrid Mamba2+MLA 300M model.
#
# Run this *inside the rocm/primus container* with the repo mounted at
# /home/vanbhati@amd.com/Primus:
#
#   cd /home/vanbhati@amd.com/Primus && bash run_mamba_hybrid_eval.sh
#
# What it does:
#   1. Converts the Primus Megatron checkpoint → FLA HuggingFace
#      `Mamba2ForCausalLM` format using tools/convert_mamba_hybrid_to_fla_hf.py
#      (handles the 3 MLA + 9 Mamba2 sublayer mix; MLA path reuses the same
#      channel-permutation fix as the GDN-hybrid converter)
#   2. Runs lm-eval on the Primus-converted HF model
#   3. Runs lm-eval on FLA's reference HF checkpoint
#   4. Diffs the two scoreboards
#
# ARCHITECTURE NOTE — the two models are NOT bit-compatible:
#
#                   Primus YAML          FLA mamba2_300M_hybrid.json
#   hidden_size     1024                 1216
#   intermediate    4096                 4864
#   state_size      64                   128
#   n_groups        8                    1
#   total params   ~273M                ~350M
#
#   We sized Primus to match the GDN-hybrid 300M so the Mamba2 mixer is a
#   drop-in swap into the same backbone.  The eval below scores each model
#   on its own arch; treat the comparison as a "what 300M-ish hybrids land
#   at this loss" benchmark, not a checkpoint-conversion parity check.
###############################################################################
set -euo pipefail

PRIMUS_CKPT=${PRIMUS_CKPT:-output/amd/root/zebra_llama_300M_mamba_hybrid-pretrain/checkpoints/iter_0004768}
PRIMUS_HF_DIR=${PRIMUS_HF_DIR:-output/mamba_hybrid_300M_fla_hf}
FLA_HF_DIR=${FLA_HF_DIR:-/home/vanbhati@amd.com/checkpoints/mamba2_hybrid_300M_10B/checkpoint-4768}
RESULTS_DIR=${RESULTS_DIR:-output/mamba_hybrid_300M_eval_results}
TASKS=${TASKS:-arc_easy,arc_challenge,hellaswag,openbookqa,piqa,winogrande,mmlu,race}
BATCH_SIZE=${BATCH_SIZE:-auto}
TOKENIZER=${TOKENIZER:-/home/vanbhati@amd.com/checkpoints/gdn_pure_300M_10B}

echo "==========[run_mamba_hybrid_eval.sh]=========="
echo "PRIMUS_CKPT     = ${PRIMUS_CKPT}"
echo "PRIMUS_HF_DIR   = ${PRIMUS_HF_DIR}"
echo "FLA_HF_DIR      = ${FLA_HF_DIR}"
echo "RESULTS_DIR     = ${RESULTS_DIR}"
echo "TASKS           = ${TASKS}"
echo "BATCH_SIZE      = ${BATCH_SIZE}"
echo "TOKENIZER       = ${TOKENIZER}"
echo "================================================"

mkdir -p "${RESULTS_DIR}"

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Convert Primus checkpoint to FLA HF format
# ─────────────────────────────────────────────────────────────────────────────
if [ ! -f "${PRIMUS_HF_DIR}/model.safetensors" ]; then
    echo
    echo "==========[Step 1] Converting Primus → FLA HF (Mamba2ForCausalLM) =========="
    python tools/convert_mamba_hybrid_to_fla_hf.py \
        --checkpoint-path "${PRIMUS_CKPT}" \
        --output-dir "${PRIMUS_HF_DIR}" \
        --tokenizer "${TOKENIZER}"
else
    echo "Primus HF checkpoint already exists at ${PRIMUS_HF_DIR}, skipping conversion."
fi

# Ensure tokenizer files are present in both eval dirs
for tdir in "${PRIMUS_HF_DIR}" "${FLA_HF_DIR}"; do
    if [ ! -f "${tdir}/tokenizer.json" ] && [ -f "${TOKENIZER}/tokenizer.json" ]; then
        cp "${TOKENIZER}/tokenizer.json" \
           "${TOKENIZER}/tokenizer_config.json" \
           "${TOKENIZER}/special_tokens_map.json" \
           "${tdir}/" 2>/dev/null || true
        echo "Copied tokenizer from ${TOKENIZER} → ${tdir}"
    fi
done

# ─────────────────────────────────────────────────────────────────────────────
# Step 1b: Sanity-load both HF models, confirm forward + param count
# ─────────────────────────────────────────────────────────────────────────────
echo
echo "==========[Step 1b] Sanity-load both HF models =========="
# Uses AutoModelForCausalLM with trust_remote_code so the Primus-converted
# checkpoint picks up the custom Mamba2FullMlpForCausalLM class (which keeps
# MLPs on every layer; stock FLA Mamba2 only has MLPs on MLA layers).
PRIMUS_HF_DIR="${PRIMUS_HF_DIR}" FLA_HF_DIR="${FLA_HF_DIR}" python tools/_sanity_load_mamba.py

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Evaluate Primus-converted HF
# ─────────────────────────────────────────────────────────────────────────────
echo
echo "==========[Step 2] lm-eval on Primus HF (mamba2 hybrid) =========="
python tools/eval_mamba2_lm_eval.py \
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
echo "==========[Step 3] lm-eval on FLA HF (mamba2 hybrid) =========="
python tools/eval_mamba2_lm_eval.py \
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
