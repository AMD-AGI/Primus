#!/bin/bash
###############################################################################
# Pre-push FLA-parity sanity check.
#
# Trains the 5 models we ported to Primus for a short, FLA-schedule-matched
# budget (each model's FLA warmup + 500 steady steps) so we can confirm
# speed AND loss match FLA's reference runs BEFORE pushing/merging.
#
#   1. 300M_gdn_pure       warmup 200  + 500 = 700  iters
#   2. 300M_kda_pure       warmup 200  + 500 = 700  iters
#   3. 300M_gdn_hybrid     warmup 200  + 500 = 700  iters
#   4. 300M_mamba_hybrid   warmup 200  + 500 = 700  iters
#   5. 1B_gdn_pure         warmup 2000 + 500 = 2500 iters
#
# Each config keeps FLA's full lr_warmup_iters + lr_decay_iters, so the cosine
# LR at every compared step is bit-identical to FLA's full-length schedule —
# only the run stops early at warmup+500.
#
# Loss parity requires FLA-init weights + FLA token order:
#   - gdn_pure / kda_pure : load FLA-init checkpoints (output/fla_init_*).
#   - hybrids             : no FLA-init converter exists, so iter-1 loss is
#                           offset from FLA but the curve SHAPE + speed are
#                           still valid (flagged in the summary).
#   - all                 : use_fla_data=true → FLAOrderGPTDataset feeds the
#                           exact same token order as FLA's HF sampler.
#
# Usage (inside the rocm/primus container, 8×MI300X):
#   cd /home/vanbhati@amd.com/Primus && bash sanity_before_push/run_sanity.sh
#
# Run a subset:
#   bash sanity_before_push/run_sanity.sh 300M_gdn_pure 1B_gdn_pure
#
# Background it:
#   nohup bash sanity_before_push/run_sanity.sh \
#       > sanity_before_push/logs/_run_all.log 2>&1 &
#
# After it finishes, build the PASS/FAIL table:
#   python3 sanity_before_push/summarize_sanity.py --print
###############################################################################
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

SANITY_DIR="sanity_before_push"
CFG_DIR="${SANITY_DIR}/configs"
LOG_DIR="${SANITY_DIR}/logs"
mkdir -p "${LOG_DIR}"

# FLA HuggingFace token-cache roots (FLAOrderGPTDataset reads from here).
# 300M models train on the 10BT sample; the 1B model uses the 100BT sample.
FLA_CACHE_10BT="/home/vanbhati@amd.com/flash-linear-attention/legacy/training/data/HuggingFaceFW/fineweb-edu/sample-10BT/train"
FLA_CACHE_100BT="/home/vanbhati@amd.com/flash-linear-attention/legacy/training/data/HuggingFaceFW/fineweb-edu/sample-100BT/train"

# All 5 models, in fastest-first order so failures surface quickly.
ALL_MODELS=(
    "300M_gdn_pure"
    "300M_kda_pure"
    "300M_gdn_hybrid"
    "300M_mamba_hybrid"
    "1B_gdn_pure"
)

# Optional positional args select a subset; default = all.
if [[ $# -gt 0 ]]; then
    MODELS=("$@")
else
    MODELS=("${ALL_MODELS[@]}")
fi

# Per-model FLA token-cache directory.
fla_cache_for() {
    case "$1" in
        1B_gdn_pure) echo "${FLA_CACHE_100BT}" ;;
        *)           echo "${FLA_CACHE_10BT}"  ;;
    esac
}

# Models that need an FLA-init checkpoint for iter-1 loss parity.
required_ckpt_for() {
    case "$1" in
        300M_gdn_pure) echo "output/fla_init_ckpt_300M" ;;
        300M_kda_pure) echo "output/fla_init_kda_300M"  ;;
        *)             echo ""                           ;;
    esac
}

# Skip a model whose log already shows a completed run.
already_done() {
    local log="$1"
    [[ -f "${log}" ]] && grep -qE "after training is done|successfully shutdown|All processes shutdown" "${log}" 2>/dev/null
}

echo "[$(date)] sanity_before_push/run_sanity.sh starting (${#MODELS[@]} models)"
echo "[$(date)] models: ${MODELS[*]}"

START_TS=$(date +%s)
declare -A STATUS

for name in "${MODELS[@]}"; do
    CFG="${CFG_DIR}/${name}.yaml"
    LOG="${LOG_DIR}/${name}.log"

    if [[ ! -f "${CFG}" ]]; then
        echo "[ERR ] config missing: ${CFG}"
        STATUS[$name]="NO_CONFIG"
        continue
    fi

    # Prereq: FLA-init checkpoint (pure models only).
    REQ_CKPT="$(required_ckpt_for "${name}")"
    if [[ -n "${REQ_CKPT}" && ! -d "${REQ_CKPT}" ]]; then
        echo "[ERR ] ${name}: missing FLA-init checkpoint ${REQ_CKPT}"
        echo "       build it first, e.g.:"
        echo "         PYTHONPATH=<fla> python3 tools/convert_fla_gdn_init_to_megatron.py ..."
        echo "         (KDA: tools/convert_fla_kda_init_to_megatron.py)"
        STATUS[$name]="NO_CKPT"
        continue
    fi

    # Prereq: FLA token cache.
    CACHE="$(fla_cache_for "${name}")"
    if [[ ! -d "${CACHE}" ]]; then
        echo "[ERR ] ${name}: missing FLA token cache ${CACHE}"
        STATUS[$name]="NO_DATA"
        continue
    fi

    if already_done "${LOG}"; then
        echo "[skip] ${name}: ${LOG} already shows a completed run"
        STATUS[$name]="SKIP"
        continue
    fi

    echo
    echo "============================================================"
    echo "[$(date)] RUN: ${name}  (cfg=${CFG})"
    echo "============================================================"

    RUN_START=$(date +%s)
    # FLA-parity knobs are also declared in each YAML; exporting them here keeps
    # a bare shell reproducible and lets env override YAML (backward compat).
    PRIMUS_FLA_MLA_ATTN=1 \
    PRIMUS_FUSED_CE=1 \
    PRIMUS_FLA_SWIGLU=1 \
    PRIMUS_FLA_NORM=1 \
    PRIMUS_FLA_CONV=1 \
    PRIMUS_FLA_DATA=1 \
    PRIMUS_FLA_CACHE_DIR="${CACHE}" \
    PRIMUS_TORCH_OPTIM=0 \
    EXP="${CFG}" \
        bash examples/run_pretrain.sh 2>&1 | tee "${LOG}"
    RUN_RC=${PIPESTATUS[0]}
    RUN_END=$(date +%s)

    if [[ ${RUN_RC} -eq 0 ]]; then
        printf "[$(date)] %-20s OK    wall=%ds\n" "${name}" "$((RUN_END - RUN_START))"
        STATUS[$name]="OK"
    else
        printf "[$(date)] %-20s FAIL(rc=%d)  wall=%ds\n" "${name}" "${RUN_RC}" "$((RUN_END - RUN_START))" >&2
        echo "  See: ${LOG}"
        STATUS[$name]="FAIL"
    fi
done

TOTAL=$(( $(date +%s) - START_TS ))
echo
echo "============================================================"
echo "[$(date)] sanity runs finished (total wall=${TOTAL}s)"
for name in "${MODELS[@]}"; do
    printf "  %-20s %s\n" "${name}" "${STATUS[$name]:-UNKNOWN}"
done
echo
echo "Build the FLA parity table with:"
echo "  python3 ${SANITY_DIR}/summarize_sanity.py --print"
echo "============================================================"
