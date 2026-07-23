#!/bin/bash
###############################################################################
# Mamba2 loss-parity experiment.
#
# Proves the cause of the ~+10% loss offset between our production
# zebra_llama_300M_mamba_hybrid and FLA's mamba2_300M_hybrid reference.
#
# It trains ONE model: an architectural byte-for-byte replica of FLA's
# mamba2_300M_hybrid.json (hidden 1216, ffn 4864, state 128, n_groups 1,
# eps 1e-5, MLP only on the 3 MLA layers).  Everything else — LR schedule,
# data, FLA runtime knobs — is identical to the production mamba sanity run,
# so the ONLY changed variable is the architecture.
#
# HYPOTHESIS : the production hybrid's +10% offset is an architecture/
#              hyperparameter mismatch vs FLA, not a Primus/Megatron bug.
# PREDICTION : this FLA-exact replica lands on FLA's loss curve within the
#              ~1-2% no-FLA-init band (same band the GDN hybrid sits in,
#              because hybrids have no FLA-init converter).
#   -> if confirmed, the offset is fully explained by config, case closed.
#   -> if it STILL shows ~10%, there is a real bug to chase.
#
# Run inside the rocm/primus container on 8xMI300X (same env as the sanity run):
#   cd /home/vanbhati@amd.com/Primus
#   bash sanity_before_push/mamba_parity/run_mamba_parity.sh
#
# Then compare:
#   python3 sanity_before_push/mamba_parity/compare_mamba_parity.py
###############################################################################
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# Mode: "flaexact" (default, FLA-arch replica, random init) or "flainit"
# (FLA-arch replica loading FLA's exact weights — the decisive init-vs-bug test).
MODE="${1:-flaexact}"

EXP_DIR="sanity_before_push/mamba_parity"
LOG_DIR="${EXP_DIR}/logs"
mkdir -p "${LOG_DIR}"

# Same FLA 10BT token cache the production mamba sanity run used.
FLA_CACHE_10BT="/home/vanbhati@amd.com/flash-linear-attention/legacy/training/data/HuggingFaceFW/fineweb-edu/sample-10BT/train"
FLA_INIT_CKPT="output/fla_init_mamba2_flaexact"
FLA_MAMBA_CFG="/home/vanbhati@amd.com/flash-linear-attention/legacy/training/configs/mamba2_300M_hybrid.json"

case "${MODE}" in
    flaexact)
        CFG="${EXP_DIR}/300M_mamba_hybrid_flaexact.yaml"
        LOG="${LOG_DIR}/300M_mamba_hybrid_flaexact.log"
        ;;
    flainit)
        CFG="${EXP_DIR}/300M_mamba_hybrid_flaexact_flainit.yaml"
        LOG="${LOG_DIR}/300M_mamba_hybrid_flaexact_flainit.log"
        # Prereq: the FLA-init checkpoint must exist (build it with the converter).
        if [[ ! -d "${FLA_INIT_CKPT}" ]]; then
            echo "[INFO] FLA-init checkpoint missing — building it now from ${FLA_MAMBA_CFG}"
            python3 tools/convert_fla_mamba2_init_to_megatron.py \
                --fla-config "${FLA_MAMBA_CFG}" \
                --output-dir "${FLA_INIT_CKPT}" --seed 42 || {
                    echo "[ERR ] converter failed"; exit 1; }
        fi
        ;;
    *)
        echo "[ERR ] unknown mode '${MODE}' (use: flaexact | flainit)"; exit 1 ;;
esac

if [[ ! -d "${FLA_CACHE_10BT}" ]]; then
    echo "[ERR ] missing FLA token cache: ${FLA_CACHE_10BT}"
    exit 1
fi
if [[ ! -f "${CFG}" ]]; then
    echo "[ERR ] missing config: ${CFG}"
    exit 1
fi

echo "[$(date)] mamba parity experiment: mode=${MODE}  (cfg=${CFG})"
echo "[$(date)] log -> ${LOG}"

PRIMUS_FLA_MLA_ATTN=1 \
PRIMUS_FUSED_CE=1 \
PRIMUS_FLA_SWIGLU=1 \
PRIMUS_FLA_NORM=1 \
PRIMUS_FLA_CONV=1 \
PRIMUS_FLA_DATA=1 \
PRIMUS_FLA_CACHE_DIR="${FLA_CACHE_10BT}" \
PRIMUS_TORCH_OPTIM=0 \
EXP="${CFG}" \
    bash examples/run_pretrain.sh 2>&1 | tee "${LOG}"
RC=${PIPESTATUS[0]}

echo
if [[ ${RC} -eq 0 ]]; then
    echo "[$(date)] DONE (rc=0).  Build the comparison with:"
    echo "  python3 ${EXP_DIR}/compare_mamba_parity.py"
else
    echo "[$(date)] FAILED (rc=${RC}).  See ${LOG}"
fi
exit ${RC}
