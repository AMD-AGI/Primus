#!/usr/bin/env bash
# End-to-end MLPerf FLUX.1 dataset setup inside the Primus repo.
# All scripts for this flow live under: examples/diffusion/mlperf_flux1/
#
# Steps: pip deps → download (minimal|preprocessed|all) → WDS export → prepare_energon_dataset_flux → energon prepare
#
# Run from anywhere; paths default under MLPERF_FLUX1_ROOT (default: /data/mlperf_flux1).
#
# Usage:
#   export MLPERF_FLUX1_ROOT=/data/mlperf_flux1          # optional
#   export MLPERF_FLUX1_INPUT_GLOB='/data/mlperf_flux1/**/*.tar'   # optional override for shard glob
#   bash examples/diffusion/mlperf_flux1/setup_mlperf_flux1_automated.sh [minimal|preprocessed|all]
#
# Skips (any non-empty value):
#   SKIP_PIP=1              Skip pip installs
#   SKIP_DOWNLOAD=1         Skip MLCommons download script
#   SKIP_EXPORT_WDS=1       Skip WebDataset → flat folder
#   SKIP_PREPARE_FLUX=1     Skip prepare_energon_dataset_flux.py
#   SKIP_ENERGON_PREPARE=1  Skip `energon prepare`
#
# Idempotent re-runs (download script also skips when R2 layout already exists; see download header):
#   If outputs already look complete, export / prepare / energon steps are skipped unless you set:
#   FORCE_EXPORT_WDS=1      Re-run WebDataset → flat folder even when cc12m_for_prepare has enough pairs
#   FORCE_PREPARE_FLUX=1    Re-run prepare_energon_dataset_flux.py even when shard-*.tar exists
#   FORCE_ENERGON_PREPARE=1 Re-run `energon prepare` even when metadataset.yaml (or dataset.yaml) exists
#   MLPERF_FLUX1_EXPORT_MIN_PAIRS=100  Minimum *.txt caption files in flat dir to treat export as done (default 100)
#
# After success, source the printed env or:
#   export MLPERF_FLUX1_ENERGON="${MLPERF_FLUX1_ROOT}/cc12m_flux_wds"
#
# Training (from Primus repo root; ties this setup script to the config via @):
#   ./primus-cli direct -- train pretrain \
#     --config examples/megatron_bridge/configs/MI300X/flux_12b_pretrain_mlperf_flux1.yaml@Primus/examples/diffusion/mlperf_flux1/setup_mlperf_flux1_automated.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# examples/diffusion/mlperf_flux1 → repo root (three levels up)
PRIMUS_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PRIMUS_ROOT}"

PHASE="${1:-minimal}"
ROOT="${MLPERF_FLUX1_ROOT:-/data/mlperf_flux1}"
PREPARE_DIR="${ROOT}/cc12m_for_prepare"
ENERGON_OUT="${ROOT}/cc12m_flux_wds"
# Recursive glob for .tar shards under the download root (quote in shell; Python expands)
INPUT_GLOB="${MLPERF_FLUX1_INPUT_GLOB:-${ROOT}/**/*.tar}"
ENV_SNIPPET="${ROOT}/mlperf_flux1_energon.env"

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "ERROR: required command not found: $1" >&2
    exit 1
  }
}

export_looks_complete() {
  local min="${MLPERF_FLUX1_EXPORT_MIN_PAIRS:-100}"
  local n
  [[ -d "${PREPARE_DIR}" ]] || return 1
  n=$(find "${PREPARE_DIR}" -maxdepth 1 -type f -name '*.txt' 2>/dev/null | wc -l)
  [[ "${n}" -ge "${min}" ]]
}

prepare_flux_looks_complete() {
  [[ -d "${ENERGON_OUT}" ]] || return 1
  find "${ENERGON_OUT}" -maxdepth 1 -type f -name '*.tar' -print -quit 2>/dev/null | grep -q .
}

energon_prepare_looks_complete() {
  [[ -f "${ENERGON_OUT}/metadataset.yaml" ]] && return 0
  [[ -f "${ENERGON_OUT}/dataset.yaml" ]] && return 0
  find "${ENERGON_OUT}" -maxdepth 4 \( -name 'metadataset.yaml' -o -name 'dataset.yaml' \) -print -quit 2>/dev/null | grep -q .
}

echo "=========================================="
echo "Primus MLPerf FLUX.1 automated setup"
echo "=========================================="
echo "PRIMUS_ROOT=${PRIMUS_ROOT}"
echo "MLPERF_SCRIPTS=${SCRIPT_DIR}"
echo "MLPERF_FLUX1_ROOT=${ROOT}"
echo "Download phase: ${PHASE}"
echo "WDS input glob: ${INPUT_GLOB}"
echo "Flat prepare dir: ${PREPARE_DIR}"
echo "Energon output: ${ENERGON_OUT}"
echo "=========================================="

require_cmd python3
require_cmd bash

if [[ -z "${SKIP_PIP:-}" ]]; then
  echo ">>> [pip] Installing MLPerf FLUX / Energon setup requirements…"
  python3 -m pip install -q --upgrade pip
  python3 -m pip install -r "${SCRIPT_DIR}/requirements-mlperf-flux1-setup.txt"
else
  echo ">>> [pip] SKIP_PIP set — skipping pip installs"
fi

if [[ -z "${SKIP_DOWNLOAD:-}" ]]; then
  require_cmd curl
  require_cmd wget
  echo ">>> [download] MLCommons R2 assets (${PHASE})…"
  MLPERF_FLUX1_ROOT="${ROOT}" bash "${SCRIPT_DIR}/download_mlperf_flux1_datasets.sh" "${PHASE}"
else
  echo ">>> [download] SKIP_DOWNLOAD set — skipping"
fi

mkdir -p "${ROOT}" "${PREPARE_DIR}" "${ENERGON_OUT}"

if [[ -z "${SKIP_EXPORT_WDS:-}" ]]; then
  if [[ -n "${FORCE_EXPORT_WDS:-}" ]] || ! export_looks_complete; then
    echo ">>> [export] WebDataset shards → flat folder for prepare…"
    python3 "${SCRIPT_DIR}/export_wds_to_flux_prepare_folder.py" \
      --input-glob "${INPUT_GLOB}" \
      --output-dir "${PREPARE_DIR}"
  else
    echo ">>> [export] Flat prepare dir already has ≥${MLPERF_FLUX1_EXPORT_MIN_PAIRS:-100} caption .txt files — skipping export"
    echo "    Set FORCE_EXPORT_WDS=1 to re-export."
  fi
else
  echo ">>> [export] SKIP_EXPORT_WDS set — skipping"
fi

REQ_PREPARE="${PRIMUS_ROOT}/examples/diffusion/recipes/flux/requirements-prepare.txt"
if [[ -z "${SKIP_PREPARE_FLUX:-}" ]]; then
  if [[ -n "${FORCE_PREPARE_FLUX:-}" ]] || ! prepare_flux_looks_complete; then
    echo ">>> [prepare] prepare_energon_dataset_flux.py (256×256, T5 len 256)…"
    if [[ -f "${REQ_PREPARE}" ]] && [[ -z "${SKIP_PIP:-}" ]]; then
      python3 -m pip install -q -r "${REQ_PREPARE}"
    fi
    python3 "${PRIMUS_ROOT}/examples/diffusion/recipes/flux/prepare_energon_dataset_flux.py" \
      --data_folder "${PREPARE_DIR}" \
      --output_dir "${ENERGON_OUT}" \
      --height 256 --width 256 \
      --resize_mode bicubic \
      --center-crop \
      --max_sequence_length 256
  else
    echo ">>> [prepare] Energon shard .tar files already present under ${ENERGON_OUT} — skipping prepare_energon_dataset_flux.py"
    echo "    Set FORCE_PREPARE_FLUX=1 to re-run prepare."
  fi
else
  echo ">>> [prepare] SKIP_PREPARE_FLUX set — skipping"
fi

if [[ -z "${SKIP_ENERGON_PREPARE:-}" ]]; then
  if [[ -n "${FORCE_ENERGON_PREPARE:-}" ]] || ! energon_prepare_looks_complete; then
    require_cmd energon
    echo ">>> [energon] energon prepare ${ENERGON_OUT}"
    energon prepare "${ENERGON_OUT}"
  else
    echo ">>> [energon] Dataset metadata already present (metadataset.yaml / dataset.yaml) — skipping energon prepare"
    echo "    Set FORCE_ENERGON_PREPARE=1 to re-run."
  fi
else
  echo ">>> [energon] SKIP_ENERGON_PREPARE set — skipping"
fi

cat >"${ENV_SNIPPET}" <<EOF
# Generated by examples/diffusion/mlperf_flux1/setup_mlperf_flux1_automated.sh — source before training:
export MLPERF_FLUX1_ROOT="${ROOT}"
export MLPERF_FLUX1_ENERGON="${ENERGON_OUT}"
EOF

echo ""
echo "=========================================="
echo "Done."
echo "Energon dataset: ${ENERGON_OUT}"
echo "Env file written: ${ENV_SNIPPET}"
echo "  source \"${ENV_SNIPPET}\""
echo "Or:"
echo "  export MLPERF_FLUX1_ENERGON=\"${ENERGON_OUT}\""
echo ""
echo "Train (from repo root):"
echo "  ./primus-cli direct -- train pretrain \\"
echo "    --config examples/megatron_bridge/configs/MI300X/flux_12b_pretrain_mlperf_flux1.yaml"
echo "=========================================="
