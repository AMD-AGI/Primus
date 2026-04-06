#!/usr/bin/env bash
# Download MLPerf Training v5.1 FLUX.1 (text-to-image) datasets per MLCommons instructions.
# Source: https://github.com/mlcommons/training/blob/master/text_to_image/README.md
#
# Location: examples/diffusion/mlperf_flux1/ (with setup_mlperf_flux1_automated.sh and export script).
#
# Usage:
#   export MLPERF_FLUX1_ROOT=/data/mlperf_flux1   # optional; default: ./mlperf_flux1_data
#   bash examples/diffusion/mlperf_flux1/download_mlperf_flux1_datasets.sh [minimal|preprocessed|all]
#
# Full pipeline (download + export + Energon prepare): see
#   bash examples/diffusion/mlperf_flux1/setup_mlperf_flux1_automated.sh minimal
#
# Phases:
#   minimal       — CC12M training subset on disk + COCO-2014 val subset + val TSV (~recommended start).
#   preprocessed  — Large precomputed embedding packs (~2.5 TB); for TorchTitan reference workflow.
#   all           — minimal + preprocessed
#
# --- End-to-end (run from Primus repository root unless noted) ---
#
# 1) Download MLPerf assets (this script)
#    export MLPERF_FLUX1_ROOT=/data/mlperf_flux1
#    bash examples/diffusion/mlperf_flux1/download_mlperf_flux1_datasets.sh minimal
#
# 2) Export WebDataset shards → flat images + sidecar .txt (fix --input-glob to your layout)
#    find "${MLPERF_FLUX1_ROOT}" -name '*.tar' | head   # locate shard paths
#    python examples/diffusion/mlperf_flux1/export_wds_to_flux_prepare_folder.py \
#      --input-glob "${MLPERF_FLUX1_ROOT}/**/*.tar" \
#      --output-dir "${MLPERF_FLUX1_ROOT}/cc12m_for_prepare"
#
# 3) Build Megatron-Energon FLUX shards (256×256, MLPerf-style) + energon metadata
#    pip install -r examples/diffusion/recipes/flux/requirements-prepare.txt   # OpenCV (cv2)
#    python examples/diffusion/recipes/flux/prepare_energon_dataset_flux.py \
#      --data_folder "${MLPERF_FLUX1_ROOT}/cc12m_for_prepare" \
#      --output_dir "${MLPERF_FLUX1_ROOT}/cc12m_flux_wds" \
#      --height 256 --width 256 --resize_mode bicubic --center-crop \
#      --max_sequence_length 256
#    energon prepare "${MLPERF_FLUX1_ROOT}/cc12m_flux_wds"
#
# 4) Point training at the prepared dataset; ensure HF can fetch FLUX.1-dev (recipe default)
#    export MLPERF_FLUX1_ENERGON="${MLPERF_FLUX1_ROOT}/cc12m_flux_wds"
#    huggingface-cli login   # if black-forest-labs/FLUX.1-dev requires your token
#
# 5) Run pretrain (single node, 8 GPUs — set MEGATRON_BRIDGE to your Megatron-Bridge clone)
#    cd /path/to/Primus
#    torchrun --nproc_per_node=8 -m primus.cli.main train pretrain \
#      --config examples/megatron_bridge/configs/MI300X/flux_12b_pretrain_mlperf_flux1.yaml \
#      --backend-path "${MEGATRON_BRIDGE}"
#
#    Optional: restrict visible GPUs → CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#    If FP8/Transformer Engine fails, override precision, e.g.:
#      ... same torchrun ...  (and in YAML set precision_config: bf16_mixed or use CLI overrides if supported)
#
# Optional encoders (only if you train from pixels inside TorchTitan, not for precomputed above):
#   export HF_TOKEN=...   # Hugging Face token with FLUX.1-schnell access
#   python "${TORCHTITAN_ROOT}/experiments/flux/scripts/download_encoders.py" --local_dir "${MLPERF_FLUX1_ROOT}/models" --hf_token "${HF_TOKEN}"
#
# Skip behavior (mlc-r2-downloader defaults: cc12m_disk/, coco/ under the data root):
#   If minimal assets already exist (val2014_30k.tsv + WebDataset .tar under cc12m_disk and coco),
#   the minimal downloads are skipped. Same for preprocessed dirs (cc12m_preprocessed,
#   coco_preprocessed, empty_encodings) with at least one file each.
#   Set FORCE_REDOWNLOAD=1 to always run the downloader / wget.

set -euo pipefail

PHASE="${1:-minimal}"
ROOT="${MLPERF_FLUX1_ROOT:-${PWD}/mlperf_flux1_data}"
DOWNLOADER='bash <(curl -fsSL https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh)'
FORCE_REDOWNLOAD="${FORCE_REDOWNLOAD:-}"

mkdir -p "${ROOT}"
cd "${ROOT}"

echo "MLPerf FLUX.1 data root: ${ROOT}"

run_dl() {
  local uri="$1"
  echo ">>> Downloading metadata URI: ${uri}"
  eval "${DOWNLOADER} ${uri}"
}

# True if dir contains at least one WebDataset-style shard (or resume target).
_has_wds_shard() {
  local dir="$1"
  [[ -d "${dir}" ]] || return 1
  find "${dir}" -type f \( -name '*.tar' -o -name '*.tgz' \) -print -quit 2>/dev/null | grep -q .
}

# Matches MLCommons R2 layout for flux-1-cc12m-disk / flux-1-coco (see text_to_image README).
minimal_datasets_present() {
  [[ -f "${ROOT}/val2014_30k.tsv" ]] || return 1
  _has_wds_shard "${ROOT}/cc12m_disk" || return 1
  _has_wds_shard "${ROOT}/coco" || return 1
  return 0
}

# Matches R2 paths from flux-1-*-preprocessed.uri and flux-1-empty-encodings.uri.
preprocessed_datasets_present() {
  local d
  for d in cc12m_preprocessed coco_preprocessed empty_encodings; do
    [[ -d "${ROOT}/${d}" ]] || return 1
    find "${ROOT}/${d}" -type f -print -quit 2>/dev/null | grep -q . || return 1
  done
  return 0
}

if [[ "${PHASE}" == "minimal" || "${PHASE}" == "all" ]]; then
  if [[ -n "${FORCE_REDOWNLOAD}" ]] || ! minimal_datasets_present; then
    # CC12M subset (~1.1M samples), 256×256 bicubic — training images on disk
    run_dl "https://training.mlcommons-storage.org/metadata/flux-1-cc12m-disk.uri"
    # COCO-2014 validation subset for benchmark eval protocol
    run_dl "https://training.mlcommons-storage.org/metadata/flux-1-coco.uri"
    if [[ ! -f "${ROOT}/val2014_30k.tsv" ]]; then
      echo ">>> wget val2014_30k.tsv"
      wget -q --show-progress -O "${ROOT}/val2014_30k.tsv" \
        "https://training.mlcommons-storage.org/flux_1/datasets/val2014_30k.tsv"
    else
      echo ">>> val2014_30k.tsv already present, skip wget"
    fi
  else
    echo ">>> Minimal FLUX.1 datasets already present under ${ROOT} (cc12m_disk + coco + val2014_30k.tsv). Skip download."
    echo "    Set FORCE_REDOWNLOAD=1 to re-download."
  fi
fi

if [[ "${PHASE}" == "preprocessed" || "${PHASE}" == "all" ]]; then
  if [[ -n "${FORCE_REDOWNLOAD}" ]] || ! preprocessed_datasets_present; then
    echo ">>> Preprocessed packs are very large (~2.5 TB total). Ensure sufficient space."
    run_dl "https://training.mlcommons-storage.org/metadata/flux-1-cc12m-preprocessed.uri"
    run_dl "https://training.mlcommons-storage.org/metadata/flux-1-coco-preprocessed.uri"
    run_dl "https://training.mlcommons-storage.org/metadata/flux-1-empty-encodings.uri"
  else
    echo ">>> Preprocessed FLUX.1 datasets already present (cc12m_preprocessed, coco_preprocessed, empty_encodings). Skip download."
    echo "    Set FORCE_REDOWNLOAD=1 to re-download."
  fi
fi

if [[ "${PHASE}" != "minimal" && "${PHASE}" != "preprocessed" && "${PHASE}" != "all" ]]; then
  echo "Unknown phase: ${PHASE}. Use: minimal | preprocessed | all" >&2
  exit 1
fi

echo "Done. Next: export WebDataset → flat folder (see header), then prepare_energon_dataset_flux.py + energon prepare."
