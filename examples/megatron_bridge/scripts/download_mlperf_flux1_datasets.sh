#!/usr/bin/env bash
# Download MLPerf Training v5.1 FLUX.1 (text-to-image) datasets per MLCommons instructions.
# Source: https://github.com/mlcommons/training/blob/master/text_to_image/README.md
#
# Usage:
#   export MLPERF_FLUX1_ROOT=/data/mlperf_flux1   # optional; default: ./mlperf_flux1_data
#   bash examples/megatron_bridge/scripts/download_mlperf_flux1_datasets.sh [minimal|preprocessed|all]
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
#    bash examples/megatron_bridge/scripts/download_mlperf_flux1_datasets.sh minimal
#
# 2) Export WebDataset shards → flat images + sidecar .txt (fix --input-glob to your layout)
#    find "${MLPERF_FLUX1_ROOT}" -name '*.tar' | head   # locate shard paths
#    python examples/megatron_bridge/scripts/export_wds_to_flux_prepare_folder.py \
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

set -euo pipefail

PHASE="${1:-minimal}"
ROOT="${MLPERF_FLUX1_ROOT:-${PWD}/mlperf_flux1_data}"
DOWNLOADER='bash <(curl -fsSL https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh)'

mkdir -p "${ROOT}"
cd "${ROOT}"

echo "MLPerf FLUX.1 data root: ${ROOT}"

run_dl() {
  local uri="$1"
  echo ">>> Downloading metadata URI: ${uri}"
  eval "${DOWNLOADER} ${uri}"
}

if [[ "${PHASE}" == "minimal" || "${PHASE}" == "all" ]]; then
  # CC12M subset (~1.1M samples), 256×256 bicubic — training images on disk
  run_dl "https://training.mlcommons-storage.org/metadata/flux-1-cc12m-disk.uri"
  # COCO-2014 validation subset for benchmark eval protocol
  run_dl "https://training.mlcommons-storage.org/metadata/flux-1-coco.uri"
  if [[ ! -f "${ROOT}/val2014_30k.tsv" ]]; then
    echo ">>> wget val2014_30k.tsv"
    wget -q --show-progress -O "${ROOT}/val2014_30k.tsv" \
      "https://training.mlcommons-storage.org/flux_1/datasets/val2014_30k.tsv"
  else
    echo ">>> val2014_30k.tsv already present, skip"
  fi
fi

if [[ "${PHASE}" == "preprocessed" || "${PHASE}" == "all" ]]; then
  echo ">>> Preprocessed packs are very large (~2.5 TB total). Ensure sufficient space."
  run_dl "https://training.mlcommons-storage.org/metadata/flux-1-cc12m-preprocessed.uri"
  run_dl "https://training.mlcommons-storage.org/metadata/flux-1-coco-preprocessed.uri"
  run_dl "https://training.mlcommons-storage.org/metadata/flux-1-empty-encodings.uri"
fi

if [[ "${PHASE}" != "minimal" && "${PHASE}" != "preprocessed" && "${PHASE}" != "all" ]]; then
  echo "Unknown phase: ${PHASE}. Use: minimal | preprocessed | all" >&2
  exit 1
fi

echo "Done. Next: export WebDataset → flat folder (see header), then prepare_energon_dataset_flux.py + energon prepare."
