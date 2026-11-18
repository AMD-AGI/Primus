#!/bin/bash
# Optional fetch of the preprocessed Flux MLPerf dataset.
# Usage: PREPROCESSED_PATH=/shared/path/to/flux1/data ./scripts/download_ref_data.sh
# Then set DATADIR to that same path.

set -euo pipefail

: "${PREPROCESSED_PATH:?set PREPROCESSED_PATH to the shared dataset root (later used as DATADIR)}"

mkdir -p "$PREPROCESSED_PATH"

downloader_dir=$(mktemp -d)
trap 'rm -rf "$downloader_dir"' EXIT
downloader="$downloader_dir/mlc-r2-downloader.sh"
curl -fsSL https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh -o "$downloader"

datasets=(
  "cc12m_preprocessed https://training.mlcommons-storage.org/metadata/flux-1-cc12m-preprocessed.uri"
  "coco_preprocessed https://training.mlcommons-storage.org/metadata/flux-1-coco-preprocessed.uri"
  "empty_encodings https://training.mlcommons-storage.org/metadata/flux-1-empty-encodings.uri"
)

for entry in "${datasets[@]}"; do
  read -r name url <<< "$entry"
  echo "Downloading $name to $PREPROCESSED_PATH/$name"
  bash "$downloader" -d "$PREPROCESSED_PATH/$name" "$url"
done

echo "All downloads completed successfully."
