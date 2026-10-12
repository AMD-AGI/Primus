#!/bin/bash
# Install only missing lightweight Python packages with --no-deps so pip cannot
# replace the ROCm torch/sglang stack shipped in the base image.
set -euo pipefail

install_if_missing() {
  local pkg="$1"
  local mod="${2:-$1}"
  mod="${mod//-/_}"
  if python -c "import ${mod}" >/dev/null 2>&1; then
    return 0
  fi
  echo "Installing ${pkg} (--no-deps) for import ${mod}"
  python -m pip install --no-deps "${pkg}"
}

# SpecForge runtime imports (excluding torch, transformers, sglang, yunchang).
PACKAGES=(
  "pydantic"
  "pyyaml:yaml"
  "accelerate"
  "tqdm"
  "huggingface-hub:huggingface_hub"
  "numpy"
  "safetensors"
  "requests"
  "tensorboard"
  "typing-extensions:typing_extensions"
  "wandb"
  "datasets"
  "psutil"
  "openai-harmony:openai_harmony"
  "openai"
)

for entry in "${PACKAGES[@]}"; do
  IFS=':' read -r pkg mod <<< "${entry}"
  install_if_missing "${pkg}" "${mod:-}"
done

echo "Light dependency probe complete."
