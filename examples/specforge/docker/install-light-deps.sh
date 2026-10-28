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
# tensorboard is not in this allowlist. The Dockerfile installs
# tensorboard>=2.14,<2.21 with its dependencies, constrained to versions
# already in the base image. SummaryWriter imports that tree, so a --no-deps
# copy leaves --report-to tensorboard broken. Do not re-add it here.
#
# The remaining entries stay --no-deps. Their job is to fill a missing import
# without letting pip replace the ROCm torch/sglang stack.
PACKAGES=(
  "pydantic"
  "pyyaml:yaml"
  "accelerate"
  "tqdm"
  "huggingface-hub:huggingface_hub"
  "numpy"
  "safetensors"
  "requests"
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
