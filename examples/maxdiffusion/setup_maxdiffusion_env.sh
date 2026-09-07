#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Install + patch the MaxDiffusion (JAX) runtime from the vendored submodule.
#
# When is this needed?
#   - Python deps (requirements-maxdiffusion.txt) are ALWAYS installed to ensure
#     Primus core deps like loguru are present before the runtime starts.
#   - If the container ALREADY ships maxdiffusion (e.g. the MAD primus_maxdiffusion
#     image or the unified docker), the script installs deps then exits early.
#     Setting PRIMUS_SKIP_PIP=1 skips calling it entirely.
#   - If the container does NOT have maxdiffusion (e.g. a bare rocm/jax-training
#     maxtext image), this script installs everything from the Primus checkout:
#     torch (ROCm wheels), editable submodule install, and 4 site-package patches.
#     Requires `third_party/maxdiffusion` to be initialized
#     (git submodule update --init).
#
# Invoked by examples/run_pretrain.sh when BACKEND=maxdiffusion (unless
# PRIMUS_SKIP_PIP=1). Idempotent: safe to re-run.
set -euo pipefail

PRIMUS_PATH="${PRIMUS_PATH:-$(realpath "$(dirname "$0")/../..")}"
MAXDIFFUSION_PATH="${MAXDIFFUSION_PATH:-$PRIMUS_PATH/third_party/maxdiffusion}"
log() { echo "[setup-maxdiffusion] $*"; }

# Always ensure Python deps are present (loguru, transformers, etc.) even when
# maxdiffusion is already importable -- the Primus runtime needs these at startup
# before any backend hooks run.
log "installing requirements-maxdiffusion.txt"
pip install -r "$PRIMUS_PATH/requirements-maxdiffusion.txt" --quiet

# Images that bake the stack (maxdiffusion installed, patches applied) only needed
# the deps above. On those there may be no submodule source to install from, so
# exit before the source check below can fail on its absence.
if python -c "import maxdiffusion" 2>/dev/null; then
  log "maxdiffusion already installed ($(python -c 'import maxdiffusion,os; print(os.path.dirname(maxdiffusion.__file__))' 2>/dev/null)): skipping submodule install + patches"
  exit 0
fi

if [ ! -e "$MAXDIFFUSION_PATH/pyproject.toml" ] && [ ! -e "$MAXDIFFUSION_PATH/setup.py" ]; then
  log "ERROR: no maxdiffusion source at $MAXDIFFUSION_PATH"
  log "       run: git -C \"$PRIMUS_PATH\" submodule update --init third_party/maxdiffusion"
  exit 1
fi

# Force Transformer Engine to load ONLY its JAX extension (torch present -> TE
# torch import would otherwise be attempted).
export NVTE_FRAMEWORK="${NVTE_FRAMEWORK:-jax}"

# 1) torch/torchvision from the ROCm wheel index (matches base ROCm rel).
if python -c "import torch" 2>/dev/null; then
  log "torch present: $(python -c 'import torch; print(torch.__version__)')"
else
  log "installing torch/torchvision (ROCm wheels)"
  pip install torch==2.8.0 torchvision==0.23.0 \
    --find-links https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.4/ \
    --no-index --quiet
fi

# 2) editable install of the vendored MaxDiffusion submodule (deps handled above).
log "pip install -e maxdiffusion (--no-deps)"
pip install -e "$MAXDIFFUSION_PATH" --no-deps --quiet

# 3) Patches (idempotent). These were previously baked into
#    docker/primus_maxdiffusion.ubuntu.amd.Dockerfile; here they target the
#    vendored submodule + the venv site-packages.
SP="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

# 4a) transformers Flax T5 (FLUX text encoder): jnp.clip a_min/a_max -> min/max.
T5="$SP/transformers/models/t5/modeling_flax_t5.py"
if [ -f "$T5" ] && grep -q "a_max=" "$T5"; then
  sed -i 's/a_max=/max=/g; s/a_min=/min=/g' "$T5"; log "patched transformers T5"
else
  log "transformers T5 patch: not needed"
fi

# 4b) preload tensorflow before Transformer Engine (import order segfault fix).
TU="$MAXDIFFUSION_PATH/src/maxdiffusion/train_utils.py"
if [ -f "$TU" ] && ! grep -q "preload before Transformer Engine" "$TU"; then
  sed -i '/from transformer_engine.jax.sharding import global_shard_guard, MeshResource/i\    import tensorflow  # noqa: F401 preload before Transformer Engine to avoid segfault' "$TU"
  log "patched maxdiffusion train_utils (TF preload)"
else
  log "train_utils TF-preload patch: already applied / n/a"
fi

# 4c) keep Shardy enabled for the GSPMD fallback path (cudnn_flash_te).
AF="$MAXDIFFUSION_PATH/src/maxdiffusion/models/attention_flax.py"
if [ -f "$AF" ]; then
  sed -i 's/jax.config.update("jax_use_shardy_partitioner", False)/jax.config.update("jax_use_shardy_partitioner", True)/g' "$AF"
  log "ensured Shardy on in attention_flax"
fi

# 4d) TE fused-attn partitioner: treat empty context-parallel axis as size 1.
TE="$SP/transformer_engine/jax/sharding.py"
if [ -f "$TE" ] && ! grep -q "if not axis:" "$TE"; then
  sed -i 's|    assert axis in mesh.shape, f"{axis} is not a axis of the given mesh {mesh.shape}"|    if not axis:\n        return 1\n    assert axis in mesh.shape, f"{axis} is not a axis of the given mesh {mesh.shape}"|' "$TE"
  log "patched transformer_engine sharding (empty CP axis)"
else
  log "TE CP-axis patch: already applied / n/a"
fi

log "MaxDiffusion env ready (source=$MAXDIFFUSION_PATH)"
