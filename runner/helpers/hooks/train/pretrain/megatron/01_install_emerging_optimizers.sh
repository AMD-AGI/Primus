#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Install NVIDIA-NeMo/Emerging-Optimizers inside the training container.
#
# The Muon optimizer path (primus/backends/megatron/core/optimizer/moun.py)
# hard-requires the ``emerging_optimizers`` package (Newton-Schulz orthogonal-
# ization, including the DeepSeek-V4 hybrid coefficient set). The package is
# NOT bundled in the default Primus container, and the public PyPI name
# ``emerging-optimizers`` is a placeholder stub (metadata-generation-failed) --
# the real package only installs from the GitHub source. This hook installs it
# from a pinned commit so a Muon run works out of the box.
#
# Gated by PRIMUS_INSTALL_EMERGING_OPTIMIZERS (default off) so non-Muon runs
# pay nothing; idempotent (skips when already importable). Pure-python wheel
# (~7s to build/install). run_deepseek_v4.sh sets the gate when OPTIMIZER=muon.
###############################################################################
set -euo pipefail

# Only do work when explicitly requested (the Muon launch path sets this).
case "${PRIMUS_INSTALL_EMERGING_OPTIMIZERS:-0}" in
  1 | true | True | TRUE | yes | on)
    ;;
  *)
    echo "[install_eo] PRIMUS_INSTALL_EMERGING_OPTIMIZERS not set; skipping."
    exit 0
    ;;
esac

# Pinned to match the third_party/Emerging-Optimizers submodule and the
# Megatron-LM muon.py integration (which passes ``use_nesterov`` to
# OrthogonalizedOptimizer). The older 06ff4c68 pin used the pre-rename
# ``nesterov`` kwarg and is incompatible with the current Megatron muon.py.
EO_COMMIT="${PRIMUS_EMERGING_OPTIMIZERS_COMMIT:-93d9eb3a6c899b50de73992826451fba3ab6adfb}"
EO_URL="git+https://github.com/NVIDIA-NeMo/Emerging-Optimizers.git@${EO_COMMIT}"

PY="${PYTHON:-python3}"

if "${PY}" -c "import emerging_optimizers" >/dev/null 2>&1; then
  echo "[install_eo] emerging_optimizers already importable; skipping install."
  exit 0
fi

echo "[install_eo] Installing emerging_optimizers from ${EO_URL} ..."
# --no-deps: the package's only hard runtime dep is torch (already in the
# container); avoid pulling an incompatible torch/transitive set.
"${PY}" -m pip install --no-cache-dir --no-deps "${EO_URL}"

if "${PY}" -c "import emerging_optimizers as e; print('[install_eo] installed', getattr(e,'__version__','?'))"; then
  echo "[install_eo] OK"
else
  echo "[install_eo] ERROR: emerging_optimizers still not importable after install" >&2
  exit 1
fi
