###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
MaxDiffusion BackendAdapter implementation.

This is the MaxDiffusion counterpart of ``MaxTextAdapter``. MaxDiffusion is a
JAX training stack, so (like MaxText) it is launched without torchrun and its
config is a MaxDiffusion ``pyconfig`` file. The adapter is responsible for:

    - Preparing the MaxDiffusion/JAX backend environment (arch env defaults)
    - Making the ``maxdiffusion`` package importable
    - Converting Primus module config -> MaxDiffusion config namespace
    - Providing the MaxDiffusion trainer class to Primus
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

from primus.backends.maxdiffusion.argument_builder import MaxDiffusionConfigBuilder
from primus.core.backend.backend_adapter import BackendAdapter
from primus.core.utils.module_utils import log_rank_0, warning_rank_0

# Where the Dockerfile clones/installs MaxDiffusion (`git clone ... /workspace/maxdiffusion`).
_DEFAULT_MAXDIFFUSION_PATH = "/workspace/maxdiffusion"


class MaxDiffusionAdapter(BackendAdapter):
    """BackendAdapter implementation for MaxDiffusion (JAX)."""

    def __init__(self, framework: str = "maxdiffusion"):
        super().__init__(framework)
        self.third_party_dir_name = "maxdiffusion"

    def prepare_backend(self, config: Any):
        """Prepare the MaxDiffusion/JAX backend environment before trainer construction.

        Applies MaxDiffusion/ROCm arch defaults via ``os.environ.setdefault`` so
        they never override values already set by the environment or a per-config
        top-level ``env:`` block (applied earlier in
        ``TrainRuntime._apply_config_env``).

        Effective precedence (highest wins):
            per-config ``env:``  >  these arch defaults  >  image-baked defaults
        """
        self._apply_arch_env_defaults()
        super().prepare_backend(config)

    @staticmethod
    def _apply_arch_env_defaults() -> None:
        """Set backend/arch env defaults (only when unset) before JAX/XLA init."""
        # gfx950 (MI350X/MI355X) NaN/hang workaround; harmless / no-op on gfx942.
        if MaxDiffusionAdapter._is_gfx950():
            if os.environ.setdefault("RCCL_WARP_SPEED_AUTO", "0") == "0":
                log_rank_0("[Primus:maxdiffusion] gfx950 detected: RCCL_WARP_SPEED_AUTO=0 (default)")

    @staticmethod
    def _is_gfx950() -> bool:
        """Best-effort detection of a gfx950 (MI350X/MI355X) device via rocminfo."""
        import shutil
        import subprocess

        rocminfo = shutil.which("rocminfo") or "/opt/rocm/bin/rocminfo"
        try:
            out = subprocess.run([rocminfo], capture_output=True, text=True, timeout=15).stdout
        except Exception:  # noqa: BLE001 - detection must never abort a run
            return False
        return "gfx950" in out

    def setup_backend_path(self, backend_path=None) -> str:
        """Make the ``maxdiffusion`` package importable.

        Unlike MaxText (a git submodule under ``third_party/``), MaxDiffusion is
        installed with ``pip install -e .`` from a clone at
        ``/workspace/maxdiffusion`` (see docker/jax_maxdiffusion.*), so it is
        usually already importable and no sys.path edit is required. This override
        is therefore tolerant: it adds the checkout (and its ``src``) to
        ``sys.path`` when present, but never hard-fails when the package is an
        installed wheel.

        Resolution order: --backend_path > BACKEND_PATH > MAXDIFFUSION_PATH >
        /workspace/maxdiffusion.
        """
        candidate = (
            backend_path
            or os.getenv("BACKEND_PATH")
            or os.getenv("MAXDIFFUSION_PATH")
            or _DEFAULT_MAXDIFFUSION_PATH
        )
        resolved = ""
        root = Path(candidate)
        if root.exists():
            resolved = str(root.resolve())
            for p in (root, root / "src"):
                ap = os.path.abspath(str(p))
                if os.path.exists(ap) and ap not in sys.path:
                    sys.path.insert(0, ap)
                    log_rank_0(f"[Primus:maxdiffusion] sys.path.insert -> {ap}")

        # Verify importability without importing heavy deps eagerly.
        import importlib.util

        if importlib.util.find_spec("maxdiffusion") is None:
            warning_rank_0(
                "[Primus:maxdiffusion] `maxdiffusion` package not importable and "
                f"no checkout found at '{candidate}'. Set MAXDIFFUSION_PATH or install "
                "maxdiffusion (pip install -e .) in the image."
            )
        return resolved

    def convert_config(self, params: Any):
        """Convert Primus params -> MaxDiffusion configuration namespace."""
        builder = MaxDiffusionConfigBuilder()
        builder.update(params)
        maxdiffusion_config = builder.finalize()
        log_rank_0("[Primus:MaxDiffusionAdapter] Converted Primus module params -> MaxDiffusion config")
        return maxdiffusion_config

    def load_trainer_class(self, stage: str = "pretrain", trainer_class: Optional[str] = None):
        """Return the MaxDiffusion trainer class for the given stage."""
        if stage == "pretrain":
            from primus.backends.maxdiffusion.maxdiffusion_pretrain_trainer import (
                MaxDiffusionPretrainTrainer,
            )

            return MaxDiffusionPretrainTrainer
        raise ValueError(f"Invalid stage: {stage}")

    def detect_backend_version(self) -> str:
        """Detect MaxDiffusion version for logging/patching (best-effort)."""
        try:
            import maxdiffusion

            if hasattr(maxdiffusion, "__version__"):
                return maxdiffusion.__version__
        except Exception as exc:  # noqa: BLE001
            warning_rank_0(f"MaxDiffusionAdapter: Failed to detect MaxDiffusion version: {exc}")
        return "unknown"
