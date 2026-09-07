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

    - Declaring the MaxDiffusion/JAX arch env defaults (applied by the base
      adapter before JAX/XLA import, via the shared env_registry mechanism)
    - Making the ``maxdiffusion`` package importable
    - Converting Primus module config -> MaxDiffusion config namespace
    - Providing the MaxDiffusion trainer class to Primus
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, List, Optional

import primus.backends.maxdiffusion.patches  # noqa: F401  # Register patches
from primus.backends.maxdiffusion.argument_builder import MaxDiffusionConfigBuilder
from primus.backends.maxdiffusion.env_spec import maxdiffusion_env_defaults
from primus.core.backend.backend_adapter import BackendAdapter
from primus.core.backend.env_registry import EnvVar
from primus.core.utils.module_utils import log_rank_0, warning_rank_0

# Where the Dockerfile clones/installs MaxDiffusion (`git clone ... /workspace/maxdiffusion`).
_DEFAULT_MAXDIFFUSION_PATH = "/workspace/maxdiffusion"


class MaxDiffusionAdapter(BackendAdapter):
    """BackendAdapter implementation for MaxDiffusion (JAX)."""

    def __init__(self, framework: str = "maxdiffusion"):
        super().__init__(framework)
        self.third_party_dir_name = "maxdiffusion"

    def env_defaults(self) -> List[EnvVar]:
        """Declare MaxDiffusion arch env defaults (single source of truth).

        Returns the declarative env spec from ``env_spec.py``. The base adapter
        applies it via the shared ``env_registry`` mechanism in
        ``prepare_backend`` -> ``apply_env_defaults`` (before JAX/XLA import), so
        this adapter no longer overrides ``prepare_backend`` itself.

        Effective precedence (highest wins):
            per-config ``env:``  >  outer/shell env  >  these defaults  >  image-baked
        """
        return maxdiffusion_env_defaults()

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
