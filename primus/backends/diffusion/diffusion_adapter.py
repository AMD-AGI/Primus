###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

from pathlib import Path
from typing import Any

from primus.backends.diffusion.argument_builder import WanArgBuilder
from primus.core.backend.backend_adapter import BackendAdapter
from primus.modules.module_utils import log_rank_0


class DiffusionAdapter(BackendAdapter):
    """Primus adapter for PyTorch diffusion training."""

    def __init__(self, framework: str = "diffusion"):
        super().__init__(framework)

    def setup_backend_path(self, backend_path=None) -> str:
        """Validate the Primus-owned in-tree diffusion package."""
        if backend_path:
            raise ValueError(
                "The diffusion backend is built into Primus and does not support "
                f"external backend_path overrides. Got: {backend_path}"
            )

        resolved = Path(__file__).resolve().parent
        if not resolved.exists():
            raise FileNotFoundError(f"[Primus:Diffusion] backend package does not exist: {resolved}")

        resolved_str = str(resolved)
        try:
            log_rank_0(f"[Primus:Diffusion] using in-tree backend package -> {resolved_str}")
        except Exception:
            pass

        return resolved_str

    def convert_config(self, params: Any):
        builder = WanArgBuilder()
        builder.update(params)
        wan_args = builder.finalize()
        # convert_config is also called by the standalone prepare hook, where the
        # Primus logger may not be initialized yet; guard the informational log.
        try:
            log_rank_0("[Primus:DiffusionAdapter] Converted Primus module params -> Wan args")
        except Exception:
            pass
        return wan_args

    def load_trainer_class(self, stage: str = "pretrain"):
        if stage in ("pretrain", "posttrain", "sft"):
            from primus.backends.diffusion.diffusion_pretrain_trainer import (
                DiffusionPretrainTrainer,
            )

            return DiffusionPretrainTrainer
        raise ValueError(f"Invalid stage for Diffusion backend: {stage}")

    def detect_backend_version(self) -> str:
        return "in-tree"
