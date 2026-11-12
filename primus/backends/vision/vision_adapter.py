###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

from pathlib import Path
from typing import Any

from primus.backends.vision.argument_builder import VisionArgBuilder
from primus.core.backend.backend_adapter import BackendAdapter
from primus.core.utils.module_utils import log_rank_0


class VisionAdapter(BackendAdapter):
    """Primus adapter for multi-view vision training."""

    def __init__(self, framework: str = "vision"):
        super().__init__(framework)

    def setup_backend_path(self, backend_path=None) -> str:
        """Validate the in-tree vision package. Model code is imported later."""
        if backend_path:
            raise ValueError(
                "The vision backend is built into Primus and does not support "
                f"external backend_path overrides. Got: {backend_path}"
            )
        resolved = Path(__file__).resolve().parent
        if not resolved.exists():
            raise FileNotFoundError(f"[Primus:Vision] backend package does not exist: {resolved}")
        resolved_str = str(resolved)
        try:
            log_rank_0(f"[Primus:Vision] using in-tree backend package -> {resolved_str}")
        except Exception:
            pass
        return resolved_str

    def convert_config(self, params: Any):
        builder = VisionArgBuilder()
        builder.update(params)
        vision_args = builder.finalize()
        try:
            model_name = vision_args.model.get("name")
            log_rank_0(f"[Primus:VisionAdapter] Converted Primus module params -> {model_name}")
        except Exception:
            pass
        return vision_args

    def load_trainer_class(self, stage: str = "pretrain", trainer_class: str | None = None):
        if stage in ("pretrain", "posttrain", "sft"):
            from primus.backends.vision.vision_pretrain_trainer import VisionPretrainTrainer

            return VisionPretrainTrainer
        raise ValueError(f"Invalid stage for Vision backend: {stage}")

    def detect_backend_version(self) -> str:
        return "in-tree"
