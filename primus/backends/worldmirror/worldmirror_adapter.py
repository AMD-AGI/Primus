###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

from typing import Any

from primus.core.backend.backend_adapter import BackendAdapter
from primus.core.utils.module_utils import log_rank_0


class WorldMirrorAdapter(BackendAdapter):
    """Launch the World Mirror submodule's Lightning trainer from Primus."""

    def __init__(self, framework: str = "worldmirror"):
        super().__init__(framework)
        self.third_party_dir_name = "HunyuanWorld-Mirror-rocm"

    def convert_config(self, params: Any) -> Any:
        if not hasattr(params, "worldmirror"):
            raise ValueError(
                "World Mirror module config is missing the 'worldmirror' block. "
                "Point model: at stage1_hypersim.yaml or stage2_hypersim.yaml."
            )
        try:
            hydra_config = params.worldmirror.hydra_config
            log_rank_0(f"[Primus:WorldMirrorAdapter] hydra train config -> {hydra_config}")
        except Exception:
            pass
        return params

    def load_trainer_class(self, stage: str = "pretrain"):
        if stage in ("pretrain", "posttrain", "sft"):
            from primus.backends.worldmirror.worldmirror_trainer import WorldMirrorTrainer

            return WorldMirrorTrainer
        raise ValueError(f"Invalid stage for World Mirror backend: {stage}")

    def detect_backend_version(self) -> str:
        return "in-tree"
