###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Any

from primus.backends.hummingbirdxt.argument_builder import HummingbirdXTArgBuilder
from primus.core.backend.backend_adapter import BackendAdapter
from primus.core.backend.backend_registry import BackendRegistry
from primus.modules.module_utils import log_rank_0


class HummingbirdXTAdapter(BackendAdapter):

    def __init__(self, framework: str = "hummingbirdxt"):
        super().__init__(framework)

    def setup_backend_path(self, backend_path=None) -> str:
        """
        Set up HummingbirdXT backend path.

        HummingbirdXT requires the 'train' subdirectory in sys.path.
        """
        import sys
        from pathlib import Path

        # Call parent to set up the main backend path
        resolved = super().setup_backend_path(backend_path)

        # Add HummingbirdXT/train subdirectory if needed
        train_path = str(Path(resolved) / "train")
        if train_path not in sys.path:
            sys.path.insert(0, train_path)
            log_rank_0(f"sys.path.insert → {train_path}")

        return resolved

    # Backend Setup & Patches
    def prepare_backend(self, config: Any):
        BackendRegistry.run_setup("hummingbirdxt")

        log_rank_0("[Primus:HummingbirdXTAdapter] Backend prepared")

    def convert_config(self, params: Any):
        """Convert Primus params to HummingbirdXT arguments."""
        builder = HummingbirdXTArgBuilder()
        builder.update(params)
        hummingbirdxt_args = builder.finalize()
        log_rank_0(f"[Primus:HummingbirdXTAdapter] Converted config → {len(vars(hummingbirdxt_args))} args")
        return hummingbirdxt_args

    def load_trainer_class(self, stage: str = "posttrain"):
        """Return HummingbirdXT trainer class for the given stage."""
        if stage == "posttrain":
            from primus.backends.hummingbirdxt.hummingbirdxt_posttrain_trainer import (
                HummingbirdXTPosttrainTrainer,
            )

            return HummingbirdXTPosttrainTrainer
        else:
            raise ValueError(f"Invalid stage: {stage}")

    def detect_backend_version(self) -> str:
        return "unknown"
