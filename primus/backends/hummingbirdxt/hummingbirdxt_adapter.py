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

    # Backend-specific sys.path setup
    def setup_sys_path(self, backend_path: str):
        import sys
        from pathlib import Path

        for i, p in enumerate(sys.path):
            if isinstance(p, str) and Path(p).name == "HummingbirdXT":
                sys.path[i] = str(Path(p) / "train")

    # Backend Setup & Patches
    def prepare_backend(self, config: Any):
        BackendRegistry.run_setup("hummingbirdxt")

        log_rank_0("[Primus:HummingbirdXTAdapter] Backend prepared")

    # Config → HummingbirdXT Args
    def convert_config(self, module_config: Any):

        builder = HummingbirdXTArgBuilder()

        builder.update(module_config.params)

        hummingbirdxt_args = builder.finalize()
        return hummingbirdxt_args

    def load_trainer_class(self, stage: str = "posttrain"):
        try:
            return BackendRegistry.get_trainer_class(self.framework, stage=stage)
        except ValueError as exc:
            raise RuntimeError(
                f"[Primus:HummingbirdXTAdapter] '{self.framework}' backend trainer not registered. "
            ) from exc

    def detect_backend_version(self) -> str:
        from primus.backends.hummingbirdxt.hummingbirdxt_base_trainer import (
            HummingbirdXTBaseTrainer,
        )

        return HummingbirdXTBaseTrainer.detect_version()
