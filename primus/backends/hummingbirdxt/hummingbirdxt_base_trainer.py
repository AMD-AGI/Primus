###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Any

from primus.core.trainer.base_trainer import BaseTrainer


class HummingbirdXTBaseTrainer(BaseTrainer):
    def __init__(self, primus_config: Any, module_config: Any, backend_args: Any):
        # Initialize BaseTrainer (stores configs, enables patch management)
        super().__init__(
            primus_config=primus_config,
            module_config=module_config,
            backend_args=backend_args,
        )

    @classmethod
    def detect_version(cls) -> str:
        return "unknown"
