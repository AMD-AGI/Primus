###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Any

from primus.core.trainer.base_trainer import BaseTrainer
from primus.modules.module_utils import log_rank_0


class HummingbirdXTBaseTrainer(BaseTrainer):
    def __init__(self, primus_config: Any, module_config: Any, backend_args: Any):
        log_rank_0("hb_base_trainer __init__ ...")

        # Initialize BaseTrainer (stores configs, enables patch management)
        super().__init__(
            primus_config=primus_config,
            module_config=module_config,
            backend_args=backend_args,
        )

    @classmethod
    def detect_version(cls) -> str:
        log_rank_0("hb_base_trainer detect_version ...")

    @classmethod
    def detect_hummingbirdxt_version(cls) -> str:
        log_rank_0("hb_base_trainer detect_hummingbirdxt_version ...")
