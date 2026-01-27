###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


from typing import Any

from primus.backends.hummingbirdxt.hummingbirdxt_base_trainer import (
    HummingbirdXTBaseTrainer,
)
from primus.core.utils.yaml_utils import nested_namespace_to_dict
from primus.modules.module_utils import log_rank_0


class HummingbirdXTPosttrainTrainer(HummingbirdXTBaseTrainer):
    def __init__(self, primus_config: Any, module_config: Any, backend_args: Any):
        super().__init__(
            primus_config=primus_config,
            module_config=module_config,
            backend_args=backend_args,
        )
        log_rank_0("hb_trainer __init__ ...")

    def setup(self):
        log_rank_0("hb_trainer setup ...")

    def init(self):
        log_rank_0(f"hb_trainer init ..., self.backend_args={self.backend_args}")

        from omegaconf import OmegaConf
        from trainer import Wan22ScoreDistillationTrainer

        configs = OmegaConf.create(nested_namespace_to_dict(self.backend_args))
        self._trainer = Wan22ScoreDistillationTrainer(configs)

    def run_train(self):
        log_rank_0("hb_trainer run_train ...")
        self._trainer.train()
