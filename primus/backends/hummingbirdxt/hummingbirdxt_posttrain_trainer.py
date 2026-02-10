###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Any

from primus.core.trainer.base_trainer import BaseTrainer
from primus.core.utils.yaml_utils import nested_namespace_to_dict


class HummingbirdXTPosttrainTrainer(BaseTrainer):
    def __init__(self, backend_args: Any):
        super().__init__(backend_args=backend_args)

    def setup(self):
        pass

    def init(self):
        from omegaconf import OmegaConf
        from trainer import Wan22ScoreDistillationTrainer

        configs = OmegaConf.create(nested_namespace_to_dict(self.backend_args))
        self._trainer = Wan22ScoreDistillationTrainer(configs)

    def train(self):
        self._trainer.train()
