###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
TorchTitanPretrainTrainer: Primus wrapper for TorchTitan pre-training.
"""

from typing import Any, Optional

from primus.backends.torchtitan.config_utils import build_job_config_from_namespace
from primus.core.trainer.base_trainer import BaseTrainer


class TorchTitanPretrainTrainer(BaseTrainer):
    """Trainer class for TorchTitan pre-training."""

    def __init__(self, backend_args: Any):
        # Patch TorchTitan logger before any other initialization
        self._patch_torchtitan_logger()

        super().__init__(backend_args=backend_args)
        self._trainer: Optional["Trainer"] = None  # type: ignore[name-defined]

    def _patch_torchtitan_logger(self):
        import torchtitan.tools.logging as titan_logging

        from primus.core.utils.logger import _logger as primus_logger

        titan_logging.logger = primus_logger
        titan_logging.init_logger = lambda: None

    def setup(self):
        pass

    def init(self):
        from torchtitan.train import Trainer  # type: ignore[import]

        job_config = build_job_config_from_namespace(self.backend_args)
        self._trainer = Trainer(job_config)

    def train(self):
        if self._trainer is None:
            raise RuntimeError("init() must be called before train()")

        self._trainer.train()
