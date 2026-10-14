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

    def __init__(self, backend_args: Any = None, **kwargs):
        # Patch TorchTitan logger before any other initialization
        self._patch_torchtitan_logger()

        # The core runtime instantiates every trainer with BaseModule-style
        # context kwargs (module_name, primus_config, module_rank, ...). Accept
        # and forward them so BaseTrainer can filter them cooperatively.
        super().__init__(backend_args=backend_args, **kwargs)
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

    def setup_model_only(self):
        """Build the model and optimizer, skipping the training loop.

        This is the hook projection's layer benchmark calls instead of
        :meth:`init` + :meth:`train`: it wants a real model built the way
        training builds it, but no tokenizer, dataset or step loop.  The model
        is exposed as ``self.model`` and the optimizer as ``self.optimizer``,
        the names the harness reads.
        """
        from primus.backends.torchtitan.model_builder import build_model_only

        job_config = build_job_config_from_namespace(self.backend_args)
        self.model, self.optimizer, self.parallel_dims = build_model_only(job_config)
        return self.model
