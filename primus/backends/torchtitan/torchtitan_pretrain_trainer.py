###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
TorchTitanPretrainTrainer: Primus wrapper for TorchTitan pre-training.

This trainer bridges Primus's configuration system with TorchTitan's training
loop, following the same pattern as ``MegatronPretrainTrainer`` does for
Megatron-LM.

The trainer inherits from ``TorchTitanBaseTrainer`` which handles:
    - Integration with the unified BaseTrainer workflow (run_patches)
    - Version detection and common logging

This class only needs to implement:
    - setup(): optional pre-initialization
    - init(): construct the underlying TorchTitan Trainer
    - run_train(): call into TorchTitan's training loop
"""

from typing import Any, Optional

from primus.backends.torchtitan.config_utils import build_job_config_from_namespace
from primus.backends.torchtitan.torchtitan_base_trainer import TorchTitanBaseTrainer
from primus.modules.module_utils import log_rank_0


class TorchTitanPretrainTrainer(TorchTitanBaseTrainer):
    """
    Trainer class for TorchTitan pre-training.
    """

    def __init__(self, backend_args: Any):
        """
        Initialize TorchTitan pretrain trainer.

        Args:
            backend_args: TorchTitan configuration as SimpleNamespace (from TorchTitanAdapter)
        """
        super().__init__(backend_args=backend_args)

        self._trainer: Optional["Trainer"] = None  # type: ignore[name-defined]

        log_rank_0("TorchTitanPretrainTrainer initialized")

    # --------------------------------------------------------------------- #
    # Lifecycle hooks
    # --------------------------------------------------------------------- #

    def setup(self):
        """
        Optional setup phase (kept for API symmetry with other trainers).
        """
        log_rank_0("TorchTitanPretrainTrainer.setup()")

    def init(self):
        """
        Construct the underlying TorchTitan Trainer using the JobConfig.
        """
        log_rank_0("TorchTitanPretrainTrainer.init() - building TorchTitan Trainer")

        from torchtitan.train import Trainer  # type: ignore[import]

        # Note: TorchTitan's logger has already been patched in __init__.py
        # to use a named logger instead of root logger for proper source tracking.
        # backend_args is a SimpleNamespace produced by TorchTitanJobConfigBuilder
        # Convert it to JobConfig for TorchTitan's Trainer (handles custom extensions)
        job_config = build_job_config_from_namespace(self.backend_args)
        self._trainer = Trainer(job_config)

    # --------------------------------------------------------------------- #
    # Training entrypoint
    # --------------------------------------------------------------------- #

    def run_train(self):
        """
        Execute TorchTitan pre-training using its Trainer.train() loop.

        This method is called by BaseTrainer.run() after applying patches.
        """
        if self._trainer is None:
            raise RuntimeError("TorchTitanPretrainTrainer.init() must be called before run_train().")

        log_rank_0("Executing TorchTitan pretrain...")
        self._trainer.train()
        log_rank_0("TorchTitan pretrain execution completed.")
