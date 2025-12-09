###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MegatronPretrainTrainer: Primus wrapper for Megatron-LM pre-training.

This trainer bridges Primus configuration system with Megatron-LM's training loop.

The trainer inherits from MegatronBaseTrainer which handles:
    - Argument injection and Megatron runtime initialization
    - Patch management (before_train, after_train) via template method
    - Common Megatron setup patterns

This class only needs to implement run_train() with the actual training logic.
"""

from typing import Any

from primus.backends.megatron.megatron_base_trainer import MegatronBaseTrainer
from primus.modules.module_utils import log_rank_0


class MegatronPretrainTrainer(MegatronBaseTrainer):
    """
    Trainer class for Megatron-LM pre-training.

    Inherits from MegatronBaseTrainer which handles:
        - Argument injection into Megatron runtime
        - Patch management via template method pattern
        - Common Megatron initialization patterns

    This class implements:
        - setup(): Pre-initialization setup (optional)
        - init(): Training-specific initialization
        - run_train(): Execute actual training loop (no patch management needed)
    """

    def __init__(self, primus_config: Any, module_config: Any, backend_args: Any):
        """
        Initialize Megatron pretrain trainer.

        Args:
            primus_config: Full Primus configuration
            module_config: Module-specific configuration
            backend_args: Megatron-LM argument namespace (from MegatronArgBuilder)
        """
        # Initialize base class (handles argument injection)
        super().__init__(
            primus_config=primus_config,
            module_config=module_config,
            backend_args=backend_args,
        )

        # Training components (will be set during training)
        self.model = None
        self.optimizer = None
        self.opt_param_scheduler = None

        log_rank_0(f"Initialized for model: {module_config.model or 'custom'}")

    def setup(self):
        """
        Setup phase (optional, for compatibility with BaseModule interface).

        Can be used for pre-initialization setup if needed.
        """
        log_rank_0("Setup phase")
        # Any pre-initialization setup can go here

    def init(self):
        """
        Initialize Megatron training components.

        Note:
            Argument injection is already done by MegatronBaseTrainer.__init__()
            This method can be used for trainer-specific initialization.
        """
        log_rank_0(f"Initializing Megatron training...")
        log_rank_0(f"Model: {self.module_config.model or 'custom'}")
        log_rank_0(f"Framework: {self.module_config.framework}")

        # Trainer-specific initialization can go here if needed

    def run_train(self):
        """
        Execute Megatron pre-training using the standard Megatron calling pattern.

        This method is called by MegatronBaseTrainer.run() after applying patches.
        It focuses solely on the training logic without patch management.
        """
        log_rank_0("Executing Megatron pretrain...")

        # Import Megatron components
        from megatron.core.enums import ModelType
        from megatron.training import inprocess_restart, pretrain
        from pretrain_gpt import forward_step, train_valid_test_datasets_provider

        from primus.core.utils.import_utils import get_model_provider

        # Configure training components
        train_valid_test_datasets_provider.is_distributed = True
        wrapped_pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

        # Execute training
        wrapped_pretrain(
            train_valid_test_datasets_provider,
            get_model_provider(),
            ModelType.encoder_or_decoder,
            forward_step,
            store=store,
        )

        log_rank_0("Megatron pretrain execution completed.")
