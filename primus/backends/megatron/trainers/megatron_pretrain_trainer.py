###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MegatronPretrainTrainer: Primus wrapper for Megatron-LM pre-training.

This trainer bridges Primus configuration system with Megatron-LM's training loop.
"""

from types import SimpleNamespace

from primus.backends.megatron.patches import apply_megatron_patches
from primus.backends.megatron.trainers.megatron_base_trainer import MegatronBaseTrainer
from primus.core.config.primus_config import ModuleConfig, PrimusConfig
from primus.core.utils.distributed_logging import log_rank_0


class MegatronPretrainTrainer(MegatronBaseTrainer):
    """
    Trainer class for Megatron-LM pre-training.

    Inherits from MegatronBaseTrainer which handles:
        - Argument injection into Megatron runtime
        - Common Megatron initialization patterns

    This class only needs to implement:
        - setup(): Pre-initialization setup (optional)
        - init(): Training-specific initialization
        - run(): Execute training loop
    """

    def __init__(
        self,
        primus_config: PrimusConfig,
        module_config: ModuleConfig,
        backend_args: SimpleNamespace,
    ):
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

    def run(self):
        """
        Execute Megatron pre-training using the standard Megatron calling pattern.
        """
        log_rank_0("Starting Megatron training...")

        # Apply before_train patches
        model_name = self.module_config.model if hasattr(self.module_config, "model") else None
        apply_megatron_patches(
            backend_version=self._detect_version(),
            model_name=model_name,
            phase="before_train",
            extra={"args": self.backend_args},
        )

        from megatron.core.enums import ModelType
        from megatron.training import inprocess_restart, pretrain
        from pretrain_gpt import forward_step, train_valid_test_datasets_provider

        from primus.backends.megatron.patches.model_provider import get_model_provider

        train_valid_test_datasets_provider.is_distributed = True
        wrapped_pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

        wrapped_pretrain(
            train_valid_test_datasets_provider,
            get_model_provider(),
            ModelType.encoder_or_decoder,
            forward_step,
            store=store,
        )

        log_rank_0("Training completed successfully.")
