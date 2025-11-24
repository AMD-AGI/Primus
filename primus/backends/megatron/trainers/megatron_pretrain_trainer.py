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

        This follows the standard Megatron pretrain pattern:
            1. Import required Megatron modules
            2. Setup data provider
            3. Get model provider
            4. Wrap pretrain() with inprocess_restart support
            5. Call pretrain() with all required arguments
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

        # Import Megatron core components
        from megatron.core.enums import ModelType  # type: ignore
        from megatron.training import inprocess_restart, pretrain  # type: ignore

        from primus.backends.megatron.patches.data_provider import (  # type: ignore
            get_train_valid_test_datasets_provider,
        )

        # Import forward step and data provider
        # Note: These can be customized per model type
        from primus.backends.megatron.patches.forward_step import (
            get_forward_step,  # type: ignore
        )

        # Import Primus model provider
        from primus.backends.megatron.patches.model_provider import get_model_provider

        # Get the model provider function
        model_provider = get_model_provider(self.backend_args)
        log_rank_0(f"Model provider: {getattr(self.backend_args, 'model_type', 'GPT')}")

        # Get the forward step function
        forward_step = get_forward_step(self.backend_args)
        log_rank_0("Forward step function configured")

        # Get the data provider
        train_valid_test_datasets_provider = get_train_valid_test_datasets_provider(self.backend_args)
        train_valid_test_datasets_provider.is_distributed = True
        log_rank_0("Dataset provider configured")

        # Wrap pretrain with inprocess restart support
        wrapped_pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

        # Determine model type
        model_type = ModelType.encoder_or_decoder
        if hasattr(self.backend_args, "model_type"):
            model_type_str = self.backend_args.model_type.upper()
            if "ENCODER_AND_DECODER" in model_type_str:
                model_type = ModelType.encoder_and_decoder
            elif "ENCODER" in model_type_str:
                model_type = ModelType.encoder_or_decoder

        log_rank_0(f"Model type: {model_type}")

        # Execute Megatron's pretrain with standard calling pattern
        wrapped_pretrain(
            train_valid_test_datasets_provider,
            model_provider,
            model_type,
            forward_step,
            store=store,
        )

        log_rank_0("Training completed successfully.")
