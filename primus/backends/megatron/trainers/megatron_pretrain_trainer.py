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

from primus.core.config.primus_config import ModuleConfig, PrimusConfig


class MegatronPretrainTrainer:
    """
    Trainer class for Megatron-LM pre-training.

    Responsibilities:
        1. Accept Primus configs and Megatron args
        2. Initialize Megatron training environment
        3. Execute Megatron's pretrain() function

    This is a lightweight wrapper that delegates to Megatron-LM's native
    training logic while maintaining Primus's configuration interface.
    """

    def __init__(
        self,
        primus_config: PrimusConfig,
        module_config: ModuleConfig,
        backend_args: SimpleNamespace,
    ):
        """
        Initialize Megatron trainer.

        Args:
            primus_config: Full Primus configuration
            module_config: Module-specific configuration
            backend_args: Megatron-LM argument namespace (from MegatronArgBuilder)
        """
        self.primus_config = primus_config
        self.module_config = module_config
        self.backend_args = backend_args

        # Will be set in init()
        self.model = None
        self.optimizer = None
        self.opt_param_scheduler = None

    def init(self):
        """
        Initialize Megatron training components.

        This includes:
            - Initializing distributed environment
            - Setting up model
            - Creating optimizer
            - Loading checkpoints if specified
        """
        print(f"[Primus:MegatronTrainer] Initializing Megatron training...")
        print(f"[Primus:MegatronTrainer] Model: {self.module_config.model or 'custom'}")
        print(f"[Primus:MegatronTrainer] Framework: {self.module_config.framework}")

        # Import Megatron modules (after backend setup in adapter)
        from megatron.training import get_args  # type: ignore
        from megatron.training.initialize import initialize_megatron  # type: ignore

        # Initialize Megatron with our prepared args
        initialize_megatron(
            extra_args_provider=None,
            args_defaults={},
            ignore_unknown_args=False,
            allow_no_cuda=False,
        )

        # Verify args were set correctly
        megatron_args = get_args()
        print(f"[Primus:MegatronTrainer] Megatron initialized with {len(vars(megatron_args))} arguments")

    def run(self):
        """
        Execute Megatron pre-training.

        This calls Megatron's main pretrain() function which handles:
            - Model setup
            - Data loading
            - Training loop
            - Checkpointing
            - Logging
        """
        print(f"[Primus:MegatronTrainer] Starting Megatron training...")

        # Import Megatron's pretrain function
        from megatron.training import pretrain  # type: ignore

        # Import model provider (this should be registered via patches)
        from primus.backends.megatron.patches.model_provider import get_model_provider

        # Get the model provider function
        model_provider = get_model_provider(self.backend_args)

        # Execute Megatron's pretrain with our model provider
        pretrain(
            model_provider,
            model_type=getattr(self.backend_args, "model_type", "GPT"),
            forward_step_func=None,  # Use Megatron's default
            extra_args_provider=None,
            args_defaults={},
        )

        print(f"[Primus:MegatronTrainer] Training completed.")
