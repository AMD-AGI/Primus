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
from primus.core.config.primus_config import ModuleConfig, PrimusConfig
from primus.modules.base_module import BaseModule
from primus.modules.module_utils import log_rank_0


class MegatronPretrainTrainer(BaseModule):
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
        # Initialize BaseModule (auto-detects distributed params from env vars)
        super().__init__(
            module_name=module_config.module,
            primus_config=primus_config,
        )

        # Store backend-specific args
        self.backend_args = backend_args

        # Training components (will be set during training)
        self.model = None
        self.optimizer = None
        self.opt_param_scheduler = None

        log_rank_0(f"[MegatronPretrainTrainer] Initialized for model: {module_config.model or 'custom'}")

    def setup(self):
        """
        Setup phase (optional, for compatibility with BaseModule interface).

        Can be used for pre-initialization setup if needed.
        """
        log_rank_0("[MegatronPretrainTrainer] Setup phase")
        # Any pre-initialization setup can go here

    def init(self):
        """
        Initialize Megatron training components.

        This includes:
            - Initializing distributed environment
            - Setting up model
            - Creating optimizer
            - Loading checkpoints if specified
        """
        log_rank_0("[MegatronPretrainTrainer] Initializing Megatron training...")
        log_rank_0(f"[MegatronPretrainTrainer] Model: {self.module_config.model or 'custom'}")
        log_rank_0(f"[MegatronPretrainTrainer] Framework: {self.module_config.framework}")

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
        log_rank_0(
            f"[MegatronPretrainTrainer] Megatron initialized with {len(vars(megatron_args))} arguments"
        )

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
        log_rank_0("[MegatronPretrainTrainer] Starting Megatron training...")

        # Apply before_train patches
        model_name = self.module_config.model if hasattr(self.module_config, "model") else None
        apply_megatron_patches(
            backend_version=self._detect_version(),
            model_name=model_name,
            phase="before_train",
            extra={"args": self.backend_args},
        )

        # Import Megatron's pretrain function
        from megatron.training import pretrain  # type: ignore

        # Import model provider (this should be registered via patches)
        from primus.backends.megatron.patches.model_provider import get_model_provider

        # Get the model provider function
        model_provider = get_model_provider(self.backend_args)

        log_rank_0(
            f"[MegatronPretrainTrainer] Using model provider for: {getattr(self.backend_args, 'model_type', 'GPT')}"
        )

        # Execute Megatron's pretrain with our model provider
        pretrain(
            model_provider,
            model_type=getattr(self.backend_args, "model_type", "GPT"),
            forward_step_func=None,  # Use Megatron's default
            extra_args_provider=None,
            args_defaults={},
        )

        log_rank_0("[MegatronPretrainTrainer] Training completed successfully.")

    def _detect_version(self) -> str:
        """Detect Megatron version."""
        try:
            import megatron

            if hasattr(megatron, "__version__"):
                return megatron.__version__
        except Exception:
            pass
        return "unknown"
