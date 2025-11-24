###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MegatronPretrainTrainer: Primus wrapper for Megatron-LM pre-training.

This trainer bridges Primus configuration system with Megatron-LM's training loop.
"""

import sys
from types import SimpleNamespace

from primus.backends.megatron.patches import apply_megatron_patches
from primus.core.config.primus_config import ModuleConfig, PrimusConfig
from primus.core.trainer.base_module import BaseModule
from primus.core.utils.distributed_logging import log_rank_0


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

        Since MegatronAdapter.convert_config() has already:
            - Merged all configs (preset + user + CLI)
            - Filled in Megatron defaults
            - Produced a complete SimpleNamespace

        We inject this prepared args into Megatron using two strategies:
            1. Direct injection: Set global_vars._GLOBAL_ARGS (if available)
            2. Monkey patch: Replace parse_args() to return our args

        Strategy 2 (monkey patch) is more robust as it intercepts any
        parse_args() calls that might happen in pretrain().
        """
        log_rank_0("[MegatronPretrainTrainer] Initializing Megatron training...")
        log_rank_0(f"[MegatronPretrainTrainer] Model: {self.module_config.model or 'custom'}")
        log_rank_0(f"[MegatronPretrainTrainer] Framework: {self.module_config.framework}")

        # Strategy 1: Try direct injection first (fastest, least invasive)
        direct_injection_success = self._try_direct_injection()

        # Strategy 2: Always patch parse_args as well (most reliable)
        # This ensures that if pretrain() calls parse_args(), it gets our args
        self._patch_parse_args()

        if direct_injection_success:
            log_rank_0(
                "[MegatronPretrainTrainer] Args injected via both direct assignment "
                "and parse_args patching"
            )
        else:
            log_rank_0("[MegatronPretrainTrainer] Args injected via parse_args patching only")

    def _try_direct_injection(self) -> bool:
        """
        Try to directly inject args into Megatron's global state.

        Returns:
            True if successful, False otherwise
        """
        try:
            from megatron.training import global_vars  # type: ignore

            # Try to set directly (some versions have _GLOBAL_ARGS)
            if hasattr(global_vars, "_GLOBAL_ARGS"):
                global_vars._GLOBAL_ARGS = self.backend_args
                return True
            elif hasattr(global_vars, "_set_args"):
                global_vars._set_args(self.backend_args)
                return True
            else:
                return False
        except (ImportError, AttributeError) as e:
            log_rank_0(f"[MegatronPretrainTrainer] Cannot directly inject args: {e}")
            return False

    def _patch_parse_args(self):
        """
        Monkey patch Megatron's parse_args to return our prepared args.

        This is the most reliable way to inject args because:
        1. Works with all Megatron versions
        2. Intercepts parse_args() wherever it's called
        3. Allows us to add custom logic (e.g., argument validation)
        """
        try:
            import megatron.training.arguments as megatron_args  # type: ignore
            import megatron.training.initialize as megatron_init  # type: ignore

            # Create a function that always returns our prepared args
            def patched_parse_args(*args, **kwargs):
                log_rank_0("[MegatronPretrainTrainer] parse_args() called, " "returning pre-configured args")
                return self.backend_args

            # Patch both locations where parse_args might be defined/called
            megatron_args.parse_args = patched_parse_args
            megatron_init.parse_args = patched_parse_args

            log_rank_0(
                f"[MegatronPretrainTrainer] Patched parse_args with "
                f"{len(vars(self.backend_args))} arguments"
            )
        except (ImportError, AttributeError) as e:
            log_rank_0(f"[MegatronPretrainTrainer] WARNING: Cannot patch parse_args: {e}")
            # If we can't patch, we'll need sys.argv fallback
            self._set_args_via_argv()

    def run(self):
        """
        Execute Megatron pre-training.

        This calls Megatron's main pretrain() function which handles:
            - Argument parsing (from sys.argv we prepared in init())
            - Megatron initialization (distributed, model, optimizer)
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

        # Import model provider
        from primus.backends.megatron.patches.model_provider import get_model_provider

        # Get the model provider function
        # Note: This needs to be compatible with the backend_args we've set
        model_provider = get_model_provider(self.backend_args)

        log_rank_0(
            f"[MegatronPretrainTrainer] Using model provider for: {getattr(self.backend_args, 'model_type', 'GPT')}"
        )

        # Execute Megatron's pretrain
        # pretrain() will:
        #   1. Call initialize_megatron() internally (parsing our sys.argv)
        #   2. Setup distributed training
        #   3. Create model using model_provider
        #   4. Run training loop
        pretrain(
            model_provider,
            model_type=getattr(self.backend_args, "model_type", "GPT"),
            forward_step_func=None,  # Use Megatron's default
            extra_args_provider=None,
            args_defaults={},
        )

        log_rank_0("[MegatronPretrainTrainer] Training completed successfully.")

    def _set_args_via_argv(self):
        """
        Fallback method: Convert backend_args to sys.argv format.

        Used when we cannot directly inject args into Megatron's global state.
        This converts the SimpleNamespace back to command-line format.
        """
        argv_list = ["megatron_pretrain"]  # Script name

        # Convert all backend_args to command-line arguments
        for key, value in vars(self.backend_args).items():
            if value is None:
                continue

            # Convert to command-line format (underscore to hyphen)
            arg_name = key.replace("_", "-")

            # Handle different value types
            if isinstance(value, bool):
                # Boolean flags: only add if True
                if value:
                    argv_list.append(f"--{arg_name}")
            elif isinstance(value, (list, tuple)):
                # Lists: add each element
                argv_list.append(f"--{arg_name}")
                for item in value:
                    argv_list.append(str(item))
            else:
                # Regular values: add key and value
                argv_list.append(f"--{arg_name}")
                argv_list.append(str(value))

        # Replace sys.argv
        self._original_argv = sys.argv
        sys.argv = argv_list
        log_rank_0(f"[MegatronPretrainTrainer] Set sys.argv with {len(argv_list)} arguments")

    def _detect_version(self) -> str:
        """Detect Megatron version."""
        try:
            import megatron

            if hasattr(megatron, "__version__"):
                return megatron.__version__
        except Exception:
            pass
        return "unknown"
