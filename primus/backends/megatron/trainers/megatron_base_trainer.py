###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MegatronBaseTrainer: Base class for all Megatron-LM trainers.

This base class handles Megatron-specific initialization logic that is
common across all Megatron training tasks (pretrain, sft, posttrain, etc.).
"""

import sys
from types import SimpleNamespace

from primus.core.config.primus_config import ModuleConfig, PrimusConfig
from primus.core.trainer.base_module import BaseModule
from primus.core.utils.distributed_logging import log_rank_0


class MegatronBaseTrainer(BaseModule):
    """
    Base trainer class for all Megatron-LM training tasks.

    This class handles Megatron-specific concerns:
        - Argument injection into Megatron's runtime
        - Common initialization patterns
        - Fallback mechanisms

    All Megatron trainers (pretrain, sft, posttrain) should inherit from this class.
    """

    def __init__(
        self,
        primus_config: PrimusConfig,
        module_config: ModuleConfig,
        backend_args: SimpleNamespace,
    ):
        """
        Initialize Megatron base trainer.

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

        # Inject arguments into Megatron runtime
        self._inject_megatron_args()

    def _inject_megatron_args(self):
        """
        Inject backend_args into Megatron's runtime.

        This uses multiple strategies to ensure arguments are properly set:
        1. Direct injection into global_vars
        2. Monkey patch parse_args()
        3. Fallback to sys.argv conversion
        """
        log_rank_0("Injecting arguments into Megatron runtime...")

        # Strategy 1: Try direct injection
        direct_success = self._try_direct_injection()

        # Strategy 2: Patch parse_args()
        patch_success = self._patch_parse_args()

        if direct_success:
            log_rank_0("Args injected via both direct assignment and parse_args patching")
        elif patch_success:
            log_rank_0("Args injected via parse_args patching only")
        else:
            log_rank_0("WARNING: All injection strategies failed, using sys.argv fallback")
            self._set_args_via_argv()

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
            log_rank_0(f"Cannot directly inject args: {e}")
            return False

    def _patch_parse_args(self) -> bool:
        """
        Monkey patch Megatron's parse_args to return our prepared args.

        Returns:
            True if patching succeeded, False otherwise
        """
        try:
            import megatron.training.arguments as megatron_args  # type: ignore
            import megatron.training.initialize as megatron_init  # type: ignore

            # Create a function that always returns our prepared args
            def patched_parse_args(*args, **kwargs):
                log_rank_0("parse_args() called, returning pre-configured args")
                return self.backend_args

            # Patch both locations where parse_args might be defined/called
            megatron_args.parse_args = patched_parse_args
            megatron_init.parse_args = patched_parse_args

            log_rank_0(f"Patched parse_args with {len(vars(self.backend_args))} arguments")
            return True
        except (ImportError, AttributeError) as e:
            log_rank_0(f"WARNING: Cannot patch parse_args: {e}")
            return False

    def _set_args_via_argv(self):
        """
        Fallback method: Convert backend_args to sys.argv format.

        Used when we cannot directly inject args into Megatron's global state.
        This converts the SimpleNamespace back to command-line format.
        """
        argv_list = ["megatron_trainer"]  # Script name

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
        log_rank_0(f"Set sys.argv with {len(argv_list)} arguments")

    def _detect_version(self) -> str:
        """Detect Megatron version."""
        try:
            import megatron

            if hasattr(megatron, "__version__"):
                return megatron.__version__
        except Exception:
            pass
        return "unknown"
