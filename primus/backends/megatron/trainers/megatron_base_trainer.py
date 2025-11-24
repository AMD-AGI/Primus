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

        We only use parse_args() patching strategy to avoid conflicts with
        Megatron's initialize_megatron() which checks if args are already initialized.

        Raises:
            RuntimeError: If patching fails
        """
        log_rank_0("Injecting arguments into Megatron runtime...")

        # Patch parse_args() to return our pre-configured args
        if not self._patch_parse_args():
            raise RuntimeError(
                "Failed to patch Megatron's parse_args(). " "Cannot inject arguments into Megatron runtime."
            )

        log_rank_0("Args injected via parse_args patching")

    def _patch_parse_args(self) -> bool:
        """
        Monkey patch Megatron's parse_args to return our prepared args.

        This also ensures distributed environment variables are set in the args.

        Returns:
            True if patching succeeded, False otherwise
        """
        try:
            import megatron.training.arguments as megatron_args  # type: ignore
            import megatron.training.initialize as megatron_init  # type: ignore

            from primus.core.runtime.distributed import get_distributed_info

            # Get distributed environment info
            dist_env = get_distributed_info()

            # Update backend_args with distributed environment variables
            # This ensures Megatron uses the correct distributed settings
            self.backend_args.world_size = dist_env["world_size"]
            self.backend_args.rank = dist_env["rank"]
            self.backend_args.local_rank = dist_env["local_rank"]

            # Create a function that always returns our prepared args
            def patched_parse_args(*args, **kwargs):
                log_rank_0("parse_args() called, returning pre-configured args")
                return self.backend_args

            # Patch both locations where parse_args might be defined/called
            megatron_args.parse_args = patched_parse_args
            megatron_init.parse_args = patched_parse_args

            log_rank_0(f"Distributed environment info: {dist_env}")
            log_rank_0(f"Backend args: {self.backend_args}")

            log_rank_0(
                f"Patched parse_args with {len(vars(self.backend_args))} arguments "
                f"(rank={self.backend_args.rank}, world_size={self.backend_args.world_size})"
            )
            return True
        except (ImportError, AttributeError) as e:
            log_rank_0(f"WARNING: Cannot patch parse_args: {e}")
            return False

    def _detect_version(self) -> str:
        """Detect Megatron version."""
        try:
            import megatron

            if hasattr(megatron, "__version__"):
                return megatron.__version__
        except Exception:
            pass
        return "unknown"
