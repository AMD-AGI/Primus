###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MegatronBaseTrainer: Base class for all Megatron-LM trainers.

This base class handles Megatron-specific initialization logic that is
common across all Megatron training tasks (pretrain, sft, posttrain, etc.).

Inherits from BaseTrainer which provides:
    - Universal training workflow (run template method)
    - Universal patch management (via run_patches)
    - Consistent training lifecycle

This class implements:
    - Megatron runtime initialization (parse_args patching, etc.)
    - ROCm compatibility patches
"""

from types import SimpleNamespace

from primus.core.config.primus_config import ModuleConfig, PrimusConfig
from primus.core.trainer.base_trainer import BaseTrainer
from primus.core.utils.distributed_logging import log_rank_0


class MegatronBaseTrainer(BaseTrainer):
    """
    Base trainer class for all Megatron-LM training tasks.

    This class handles Megatron-specific concerns:
        - Argument injection into Megatron's runtime (parse_args patching)
        - ROCm compatibility patches
        - Megatron version detection

    All Megatron trainers (pretrain, sft, posttrain) should inherit from this class.

    Note:
        Patch management is handled by BaseTrainer via run_patches().
        This class only handles Megatron-specific initialization.
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
        log_rank_0("=" * 80)
        log_rank_0("Initializing MegatronBaseTrainer...")
        log_rank_0("=" * 80)

        # Initialize BaseTrainer (auto-detects distributed params, stores configs)
        super().__init__(
            primus_config=primus_config,
            module_config=module_config,
            backend_args=backend_args,
        )

        # Log version information
        log_rank_0(f"Megatron-LM version: {self.backend_version}")
        log_rank_0(f"Model: {module_config.model}")
        log_rank_0(f"Framework: {module_config.framework}")

        # Inject arguments into Megatron runtime by patching parse_args()
        log_rank_0("Injecting arguments into Megatron runtime...")
        if not self._patch_parse_args():
            raise RuntimeError(
                "Failed to patch Megatron's parse_args(). " "Cannot inject arguments into Megatron runtime."
            )
        log_rank_0("Args injected via parse_args patching")

        # Apply AMD GPU runtime patches (Primus only supports AMD GPUs)
        self._patch_megatron_runtime_hooks()

        log_rank_0("=" * 80)
        log_rank_0("MegatronBaseTrainer initialized successfully")
        log_rank_0("=" * 80)

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

            # Create a lambda that always returns our prepared args
            patched_parse_args = lambda *args, **kwargs: (
                log_rank_0("parse_args() called, returning pre-configured args") or self.backend_args
            )

            # Patch both locations where parse_args might be defined/called
            megatron_args.parse_args = patched_parse_args
            megatron_init.parse_args = patched_parse_args

            log_rank_0(f"Patched parse_args with {len(vars(self.backend_args))} arguments")
            return True
        except (ImportError, AttributeError) as e:
            log_rank_0(f"WARNING: Cannot patch parse_args: {e}")
            return False

    def _detect_version(self) -> str:
        """
        Detect Megatron version.

        Overrides BaseTrainer._detect_version() to provide accurate
        Megatron version detection.

        Returns:
            Megatron version string (e.g., "0.8.0", "unknown")
        """
        try:
            import megatron

            if hasattr(megatron, "__version__"):
                return megatron.__version__
        except Exception:
            pass
        return "unknown"

    def _patch_megatron_runtime_hooks(self):
        """
        Apply runtime patches to Megatron for ROCm compatibility.

        Primus only supports AMD GPUs with ROCm. This method patches
        Megatron functions that try to compile CUDA-specific kernels.
        """
        try:
            import megatron.training.initialize as megatron_initialize  # type: ignore

            # Skip CUDA fused kernel compilation (not compatible with ROCm)
            megatron_initialize._compile_dependencies = lambda: log_rank_0(
                "Skipped Megatron _compile_dependencies() for ROCm compatibility"
            )
            log_rank_0("Applied Megatron runtime patches for ROCm")

        except (ImportError, AttributeError) as e:
            log_rank_0(f"WARNING: Failed to patch Megatron runtime hooks: {e}")
