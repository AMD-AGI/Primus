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

from typing import Any

from primus.core.trainer.base_trainer import BaseTrainer
from primus.modules.module_utils import log_rank_0


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

    def __init__(self, primus_config: Any, module_config: Any, backend_args: Any):
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
        log_rank_0(f"Megatron-LM version: {type(self).detect_version()}")
        log_rank_0(f"Model: {module_config.model}")
        log_rank_0(f"Framework: {module_config.framework}")

        # Inject arguments into Megatron runtime by patching parse_args()
        if not self._patch_parse_args():
            raise RuntimeError(
                "Failed to patch Megatron's parse_args(). " "Cannot inject arguments into Megatron runtime."
            )

        # Apply AMD GPU runtime patches (Primus only supports AMD GPUs)
        self._patch_megatron_runtime_hooks()

        log_rank_0("=" * 80)
        log_rank_0("MegatronBaseTrainer initialized successfully")
        log_rank_0("=" * 80)

    @classmethod
    def detect_version(cls) -> str:
        """
        Detect Megatron-LM version using the official method.

        Returns:
            Megatron version string (e.g., "0.15.0rc8")

        Raises:
            RuntimeError: If version cannot be detected (critical requirement)
        """
        try:
            from megatron.core import package_info

            return package_info.__version__
        except Exception as e:
            raise RuntimeError(
                "Failed to detect Megatron-LM version. "
                "Please ensure Megatron-LM is properly installed and "
                "megatron.core.package_info is available."
            ) from e

    def _patch_megatron_runtime_hooks(self):
        """
        Patch Megatron-LM runtime for ROCm compatibility.

        Primus only supports AMD GPUs with ROCm, so this method patches
        functions that would try to compile CUDA-specific kernels.
        """
        try:
            import megatron.training.initialize as megatron_initialize  # type: ignore

            # Skip CUDA fused kernel compilation (not compatible with ROCm)
            log_rank_0("Patching _compile_dependencies to skip CUDA kernel compilation")
            megatron_initialize._compile_dependencies = lambda: log_rank_0(
                "    Skipped _compile_dependencies() because CUDA kernels are not compatible with ROCm"
            )

            log_rank_0("Patched _compile_dependencies to skip CUDA kernel compilation")

        except (ImportError, AttributeError) as e:
            log_rank_0(f"WARNING: Failed to patch Megatron-LM runtime hooks: {e}")

    def _patch_parse_args(self) -> bool:
        """
        Patch Megatron-LM's parse_args to return pre-configured Primus arguments.

        This ensures Megatron-LM uses the pre-configured Primus arguments.

        Returns:
            True if patching succeeded, False otherwise.
        """
        try:
            import megatron.training.arguments as megatron_args  # type: ignore
            import megatron.training.initialize as megatron_init  # type: ignore

            log_rank_0("Patching Megatron-LM parse_args() to use pre-configured Primus arguments")

            # Create a lambda that always returns our prepared args
            patched_parse_args = lambda *args, **kwargs: (
                log_rank_0("parse_args() called; returning pre-configured Primus arguments")
                or self.backend_args
            )

            # Patch both locations where parse_args might be defined/called
            megatron_args.parse_args = patched_parse_args
            megatron_init.parse_args = patched_parse_args

            log_rank_0(
                f"Patched Megatron-LM parse_args(); Primus provided {len(vars(self.backend_args))} arguments"
            )
            return True
        except (ImportError, AttributeError) as e:
            log_rank_0(f"WARNING: Failed to patch Megatron-LM parse_args(): {e}")
            return False
