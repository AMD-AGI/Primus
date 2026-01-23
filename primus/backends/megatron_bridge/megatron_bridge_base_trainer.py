###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MegatronBridgeBaseTrainer: Base class for all Megatron-Bridge trainers.

This mirrors the role of TorchTitanBaseTrainer for TorchTitan:

    - Inherits from the unified BaseTrainer so it participates in the
      common training workflow and patch management (via run_patches)
    - Provides a central place for Megatron-Bridge-specific initialization logic
      and version detection
    - Handles common setup logic shared across all Megatron-Bridge training tasks
"""

from typing import Any

from primus.core.patches import run_patches
from primus.core.trainer.base_trainer import BaseTrainer
from primus.modules.module_utils import log_rank_0


class MegatronBridgeBaseTrainer(BaseTrainer):
    """
    Base trainer class for all Megatron-Bridge training tasks.

    This class provides common functionality for all Megatron-Bridge trainers,
    including version detection, initialization logging, and shared setup logic.

    Responsibilities:
        - Call into the shared BaseTrainer to enable the unified workflow
          (before/after_train patches, lifecycle, logging)
        - Log Megatron-Bridge metadata (version, model, framework, task)
        - Provide a classmethod detect_version used by the patch system
        - Handle Megatron-Bridge specific initialization and setup
    """

    def __init__(self, primus_config: Any, module_config: Any, backend_args: Any):
        """
        Initialize Megatron-Bridge base trainer.

        Args:
            primus_config: Full Primus configuration
            module_config: Module-specific configuration
            backend_args: Megatron-Bridge configuration as SimpleNamespace
                         (from MegatronBridgeArgBuilder)
        """
        log_rank_0("=" * 80)
        log_rank_0("Initializing MegatronBridgeBaseTrainer...")
        log_rank_0("=" * 80)

        # Initialize BaseTrainer (stores configs, enables patch management)
        super().__init__(
            primus_config=primus_config,
            module_config=module_config,
            backend_args=backend_args,
        )

        import primus.backends.megatron.patches # noqa: F401
        run_patches(
            backend="megatron",
            phase="before_train",
            backend_version=type(self).detect_megatron_version(),
            model_name=self.model_name,
            extra={
                "backend_args": self.backend_args,
                "primus_config": self.primus_config,
                "module_config": self.module_config,
            },
        )

        log_rank_0("=" * 80)
        log_rank_0("MegatronBridgeBaseTrainer initialized successfully")
        log_rank_0("=" * 80)

    @classmethod
    def detect_version(cls) -> str:
        """
        Detect Megatron-Bridge version.

        Returns:
            Version string (e.g., "0.3.0rc0") from package_info

        Raises:
            RuntimeError: If version detection fails
        """
        try:
            from megatron.bridge.package_info import __version__

            return __version__
        except ImportError as e:
            raise RuntimeError(
                "Failed to detect Megatron-Bridge version. " "Make sure Megatron-Bridge is installed."
            ) from e

    @classmethod
    def detect_megatron_version(cls) -> str:
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

    
