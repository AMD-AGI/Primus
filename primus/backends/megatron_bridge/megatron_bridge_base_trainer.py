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

from types import SimpleNamespace
from typing import Any

from primus.backends.megatron.training.global_vars import set_primus_global_variables
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
        - Handle Megatron-Bridge specific initialization and setup
    """

    def __init__(self, backend_args: Any):
        """
        Initialize Megatron-Bridge base trainer.

        Args:
            backend_args: Megatron-Bridge configuration as SimpleNamespace
                         (from MegatronBridgeArgBuilder)
        """
        log_rank_0("=" * 80)
        log_rank_0("Initializing MegatronBridgeBaseTrainer...")
        log_rank_0("=" * 80)

        # Initialize BaseTrainer
        super().__init__(backend_args=backend_args)
        set_primus_global_variables(self.backend_args)

        import primus.backends.megatron.patches  # noqa: F401

        # Create module_config from backend_args for patch context
        module_config = SimpleNamespace(params=self.backend_args)

        run_patches(
            backend="megatron",
            phase="before_train",
            backend_version=type(self).detect_megatron_version(),
            extra={
                "module_config": module_config,
                "backend_args": self.backend_args,
            },
        )

        log_rank_0("=" * 80)
        log_rank_0("MegatronBridgeBaseTrainer initialized successfully")
        log_rank_0("=" * 80)

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
