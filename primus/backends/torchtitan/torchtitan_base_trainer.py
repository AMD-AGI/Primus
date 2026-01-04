###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
TorchTitanBaseTrainer: Base class for all TorchTitan trainers.

This mirrors the role of ``MegatronBaseTrainer`` for Megatron-LM:

    - Inherits from the unified ``BaseTrainer`` so it participates in the
      common training workflow and patch management (via ``run_patches``)
    - Provides a central place for TorchTitan-specific initialization logic
      and version detection
"""

from typing import Any

from primus.core.trainer.base_trainer import BaseTrainer
from primus.modules.module_utils import log_rank_0


class TorchTitanBaseTrainer(BaseTrainer):
    """
    Base trainer class for all TorchTitan training tasks.

    This class is intentionally lightweight compared to ``MegatronBaseTrainer``,
    because TorchTitan already consumes a fully-formed ``JobConfig`` without
    needing argparse patching.

    Responsibilities:
        - Call into the shared ``BaseTrainer`` to enable the unified workflow
          (before/after_train patches, lifecycle, logging)
        - Log TorchTitan metadata (version, model, framework)
        - Provide a classmethod ``detect_version`` used by the patch system
    """

    def __init__(self, primus_config: Any, module_config: Any, backend_args: Any):
        """
        Initialize TorchTitan base trainer.

        Args:
            primus_config: Full Primus configuration
            module_config: Module-specific configuration
            backend_args: TorchTitan configuration as SimpleNamespace (from TorchTitanAdapter)
        """
        # Patch TorchTitan logger before any other initialization
        self.patch_torchtitan_logger()

        log_rank_0("=" * 80)
        log_rank_0("Initializing TorchTitanBaseTrainer...")
        log_rank_0("=" * 80)

        # Initialize BaseTrainer (stores configs, enables patch management)
        super().__init__(
            primus_config=primus_config,
            module_config=module_config,
            backend_args=backend_args,
        )

        # Log version and basic metadata
        log_rank_0(f"TorchTitan version: {type(self).detect_version()}")
        log_rank_0(f"Model: {module_config.model}")
        log_rank_0(f"Framework: {module_config.framework}")

        log_rank_0("=" * 80)
        log_rank_0("TorchTitanBaseTrainer initialized successfully")
        log_rank_0("=" * 80)

    @classmethod
    def detect_version(cls) -> str:
        """
        Detect TorchTitan version.

        We try to resolve the installed ``torchtitan`` package version.
        If this fails (e.g., editable install without metadata), we fall
        back to "unknown".
        """
        try:
            import importlib.metadata as importlib_metadata
        except Exception:  # pragma: no cover - very old Python / fallback
            return "unknown"

        try:
            return importlib_metadata.version("torchtitan")
        except Exception:
            return "unknown"

    def patch_torchtitan_logger(self):
        from primus.core.utils.logger import _logger as primus_logger

        primus_logger.info("Monkey patch torchtitan logger...")

        import torchtitan.tools.logging as titan_logging

        titan_logging.logger = primus_logger
        titan_logging.init_logger = lambda: None
