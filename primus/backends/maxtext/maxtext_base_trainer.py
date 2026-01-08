###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
MaxTextBaseTrainer: Base class for all MaxText trainers.

This mirrors the role of ``TorchTitanBaseTrainer``:

    - Inherits from the unified ``BaseTrainer`` so it participates in the
      common training workflow and patch management (via ``run_patches``)
    - Provides a central place for MaxText-specific initialization logic
      and version detection
"""

from typing import Any

from primus.core.trainer.base_trainer import BaseTrainer
from primus.modules.module_utils import log_rank_0


class MaxTextBaseTrainer(BaseTrainer):
    """
    Base trainer class for all MaxText training tasks.

    Responsibilities:
        - Call into the shared ``BaseTrainer`` to enable the unified workflow
          (before/after_train patches, lifecycle, logging)
        - Log MaxText metadata (version, model, framework)
        - Provide a classmethod ``detect_version`` used by the patch system
    """

    def __init__(self, primus_config: Any, module_config: Any, backend_args: Any):
        """
        Initialize MaxText base trainer.

        Args:
            primus_config: Full Primus configuration
            module_config: Module-specific configuration
            backend_args: MaxText configuration (from MaxTextAdapter)
        """
        log_rank_0("=" * 80)
        log_rank_0("Initializing MaxTextBaseTrainer...")
        log_rank_0("=" * 80)

        # Initialize BaseTrainer (stores configs, enables patch management)
        super().__init__(
            primus_config=primus_config,
            module_config=module_config,
            backend_args=backend_args,
        )

        # Log version and basic metadata
        log_rank_0(f"MaxText version: {type(self).detect_version()}")
        log_rank_0(f"Model: {module_config.model}")
        log_rank_0(f"Framework: {module_config.framework}")

        log_rank_0("=" * 80)
        log_rank_0("MaxTextBaseTrainer initialized successfully")
        log_rank_0("=" * 80)

    @classmethod
    def detect_version(cls) -> str:
        """
        Detect MaxText version.

        MaxText typically doesn't have a version number, so we return a placeholder.
        """
        try:
            import MaxText

            if hasattr(MaxText, "__version__"):
                return MaxText.__version__
        except Exception:
            pass

        return "unknown"
