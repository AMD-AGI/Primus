###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MegatronBridgeBaseTrainer: Base class for all Megatron-Bridge trainers.

Responsibilities:
    - Inherits from BaseTrainer for common training workflow
    - Provides Megatron-Bridge-specific initialization and setup
    - Handles common logic shared across all Megatron-Bridge training tasks
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
        import primus.backends.megatron_bridge.patches  # noqa: F401

        # Create module_config from backend_args for patch context
        module_config = SimpleNamespace(params=self.backend_args)

        megatron_version = type(self).detect_megatron_version()
        patch_extra = {
            "module_config": module_config,
            "backend_args": self.backend_args,
        }

        run_patches(
            backend="megatron",
            phase="before_train",
            backend_version=megatron_version,
            extra=patch_extra,
        )

        run_patches(
            backend="megatron_bridge",
            phase="before_train",
            backend_version=megatron_version,
            extra=patch_extra,
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

    def _apply_nested_overrides(self) -> None:
        """Apply flat backend_args overrides to nested ConfigContainer fields.

        ConfigContainer uses nested dataclasses (train, logger, checkpoint, etc.)
        that cannot be reached by the flat _merge_dict_to_dataclass pass in
        load_recipe_config. This bridges user-facing YAML keys (e.g.
        ``log_interval: 99999``) to their nested targets.
        """
        args = self.backend_args
        cfg = self.cfg_container

        if hasattr(args, "log_interval"):
            val = getattr(args, "log_interval")
            if val is not None:
                cfg.logger.log_interval = int(val)
                log_rank_0(f"  ↳ Override logger.log_interval = {cfg.logger.log_interval}")

        for key in ("eval_interval", "eval_iters"):
            if hasattr(args, key):
                val = getattr(args, key)
                if val is not None:
                    setattr(cfg.train, key, int(val))
                    log_rank_0(f"  ↳ Override train.{key} = {val}")

        if hasattr(args, "save_interval"):
            val = getattr(args, "save_interval")
            if val is not None:
                cfg.checkpoint.save_interval = int(val)
                log_rank_0(f"  ↳ Override checkpoint.save_interval = {val}")

        if hasattr(args, "skip_save") and args.skip_save:
            cfg.checkpoint.save_interval = 0
            cfg.checkpoint.save = None
            log_rank_0("  ↳ Override checkpoint.save_interval = 0 (skip periodic save)")
            log_rank_0("  ↳ Override checkpoint.save = None (skip final save)")
