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

        # Log version and basic metadata
        log_rank_0(f"Megatron-Bridge version: {type(self).detect_version()}")
        log_rank_0(f"Model: {module_config.model or 'custom'}")
        log_rank_0(f"Framework: {module_config.framework}")

        # Log task type if available
        task_type = getattr(self, "TASK_TYPE", "unknown")
        if task_type != "unknown":
            log_rank_0(f"Task: {task_type}")

        # Check for recipe-based configuration
        if hasattr(backend_args, "recipe") and backend_args.recipe:
            log_rank_0(f"Using recipe: {backend_args.recipe}")

        # Check for LoRA configuration
        if hasattr(backend_args, "use_lora") and backend_args.use_lora:
            log_rank_0("LoRA fine-tuning enabled")
            if hasattr(backend_args, "lora_rank"):
                log_rank_0(f"  LoRA rank: {backend_args.lora_rank}")
            if hasattr(backend_args, "lora_alpha"):
                log_rank_0(f"  LoRA alpha: {backend_args.lora_alpha}")

        # Check for HuggingFace conversion
        if hasattr(backend_args, "convert_from_hf") and backend_args.convert_from_hf:
            log_rank_0("HuggingFace model conversion: enabled")
            if hasattr(backend_args, "hf_model_name_or_path"):
                log_rank_0(f"  Model: {backend_args.hf_model_name_or_path}")

        log_rank_0("=" * 80)
        log_rank_0("MegatronBridgeBaseTrainer initialized successfully")
        log_rank_0("=" * 80)

    @classmethod
    def detect_version(cls) -> str:
        """
        Detect Megatron-Bridge version.

        Returns:
            Version string (e.g., "0.3.0rc0") or "unknown" if detection fails
        """
        try:
            from megatron.bridge.package_info import __version__

            return __version__
        except ImportError:
            log_rank_0(
                "Warning: Could not detect Megatron-Bridge version from package_info"
            )
            return "unknown"
