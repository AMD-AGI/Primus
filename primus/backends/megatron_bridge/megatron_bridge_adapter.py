###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MegatronBridge BackendAdapter implementation.

This is the Megatron-Bridge counterpart of ``MegatronAdapter``. It is responsible for:

    - Preparing the Megatron-Bridge backend environment
    - Converting Primus module config → Megatron-Bridge configuration
    - Providing the Megatron-Bridge trainer class to Primus
    - Exposing a backend version string for patching/diagnostics
"""

from __future__ import annotations

from typing import Any

# Trigger registration of all Megatron-Bridge patches
import primus.backends.megatron_bridge.patches  # noqa: F401
from primus.backends.megatron_bridge.argument_builder import MegatronBridgeArgBuilder
from primus.core.backend.backend_adapter import BackendAdapter
from primus.core.backend.backend_registry import BackendRegistry
from primus.modules.module_utils import log_rank_0


class MegatronBridgeAdapter(BackendAdapter):
    """
    Complete BackendAdapter implementation for Megatron-Bridge.

    This adapter is designed to:
        - Integrate Megatron-Bridge's configuration with Primus configs
        - Apply setup/build_args patches via the unified patch system
        - Load the appropriate Megatron-Bridge trainer class
        - Handle bidirectional Hugging Face conversion capabilities
    """

    def __init__(self, framework: str = "megatron_bridge"):
        super().__init__(framework)

    # Backend Setup & Patches
    def prepare_backend(self, config: Any):
        """
        Megatron-Bridge-specific environment preparation.

        Steps:
            - Run Primus setup hooks
            - Set up Megatron-Bridge specific environment variables
            - Initialize Hugging Face model conversion capabilities

        Note: Patches are already registered at module import time.
        Setup patches are applied automatically by the base class before this method is called.
        """
        # Run setup hooks from BackendRegistry
        BackendRegistry.run_setup("megatron_bridge")

        log_rank_0("[Primus:MegatronBridgeAdapter] Backend prepared")

    # Config → Megatron-Bridge Args
    def convert_config(self, module_config: Any):
        """
        Convert Primus ModuleConfig → Megatron-Bridge configuration Namespace.

        This layer:
            - Takes module_config.params (which already includes CLI overrides)
            - Merges them into Megatron-Bridge's configuration
            - Produces a SimpleNamespace (consistent with MegatronAdapter)
            - Handles recipe-based configuration if specified

        Note: build_args patches are applied automatically by the base class
        after this method returns.

        Args:
            module_config: ModuleConfig instance with params dict

        Returns:
            SimpleNamespace with Megatron-Bridge configuration
        """
        # Instantiate the builder
        builder = MegatronBridgeArgBuilder()

        # Feed in config params (already merged with CLI overrides in train_launcher)
        # module_config.params is a flat dict of Megatron-Bridge-recognized fields.
        builder.update(module_config.params)

        # Produce the final Megatron-Bridge Namespace
        bridge_args = builder.finalize()

        log_rank_0(
            f"[Primus:MegatronBridgeAdapter] Converted config → {len(vars(bridge_args))} Megatron-Bridge args"
        )

        return bridge_args

    # Load Trainer Class
    def load_trainer_class(self):
        """
        Load Megatron-Bridge trainer class registered via BackendRegistry.

        This allows Primus runtime to remain agnostic to the actual trainer
        implementation (pretrain, sft, etc.).
        """
        try:
            return BackendRegistry.get_trainer_class(self.framework)
        except ValueError as exc:
            raise RuntimeError(
                "[Primus:MegatronBridgeAdapter] 'megatron_bridge' backend trainer not registered. "
                "Ensure primus.backends.megatron_bridge defines the trainer class "
                "and imports BackendRegistry."
            ) from exc

    # Version Detection
    def detect_backend_version(self) -> str:
        """
        Detect Megatron-Bridge version for logging and patching.

        Returns:
            Version string (e.g., "0.2.2")

        Raises:
            RuntimeError: If version cannot be detected
        """
        # Get trainer class and call its detect_version classmethod
        TrainerClass = self.load_trainer_class()
        return TrainerClass.detect_version()
