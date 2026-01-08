###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
MaxText BackendAdapter implementation.

This is the MaxText counterpart of ``TorchTitanAdapter``. It is responsible for:

    - Preparing the MaxText/JAX backend environment
    - Converting Primus module config → MaxText PyConfig
    - Providing the MaxText trainer class to Primus
    - Exposing a backend version string for patching/diagnostics
"""

from __future__ import annotations

from typing import Any

# Trigger registration of all MaxText patches (logger, wandb, etc.)
import primus.backends.maxtext.patches  # noqa: F401
from primus.backends.maxtext.argument_builder import MaxTextConfigBuilder
from primus.core.backend.backend_adapter import BackendAdapter
from primus.core.backend.backend_registry import BackendRegistry
from primus.modules.module_utils import log_rank_0


class MaxTextAdapter(BackendAdapter):
    """
    Complete BackendAdapter implementation for MaxText.

    This adapter is designed to:
        - Integrate MaxText's PyConfig with Primus configs
        - Apply setup/build_args patches via the unified patch system
        - Load the appropriate MaxText trainer class
    """

    def __init__(self, framework: str = "maxtext"):
        super().__init__(framework)

    # Backend Setup & Patches
    def prepare_backend(self, config: Any):
        """
        MaxText-specific environment preparation.

        Steps:
            - Run Primus setup hooks
            - (Future) add any MaxText-specific env setup if needed

        Note: Patches are already registered at module import time (top of this file).
        Setup patches are applied automatically by the base class before this method is called.
        """
        # Run setup hooks from BackendRegistry
        BackendRegistry.run_setup("maxtext")

        log_rank_0("[Primus:MaxTextAdapter] Backend prepared")

    # Config → MaxText PyConfig
    def convert_config(self, module_config: Any):
        """
        Convert Primus ModuleConfig → MaxText configuration.

        This layer:
            - Takes module_config.params (which already includes CLI overrides)
            - Produces MaxText PyConfig format

        Note: build_args patches are applied automatically by the base class
        after this method returns.

        Args:
            module_config: ModuleConfig instance with params dict

        Returns:
            MaxText configuration object
        """
        # Instantiate the builder
        builder = MaxTextConfigBuilder()

        # Feed in config params (already merged with CLI overrides in train_launcher)
        builder.update(module_config.params)

        # Produce the final MaxText config
        maxtext_config = builder.finalize()

        log_rank_0(f"[Primus:MaxTextAdapter] Converted Primus module params → MaxText config")
        return maxtext_config

    # Load Trainer Class
    def load_trainer_class(self):
        """
        Load MaxText trainer class registered via BackendRegistry.

        This allows Primus runtime to remain agnostic to the actual trainer
        implementation (pretrain, sft, etc.).
        """
        try:
            return BackendRegistry.get_trainer_class(self.framework)
        except AssertionError as exc:
            raise RuntimeError(
                "[Primus:MaxTextAdapter] 'maxtext' backend trainer not registered. "
                "Ensure primus.backends.maxtext.trainers (or equivalent) registers "
                "the trainer class via BackendRegistry."
            ) from exc

    # Version Detection
    def detect_backend_version(self) -> str:
        """
        Detect MaxText version for logging and patching.

        MaxText typically doesn't have a version number, so we return a placeholder.
        """
        try:
            import MaxText

            if hasattr(MaxText, "__version__"):
                return MaxText.__version__
        except Exception:
            pass

        return "unknown"
