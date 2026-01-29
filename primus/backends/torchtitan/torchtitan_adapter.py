###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
TorchTitan BackendAdapter implementation.

This is the TorchTitan counterpart of ``MegatronAdapter``. It is responsible for:

    - Preparing the TorchTitan backend environment
    - Converting Primus module config → TorchTitan JobConfig
    - Providing the TorchTitan trainer class to Primus
    - Exposing a backend version string for patching/diagnostics
"""

from __future__ import annotations

from typing import Any

# Trigger registration of all TorchTitan patches (logger, turbo, model_override, etc.)
import primus.backends.torchtitan.patches  # noqa: F401
from primus.backends.torchtitan.argument_builder import TorchTitanJobConfigBuilder
from primus.core.backend.backend_adapter import BackendAdapter
from primus.core.backend.backend_registry import BackendRegistry
from primus.modules.module_utils import log_rank_0


class TorchTitanAdapter(BackendAdapter):
    """
    Complete BackendAdapter implementation for TorchTitan.

    This adapter is designed to:
        - Integrate TorchTitan's JobConfig with Primus configs
        - Apply setup/build_args patches via the unified patch system
        - Load the appropriate TorchTitan trainer class
    """

    def __init__(self, framework: str = "torchtitan"):
        super().__init__(framework)

    # Backend Setup & Patches
    def prepare_backend(self, config: Any):
        """
        TorchTitan-specific environment preparation.

        Steps:
            - Run Primus setup hooks
            - (Future) add any TorchTitan-specific env setup if needed

        Note: Patches are already registered at module import time (top of this file).
        Setup patches are applied automatically by the base class before this method is called.
        """
        # Run setup hooks from BackendRegistry
        BackendRegistry.run_setup("torchtitan")

        log_rank_0("[Primus:TorchTitanAdapter] Backend prepared")

    # Config → TorchTitan JobConfig
    def convert_config(self, module_config: Any):
        """
        Convert Primus ModuleConfig → TorchTitan configuration Namespace.

        This layer:
            - Takes module_config.params (which already includes CLI overrides)
            - Merges them into TorchTitan's default JobConfig
            - Produces a SimpleNamespace (consistent with MegatronAdapter)

        Note: build_args patches are applied automatically by the base class
        after this method returns.

        Args:
            module_config: ModuleConfig instance with params dict

        Returns:
            SimpleNamespace with TorchTitan configuration
        """
        # Instantiate the builder
        builder = TorchTitanJobConfigBuilder()

        # Feed in config params (already merged with CLI overrides in train_launcher)
        # module_config.params is a flat dict of TorchTitan-recognized fields.
        builder.update(module_config.params)

        # Produce the final TorchTitan Namespace (with distributed env injected)
        titan_args = builder.finalize()

        log_rank_0(f"[Primus:TorchTitanAdapter] Converted Primus module params → TorchTitan args")
        return titan_args

    # Load Trainer Class
    def load_trainer_class(self, module_config=None):
        """
        Load TorchTitan trainer class registered via BackendRegistry.

        Args:
            module_config: Module configuration (optional, for stage detection)

        This allows Primus runtime to remain agnostic to the actual trainer
        implementation (pretrain, sft, etc.).
        """
        try:
            return BackendRegistry.get_trainer_class(self.framework)
        except (ValueError, AssertionError) as exc:
            raise RuntimeError(
                "[Primus:TorchTitanAdapter] 'torchtitan' backend trainer not registered. "
                "Ensure primus.backends.torchtitan.trainers (or equivalent) registers "
                "the trainer class via BackendRegistry."
            ) from exc

    # Version Detection
    def detect_backend_version(self) -> str:
        """
        Detect TorchTitan version for logging and patching.

        We try to read the ``torchtitan`` package version if installed; otherwise
        we fall back to 'unknown'. This is sufficient until we need stricter
        version gating for patches.
        """
        try:
            import importlib.metadata as importlib_metadata
        except Exception:  # pragma: no cover - very old Python
            return "unknown"

        try:
            return importlib_metadata.version("torchtitan")
        except Exception:
            return "unknown"
