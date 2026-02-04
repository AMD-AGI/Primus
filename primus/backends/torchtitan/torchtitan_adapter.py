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
        self.third_party_dir_name = "torchtitan"

    def load_trainer_class(self, stage: str = "pretrain"):
        """
        Return backend Trainer class registered in `BackendRegistry`.

        Default behavior:
          - Lookup trainer via `BackendRegistry.get_trainer_class(self.framework, stage=stage)`
        """
        if stage == "pretrain":
            from primus.backends.torchtitan.torchtitan_pretrain_trainer import (
                TorchTitanPretrainTrainer,
            )

            return TorchTitanPretrainTrainer
        else:
            raise ValueError(f"Invalid stage: {stage}")

    # Version Detection
    def detect_backend_version(self) -> str:
        """
        Detect TorchTitan version for patching/diagnostics.

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
