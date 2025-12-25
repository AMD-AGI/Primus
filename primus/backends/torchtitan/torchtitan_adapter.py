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

from primus.backends.torchtitan.argument_builder import TorchTitanJobConfigBuilder
from primus.core.backend.backend_adapter import BackendAdapter
from primus.core.backend.backend_registry import BackendRegistry
from primus.core.utils.yaml_utils import nested_namespace_to_dict
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

    # 1. Backend Setup & Patches
    def prepare_backend(self, config: Any):
        """
        TorchTitan-specific environment preparation.

        Steps:
            - Run Primus setup hooks
            - (Future) add any TorchTitan-specific env setup if needed

        Note: setup patches are applied automatically by the base class
        before this method is called.
        """
        # Run setup hooks from BackendRegistry
        BackendRegistry.run_setup("torchtitan")

        log_rank_0("[Primus:TorchTitanAdapter] Backend prepared")

    # 2. Config → TorchTitan JobConfig
    def convert_config(self, module_config: Any):
        """
        Convert Primus ModuleConfig → TorchTitan JobConfig.

        This layer:
            - Takes module_config.params (which already includes CLI overrides)
            - Merges them into TorchTitan's default JobConfig
            - Produces a fully-populated JobConfig dataclass for Titan

        Note: build_args patches are applied automatically by the base class
        after this method returns.
        """
        # Convert Primus module params into a nested dict
        params_ns = getattr(module_config, "params", None)
        cfg_dict = nested_namespace_to_dict(params_ns) if params_ns is not None else {}

        # Build final TorchTitan JobConfig
        builder = TorchTitanJobConfigBuilder()
        builder.update(cfg_dict)
        job_cfg = builder.to_job_config()

        log_rank_0("[Primus:TorchTitanAdapter] Converted Primus module params → TorchTitan JobConfig")

        return job_cfg

    # 3. Load Trainer Class
    def load_trainer_class(self):
        """
        Load TorchTitan trainer class registered via BackendRegistry.

        This allows Primus runtime to remain agnostic to the actual trainer
        implementation (pretrain, sft, etc.).
        """
        try:
            return BackendRegistry.get_trainer_class(self.framework)
        except AssertionError as exc:
            raise RuntimeError(
                "[Primus:TorchTitanAdapter] 'torchtitan' backend trainer not registered. "
                "Ensure primus.backends.torchtitan.trainers (or equivalent) registers "
                "the trainer class via BackendRegistry."
            ) from exc

    # 4. Version Detection
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
