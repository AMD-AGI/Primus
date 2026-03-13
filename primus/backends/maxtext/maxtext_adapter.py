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

import sys
from pathlib import Path
from typing import Any

# Trigger registration of all MaxText patches (logger, wandb, etc.)
from primus.backends.maxtext.argument_builder import MaxTextConfigBuilder
from primus.core.backend.backend_adapter import BackendAdapter
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
        self.third_party_dir_name = "maxtext"

    def setup_backend_path(self, backend_path=None) -> str:
        """
        Set up MaxText backend path.

        MaxText requires the 'src' subdirectory in sys.path after Dec Version.
        """
        # Call parent to set up the main backend path
        resolved = super().setup_backend_path(backend_path)
        src_path = Path(resolved) / "src"
        if src_path.exists() and str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
            log_rank_0(f"sys.path.insert → {src_path}")

        return resolved

    # Config → MaxText PyConfig
    def convert_config(self, params: Any):
        """
        Convert Primus params → MaxText configuration namespace.

        This layer:
            - Takes params (SimpleNamespace, which is module_config.params)
            - Produces a MaxText-compatible SimpleNamespace

        Note: build_args patches are applied automatically by the runtime
        after this method returns.

        Args:
            params: module_config.params (SimpleNamespace with all merged params)

        Returns:
            MaxText configuration SimpleNamespace
        """
        # Instantiate the builder
        builder = MaxTextConfigBuilder()

        # Feed in config params (already merged with CLI overrides in train_runtime)
        builder.update(params)

        # Produce the final MaxText config
        maxtext_config = builder.finalize()

        log_rank_0("[Primus:MaxTextAdapter] Converted Primus module params → MaxText config")
        return maxtext_config

    # Load Trainer Class
    def load_trainer_class(self, stage: str = "pretrain"):
        """
        This allows Primus runtime to remain agnostic to the actual trainer
        implementation (pretrain, sft, etc.).
        """
        if stage == "pretrain":
            from primus.backends.maxtext.maxtext_pretrain_trainer import (
                MaxTextPretrainTrainer,
            )

            return MaxTextPretrainTrainer
        else:
            raise ValueError(f"Invalid stage: {stage}")

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
