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

from primus.backends.megatron_bridge.argument_builder import MegatronBridgeArgBuilder
from primus.core.backend.backend_adapter import BackendAdapter
from primus.modules.module_utils import log_dict_aligned, log_rank_0


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
        self.third_party_dir_name = "Megatron-Bridge"

    def load_trainer_class(self, stage: str = "pretrain"):
        """
        Return the Megatron-Bridge Trainer class for the specified training stage.

        Args:
            stage: Training stage ("sft" for supervised fine-tuning)

        Returns:
            Trainer class for the specified stage

        Raises:
            ValueError: If stage is not supported
        """
        if stage == "sft":
            from primus.backends.megatron_bridge.megatron_bridge_posttrain_trainer import (
                MegatronBridgePosttrainTrainer,
            )

            return MegatronBridgePosttrainTrainer
        else:
            raise ValueError(f"Invalid stage: {stage}")

    def setup_backend_path(self, backend_path=None) -> str:
        """
        Set up Megatron-Bridge backend path, then add additional paths.

        Megatron-Bridge uses a src-layout structure:
            third_party/
            └── Megatron-Bridge/
                ├── src/
                │   └── megatron/
                │       └── bridge/
                └── 3rdparty/
                    └── Megatron-LM/
                        └── megatron/

        We need to add:
        1. Megatron-Bridge root (via parent class)
        2. Megatron-Bridge/src/ for 'import megatron.bridge'
        3. Megatron-Bridge/3rdparty/Megatron-LM/ for base Megatron functionality
        """
        import os
        import sys

        # 1. Call parent to set up the main backend path
        resolved = super().setup_backend_path(backend_path)

        # 2. Add Megatron-Bridge src directory
        src_path = os.path.join(resolved, "src")
        if os.path.isdir(src_path) and src_path not in sys.path:
            sys.path.insert(0, src_path)
            log_rank_0(f"sys.path.insert → {src_path}")

        # 3. Add Megatron-LM directory (from megatron-bridge/3rdparty/)
        megatron_lm_path = os.path.join(resolved, "3rdparty", "Megatron-LM")
        if os.path.isdir(megatron_lm_path) and megatron_lm_path not in sys.path:
            sys.path.insert(0, megatron_lm_path)
            log_rank_0(f"sys.path.insert → {megatron_lm_path}")

        return resolved

    def convert_config(self, params: Any):
        """Convert Primus params to Megatron-Bridge argument Namespace."""
        builder = MegatronBridgeArgBuilder()
        builder.update(params)

        # Produce the final Megatron-Bridge Namespace
        bridge_args = builder.finalize()

        log_rank_0(
            f"[Primus:MegatronBridgeAdapter] Converted config → {len(vars(bridge_args))} Megatron-Bridge args"
        )

        log_dict_aligned("Megatron-Bridge args", bridge_args)

        return bridge_args

    def detect_backend_version(self) -> str:
        """Detect Megatron-Bridge version via AST parsing (avoids __init__.py execution)."""
        import ast
        import sys
        from pathlib import Path

        def parse_version(package_info_path: Path) -> str:
            tree = ast.parse(package_info_path.read_text())
            for node in tree.body:
                if isinstance(node, ast.Assign) and len(node.targets) == 1:
                    name = getattr(node.targets[0], "id", None)
                    if name == "__version__":
                        return ast.literal_eval(node.value)
            raise RuntimeError(f"__version__ not found in {package_info_path}")

        for path in sys.path:
            package_info_path = Path(path) / "megatron" / "bridge" / "package_info.py"
            if package_info_path.exists():
                return parse_version(package_info_path)

        raise RuntimeError("Cannot locate megatron/bridge/package_info.py in sys.path")
