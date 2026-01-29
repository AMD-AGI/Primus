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
from primus.core.backend.backend_registry import BackendRegistry
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

    # Backend-specific sys.path setup
    def setup_sys_path(self, backend_path: str):
        """
        Add Megatron-Bridge and Megatron-LM paths to sys.path.

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
        1. Megatron-Bridge/src/ for 'import megatron.bridge'
        2. Megatron-Bridge/3rdparty/Megatron-LM/ for base Megatron functionality
        """
        import os
        import sys

        # 1. Add Megatron-Bridge src directory
        src_path = os.path.join(backend_path, "src")
        if os.path.isdir(src_path) and src_path not in sys.path:
            sys.path.insert(0, src_path)
            log_rank_0(f"sys.path.insert → {src_path}")

        # 2. Add Megatron-LM directory (from megatron-bridge/3rdparty/)
        megatron_lm_path = os.path.join(backend_path, "3rdparty", "Megatron-LM")
        if os.path.isdir(megatron_lm_path) and megatron_lm_path not in sys.path:
            sys.path.insert(0, megatron_lm_path)
            log_rank_0(f"sys.path.insert → {megatron_lm_path}")

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

        log_dict_aligned("Megatron-Bridge args", bridge_args)

        return bridge_args

    # Load Trainer Class
    def load_trainer_class(self, stage: str = "pretrain"):
        """
        Load Megatron-Bridge trainer class registered via BackendRegistry.
        This allows Primus runtime to remain agnostic to the actual trainer
        implementation (pretrain, sft, etc.).
        """
        try:
            return BackendRegistry.get_trainer_class(self.framework, stage=stage)
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

        Note: We read the version file directly instead of importing to avoid
        triggering __init__.py, which imports all model classes (some may not
        be available in the current transformers version).

        Returns:
            Version string (e.g., "0.3.0rc0")

        Raises:
            RuntimeError: If version cannot be detected
        """
        import os
        import re
        import sys

        # Strategy 1: Read package_info.py directly (avoids __init__.py imports)
        for path in sys.path:
            package_info_file = os.path.join(path, "megatron", "bridge", "package_info.py")
            if os.path.exists(package_info_file):
                try:
                    with open(package_info_file, "r", encoding="utf-8") as f:
                        content = f.read()

                    # Match: __version__ = "x.y.z" or __version__ = 'x.y.z'
                    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
                    if match:
                        version = match.group(1)
                        log_rank_0(f"Detected Megatron-Bridge version: {version}")
                        return version

                except Exception as e:
                    log_rank_0(f"Warning: Failed to read {package_info_file}: {e}")
                    continue

        # Strategy 2: Try importing (may fail due to __init__.py model imports)
        try:
            from megatron.bridge.package_info import __version__

            log_rank_0(f"Detected Megatron-Bridge version: {__version__}")
            return __version__
        except ImportError as e:
            raise RuntimeError(
                f"Failed to detect Megatron-Bridge version. "
                f"package_info.py not found and import failed: {e}"
            )
