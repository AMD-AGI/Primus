###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


# Trigger registration of all Megatron patches (args_patches, env_patches, etc.)
import primus.backends.megatron.patches  # noqa: F401
from primus.backends.megatron.argument_builder import MegatronArgBuilder
from primus.core.backend.backend_adapter import BackendAdapter
from primus.modules.module_utils import log_rank_0


class MegatronAdapter(BackendAdapter):
    """
    The complete BackendAdapter implementation for Megatron-LM.

    This adapter is designed to:
        - Handle multi-version Megatron differences
        - Convert Primus config → Megatron args using ArgBuilder
        - Apply patches automatically (PR fixes, kernel bugs, attention fixes)
        - Load the appropriate Trainer class depending on Megatron version
    """

    def __init__(self, framework="megatron"):
        super().__init__(framework)
        self.third_party_dir_name = "Megatron-LM"

    def load_trainer_class(self, stage: str = "pretrain"):
        """
        Return the Megatron Trainer class for the specified training stage.

        Args:
            stage: Training stage ("pretrain" for pre-training)

        Returns:
            Trainer class for the specified stage

        Raises:
            ValueError: If stage is not supported
        """
        if stage == "pretrain":
            from primus.backends.megatron.megatron_pretrain_trainer import (
                MegatronPretrainTrainer,
            )

            return MegatronPretrainTrainer
        else:
            raise ValueError(f"Invalid stage: {stage}")

    def detect_backend_version(self) -> str:
        """
        Detect Megatron-LM version.

        Returns:
            Version string (e.g., "0.15.0rc8")

        Raises:
            RuntimeError: If version cannot be detected
        """
        import ast
        from pathlib import Path

        def get_megatron_version_str(package_info_path: str | Path) -> str:
            """
            Return version string equivalent to package_info.__version__
            without importing megatron.

            Example: '0.15.0rc8'
            """
            path = Path(package_info_path)
            if not path.exists():
                raise RuntimeError(f"{path} does not exist")

            tree = ast.parse(path.read_text())

            values = {}
            for node in tree.body:
                if isinstance(node, ast.Assign) and len(node.targets) == 1:
                    name = getattr(node.targets[0], "id", None)
                    if name in {"MAJOR", "MINOR", "PATCH", "PRE_RELEASE"}:
                        values[name] = ast.literal_eval(node.value)
            major = values["MAJOR"]
            minor = values["MINOR"]
            patch = values["PATCH"]
            pre = values.get("PRE_RELEASE")

            return f"{major}.{minor}.{patch}" + (str(pre) if pre else "")

        # Scan sys.path manually to avoid triggering any __init__.py execution
        # (importlib.util.find_spec would execute parent package __init__.py files)
        import sys

        for path in sys.path:
            package_info_path = Path(path) / "megatron" / "core" / "package_info.py"
            if package_info_path.exists():
                return get_megatron_version_str(package_info_path)

        raise RuntimeError("Cannot locate megatron/core/package_info.py in sys.path")

    def convert_config(self, module_config):
        """
        Convert Primus ModuleConfig → final Megatron-LM argument Namespace.

        This layer:
            - Takes module_config.params (which already includes CLI overrides)
            - Fills missing fields using Megatron-LM defaults
            - Injects distributed environment variables (via builder)
            - Produces a Megatron-compatible argparse-like Namespace

        Note: build_args patches are applied automatically by the base class
        after this method returns.

        Args:
            module_config: ModuleConfig instance with params dict

        Returns:
            SimpleNamespace with Megatron args
        """
        # Instantiate the builder
        builder = MegatronArgBuilder()

        # Feed in config params (already merged with CLI overrides in train_launcher)
        # module_config.params is a flat dict of Megatron-recognized fields.
        builder.update(module_config.params)

        # Produce the final Megatron Namespace (with distributed env injected)
        megatron_args = builder.finalize()

        log_rank_0(f"[Primus:MegatronAdapter] Converted config → {len(vars(megatron_args))} Megatron args")

        return megatron_args
