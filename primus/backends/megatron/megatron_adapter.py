###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import importlib
from typing import Optional

import primus.backends.megatron.patches  # noqa: F401  # Register patches
from primus.backends.megatron.argument_builder import MegatronArgBuilder
from primus.core.backend.backend_adapter import BackendAdapter
from primus.core.backend.backend_registry import BackendRegistry
from primus.core.utils.module_utils import log_rank_0


class MegatronAdapter(BackendAdapter):
    """BackendAdapter implementation for Megatron-LM."""

    def __init__(self, framework="megatron"):
        super().__init__(framework)
        self.third_party_dir_name = "Megatron-LM"

    def load_trainer_class(self, stage: str = "pretrain", trainer_class: Optional[str] = None):
        """
        Return the Trainer class for the specified training stage or trainer class name.

        Args:
            stage: Training stage (e.g., "pretrain", "sft"). Defaults to "pretrain".
            trainer_class: Optional specific trainer class name for dynamic loading.
                          If provided, this takes precedence over stage-based selection.

        Returns:
            Trainer class

        Raises:
            ImportError: If trainer class cannot be imported
            ValueError: If stage is invalid or trainer class is not found
        """
        # If trainer_class is specified, attempt dynamic loading
        if trainer_class:
            return self._load_trainer_class_by_name(trainer_class)

        # Fallback to stage-based selection via the backend registry
        try:
            trainer_cls = BackendRegistry.get_trainer_class(self.framework, stage=stage)
        except (ValueError, AssertionError) as exc:
            raise RuntimeError(
                "[Primus:MegatronAdapter] 'megatron' backend trainer not registered. "
                "Ensure primus.backends.megatron registers trainer classes via BackendRegistry."
            ) from exc

        log_rank_0(f"[Primus:MegatronAdapter] Loaded trainer class: {trainer_cls.__name__}")
        return trainer_cls

    def _load_trainer_class_by_name(self, trainer_class: str):
        """
        Dynamically load trainer class by name.

        Args:
            trainer_class: Trainer class name (e.g., "FluxPretrainTrainer", "MegatronPretrainTrainer")

        Returns:
            Trainer class

        Raises:
            ImportError: If trainer class cannot be imported
            ValueError: If trainer class is not found
        """
        # Define trainer registry for Megatron backend
        # This maps trainer class names to their module paths
        MEGATRON_TRAINERS = {
            "MegatronPretrainTrainer": "primus.backends.megatron.megatron_pretrain_trainer.MegatronPretrainTrainer",
            "FluxPretrainTrainer": "primus.backends.megatron.flux_pretrain_trainer.FluxPretrainTrainer",
        }

        if trainer_class not in MEGATRON_TRAINERS:
            # Try to load from common locations as fallback
            log_rank_0(
                f"[Primus:MegatronAdapter] Trainer '{trainer_class}' not in registry, attempting dynamic import..."
            )

            possible_paths = [
                f"primus.backends.megatron.{trainer_class.lower()}.{trainer_class}",
                f"primus.backends.megatron.{trainer_class}",
            ]

            for module_path in possible_paths:
                try:
                    module_name, class_name = module_path.rsplit(".", 1)
                    module = importlib.import_module(module_name)
                    trainer_cls = getattr(module, class_name)
                    log_rank_0(
                        f"[Primus:MegatronAdapter] Successfully loaded trainer: {trainer_class} from {module_name}"
                    )
                    return trainer_cls
                except (ImportError, AttributeError):
                    continue

            # If all attempts failed, provide helpful error message
            available = list(MEGATRON_TRAINERS.keys())
            raise ValueError(
                f"Trainer class '{trainer_class}' not found.\n"
                f"Available trainers: {', '.join(available) if available else 'none'}\n"
                f"Hint: Set 'trainer_class' in your experiment YAML to a registered trainer "
                f"(e.g. trainer_class: FluxPretrainTrainer), register it in MEGATRON_TRAINERS, "
                f"or ensure it is importable from standard locations."
            )

        # Load from registry
        trainer_path = MEGATRON_TRAINERS[trainer_class]
        module_path, class_name = trainer_path.rsplit(".", 1)

        try:
            module = importlib.import_module(module_path)
            trainer_cls = getattr(module, class_name)
            log_rank_0(f"[Primus:MegatronAdapter] Loaded trainer: {trainer_class} from {module_path}")
            return trainer_cls
        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"Failed to load trainer class '{trainer_class}' from '{module_path}': {e}\n"
                f"Hint: Check that the module exists and the class name is correct."
            ) from e

    def detect_backend_version(self) -> str:
        """Detect Megatron-LM version via AST parsing (avoids __init__.py execution)."""
        import ast
        import sys
        from pathlib import Path

        def parse_version(package_info_path: Path) -> str:
            tree = ast.parse(package_info_path.read_text())
            values = {}
            for node in tree.body:
                if isinstance(node, ast.Assign) and len(node.targets) == 1:
                    name = getattr(node.targets[0], "id", None)
                    if name in {"MAJOR", "MINOR", "PATCH", "PRE_RELEASE"}:
                        values[name] = ast.literal_eval(node.value)
            pre = values.get("PRE_RELEASE")
            return f"{values['MAJOR']}.{values['MINOR']}.{values['PATCH']}" + (str(pre) if pre else "")

        for path in sys.path:
            package_info_path = Path(path) / "megatron" / "core" / "package_info.py"
            if package_info_path.exists():
                return parse_version(package_info_path)

        raise RuntimeError("Cannot locate megatron/core/package_info.py in sys.path")

    def convert_config(self, params):
        """Convert Primus params to Megatron-LM argument Namespace."""
        builder = MegatronArgBuilder()
        builder.update(params)
        megatron_args = builder.finalize()
        log_rank_0(f"[Primus:MegatronAdapter] Converted config → {len(vars(megatron_args))} Megatron args")
        return megatron_args
