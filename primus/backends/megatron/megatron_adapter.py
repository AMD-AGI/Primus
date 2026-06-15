###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import primus.backends.megatron.patches  # noqa: F401  # Register patches
from primus.backends.megatron.argument_builder import MegatronArgBuilder
from primus.core.backend.backend_adapter import BackendAdapter
from primus.core.backend.backend_registry import BackendRegistry
from primus.modules.module_utils import log_rank_0


class MegatronAdapter(BackendAdapter):
    """BackendAdapter implementation for Megatron-LM."""

    def __init__(self, framework="megatron"):
        super().__init__(framework)
        self.third_party_dir_name = "Megatron-LM"

    def load_trainer_class(self, stage: str = "pretrain"):
        """Return the trainer class for the specified training stage."""
        try:
            trainer_cls = BackendRegistry.get_trainer_class(self.framework, stage=stage)
        except (ValueError, AssertionError) as exc:
            raise RuntimeError(
                "[Primus:MegatronAdapter] 'megatron' backend trainer not registered. "
                "Ensure primus.backends.megatron registers trainer classes via BackendRegistry."
            ) from exc

        log_rank_0(f"[Primus:MegatronAdapter] Loaded trainer class: {trainer_cls.__name__}")
        return trainer_cls

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
