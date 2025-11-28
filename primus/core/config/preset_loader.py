###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
from typing import Dict

from primus.configs import models as MODELS_ROOT
from primus.core.config.merge_utils import deep_merge
from primus.core.config.yaml_loader import parse_yaml


class PresetLoader:
    """
    Load framework-aware presets (models, modules) with full extends support.

    Usage:
      preset = PresetLoader.load("llama2_7B", framework="megatron")
    """

    @staticmethod
    def load(name: str, framework: str, config_type: str = "models") -> Dict:
        """
        Load:
            primus/configs/<config_type>/<framework>/<name>[.yaml]

        And automatically resolve:
            - extends: [...]
            - nested extends
            - deep merge
            - env replacement
        """
        # Handle suffix
        if name.endswith(".yaml") or name.endswith(".yml"):
            filename = name
        else:
            filename = f"{name}.yaml"

        # Resolve base directory: primus/configs
        # MODELS_ROOT.__file__ -> primus/configs/models/__init__.py
        models_dir = os.path.dirname(MODELS_ROOT.__file__)
        configs_root = os.path.dirname(models_dir)

        preset_path = os.path.join(configs_root, config_type, framework, filename)
        # Validate that the path stays within configs_root
        abs_preset_path = os.path.abspath(preset_path)
        abs_configs_root = os.path.abspath(configs_root)
        if os.path.commonpath([abs_preset_path, abs_configs_root]) != abs_configs_root:
            raise ValueError(
                f"[Primus] Invalid preset path: path traversal detected for '{preset_path}'."
            )

        if not os.path.exists(abs_preset_path):
            raise FileNotFoundError(
                f"[Primus] Preset '{name}' not found for framework '{framework}' in '{config_type}'.\n"
                f"Expected: {abs_preset_path}"
            )

        preset = parse_yaml(abs_preset_path)

        return preset

    @staticmethod
    def merge_with_user_params(preset: Dict, params: Dict) -> Dict:
        """
        Combine:
            - model preset (base)
            - user params (override)
        """
        return deep_merge(preset, params)
