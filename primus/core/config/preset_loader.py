###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
from typing import Dict, Union

from primus.configs import models as MODELS_ROOT
from primus.core.config.merge_utils import deep_merge
from primus.core.config.yaml_loader import parse_yaml


class PresetLoader:
    """
    Load framework-aware presets (models, modules) with full extends support.

    Usage:
      preset = PresetLoader.load("llama2_7B", framework="megatron")
    """

    # file suffixes a preset name may carry; any other name gets .yaml
    _SUFFIXES = (".yaml", ".yml")

    @staticmethod
    def preset_path(name: Union[str, os.PathLike], framework: str, config_type: str = "models") -> str:
        """
        Absolute path of primus/configs/<config_type>/<framework>/<name>[.yaml] (the file ``load``
        reads; it may not exist). A name ending in .yaml or .yml is used as is; any other gets .yaml.
        Raises TypeError for a name that is not a str or str os.PathLike, and ValueError for a path
        outside primus/configs.
        """
        name = os.fspath(name) if isinstance(name, os.PathLike) else name
        if not isinstance(name, str):
            raise TypeError(
                f"[Primus] preset name must be a str or a str os.PathLike, not {type(name).__name__}"
            )
        suffix = next((s for s in PresetLoader._SUFFIXES if name.endswith(s)), None)
        filename = name if suffix else f"{name}.yaml"

        # Resolve base directory: primus/configs
        # MODELS_ROOT.__file__ -> primus/configs/models/__init__.py
        models_dir = os.path.dirname(MODELS_ROOT.__file__)
        configs_root = os.path.dirname(models_dir)

        preset_path = os.path.join(configs_root, config_type, framework, filename)
        # Validate that the path stays within configs_root
        abs_preset_path = os.path.abspath(preset_path)
        abs_configs_root = os.path.abspath(configs_root)
        if os.path.commonpath([abs_preset_path, abs_configs_root]) != abs_configs_root:
            raise ValueError(f"[Primus] Invalid preset path: path traversal detected for '{preset_path}'.")
        return abs_preset_path

    @staticmethod
    def load(name: Union[str, os.PathLike], framework: str, config_type: str = "models") -> Dict:
        """
        Load:
            primus/configs/<config_type>/<framework>/<name>[.yaml]

        And automatically resolve:
            - extends: [...]
            - nested extends
            - deep merge
            - env replacement
        """
        abs_preset_path = PresetLoader.preset_path(name, framework, config_type)

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
