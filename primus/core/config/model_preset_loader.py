import os
from typing import Dict

from primus.configs import models as MODELS_ROOT
from primus.core.config.merge_utils import deep_merge
from primus.core.config.yaml_loader import parse_yaml


class ModelPresetLoader:
    """
    Load framework-aware model presets with full extends support.

    Usage:
      preset = ModelPresetLoader.load("llama2_7B", framework="megatron")
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

        if not os.path.exists(preset_path):
            raise FileNotFoundError(
                f"[Primus] Preset '{name}' not found for framework '{framework}' in '{config_type}'.\n"
                f"Expected: {preset_path}"
            )

        preset = parse_yaml(preset_path)

        return preset

    @staticmethod
    def merge_with_user_params(preset: Dict, params: Dict) -> Dict:
        """
        Combine:
            - model preset (base)
            - user params (override)
        """
        return deep_merge(preset, params)
