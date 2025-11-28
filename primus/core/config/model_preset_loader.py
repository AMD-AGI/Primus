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
    def load(model_name: str, framework: str) -> Dict:
        """
        Load:
            primus/configs/models/<framework>/<model_name>.yaml

        And automatically resolve:
            - extends: [...]
            - nested extends
            - deep merge
            - env replacement
        """
        base_dir = os.path.dirname(MODELS_ROOT.__file__)
        preset_path = os.path.join(base_dir, framework, f"{model_name}.yaml")

        if not os.path.exists(preset_path):
            raise FileNotFoundError(
                f"[Primus] Model preset '{model_name}' not found for framework '{framework}'.\n"
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
