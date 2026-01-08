###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from dataclasses import asdict, is_dataclass
from types import SimpleNamespace
from typing import Any, Dict

# Trigger registration of all TorchTitan patches
import primus.backends.torchtitan.patches  # noqa: F401
from primus.core.patches import run_patches
from primus.core.utils.yaml_utils import nested_namespace_to_dict
from primus.modules.base_module import BaseModule


class TorchTitanPretrainTrainer(BaseModule):
    def __init__(self, *args, **kwargs):
        extra_args = kwargs.pop("extra_args", None)
        super().__init__(*args, **kwargs)

        self.primus_cfg = kwargs.pop("primus_config", None)
        if self.primus_cfg is None:
            raise ValueError("primus_config is required")

        pre_trainer_cfg = self.primus_cfg.get_module_config("pre_trainer")
        cfg_dict = nested_namespace_to_dict(pre_trainer_cfg)

        # Run before_train phase patches (e.g., Primus-Turbo patches)
        # Construct a temporary module_config for patch context
        temp_module_config = SimpleNamespace()
        temp_module_config.params = pre_trainer_cfg
        temp_module_config.name = "pre_trainer"
        temp_module_config.framework = "torchtitan"
        temp_module_config.model = getattr(pre_trainer_cfg, "model.name", None)

        # Merge extra_args into temp_module_config.params if provided
        if extra_args:
            temp_module_config.params = self._merge_extra_args_into_params(
                temp_module_config.params, extra_args
            )

        run_patches(
            backend="torchtitan",
            phase="setup",
            backend_version="unknown",
            extra={
                "module_config": temp_module_config,
            },
        )

        from torchtitan.config.job_config import JobConfig
        from torchtitan.train import Trainer

        self.TrainerClass = Trainer
        self.JobConfigClass = JobConfig

        self.titan_config = self.build_job_config(cfg_dict, self.JobConfigClass)

        self.log_config(self.titan_config)
        self.trainer = None

    def _merge_extra_args_into_params(
        self, params: SimpleNamespace, extra_args: Dict[str, Any]
    ) -> SimpleNamespace:
        """
        Merge extra_args into params with specific rules:

        1. Non-model.* parameters: Only merge if they already exist in params
        2. model.* parameters: Extract and merge (can add new fields)

        Args:
            params: Original params namespace
            extra_args: Additional arguments to merge

        Returns:
            Updated params with merged extra_args
        """
        from primus.modules.module_utils import log_rank_0

        # First, flatten extra_args if it contains nested dicts
        flattened_extra_args = self._flatten_dict(extra_args)

        # Separate model.* and non-model.* parameters
        model_params = {}
        non_model_params = {}

        for key, value in flattened_extra_args.items():
            if key.startswith("model."):
                # Extract model parameter (remove "model." prefix)
                model_key = key[6:]  # Remove "model." prefix
                model_params[model_key] = value
            else:
                non_model_params[key] = value

        # Convert params to dict for easier manipulation
        params_dict = nested_namespace_to_dict(params)

        # Merge non-model.* parameters (only if they already exist)
        for key, value in non_model_params.items():
            if self._key_exists_in_dict(params_dict, key):
                self._set_nested_dict_value(params_dict, key, value)
                log_rank_0(f"[ExtraArgs] Merged non-model param: {key} = {value}")
            else:
                log_rank_0(f"[ExtraArgs] Skipped non-model param (not in params): {key} = {value}")

        # Merge model.* parameters (can add new fields)
        if model_params:
            if "model" not in params_dict:
                params_dict["model"] = {}

            for key, value in model_params.items():
                self._set_nested_dict_value(params_dict, f"model.{key}", value)
                log_rank_0(f"[ExtraArgs] Merged model param: model.{key} = {value}")

        # Convert back to SimpleNamespace
        return self._dict_to_namespace(params_dict)

    def _flatten_dict(self, d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Flatten nested dictionary using dot notation."""
        result = {}

        for key, value in d.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                # Recursively flatten nested dict
                result.update(self._flatten_dict(value, full_key))
            else:
                # Leaf value
                result[full_key] = value

        return result

    def _key_exists_in_dict(self, d: Dict[str, Any], key: str) -> bool:
        """Check if a nested key exists in dictionary (supports dot notation)."""
        parts = key.split(".")
        current = d

        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return False
            current = current[part]

        return True

    def _set_nested_dict_value(self, d: Dict[str, Any], key: str, value: Any) -> None:
        """Set a nested dictionary value using dot notation."""
        parts = key.split(".")
        current = d

        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[parts[-1]] = value

    def _dict_to_namespace(self, d: Dict[str, Any]) -> SimpleNamespace:
        """Recursively convert dictionary to SimpleNamespace."""
        if isinstance(d, dict):
            return SimpleNamespace(**{k: self._dict_to_namespace(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [self._dict_to_namespace(item) for item in d]
        else:
            return d

        self.patch_classic_attention()

    def setup(self):
        pass

    def init(self, *init_args, **kwargs):
        self.trainer = self.TrainerClass(self.titan_config)

    def run(self, *args, **kwargs):
        if self.trainer is None:
            raise RuntimeError("Trainer has not been initialized. Call init() first.")
        self.trainer.train()

    def flatten_config(self, obj: Any, prefix: str = "") -> Dict[str, Any]:
        flat_dict = {}
        if is_dataclass(obj):
            obj = asdict(obj)

        if isinstance(obj, dict):
            for key, value in obj.items():
                full_key = f"{prefix}.{key}" if prefix else key
                if is_dataclass(value) or isinstance(value, dict):
                    flat_dict.update(self.flatten_config(value, full_key))
                else:
                    flat_dict[full_key] = value
        else:
            flat_dict[prefix] = obj

        return flat_dict

    def log_config(self, obj: Any, header: str = "TorchTitan Config"):
        from torchtitan.tools.logging import logger

        logger.info("========== %s ==========" % header)
        flat = self.flatten_config(obj)
        max_key_len = max(len(k) for k in flat.keys())
        for key in sorted(flat):
            val = flat[key]
            formatted_line = f"arguments {key.ljust(max_key_len, '.')} {val}"
            logger.info(formatted_line)

    def build_job_config(self, cfg_dict: dict, JobConfigType) -> Any:
        import importlib

        from torchtitan.config.job_config import Experimental
        from torchtitan.tools.logging import logger

        # Step 1: Parse the experimental section to check for a custom JobConfig extension
        experimental_cfg = cfg_dict.get("experimental", {})
        experimental = Experimental(**experimental_cfg)

        # Step 2: If a custom_args_module is defined, import and merge with JobConfig
        custom_job_config_cls = JobConfigType
        if experimental and getattr(experimental, "custom_args_module", None):
            try:
                module = importlib.import_module(experimental.custom_args_module)
                ExtendedJobConfig = getattr(module, "JobConfig")
                custom_job_config_cls = self.merge_configs(JobConfigType, ExtendedJobConfig)
                logger.info(f"Loaded and merged custom JobConfig from {experimental.custom_args_module}")
            except Exception as e:
                logger.warning(f"Failed to load custom_args_module '{experimental.custom_args_module}': {e}")

        # Step 3: Parse config dict (including custom fields) into dataclass recursively
        return self._dict_to_dataclass(custom_job_config_cls, cfg_dict)

    @staticmethod
    def merge_configs(base_cls, custom_cls):
        """
        Merges two dataclass types into one unified dataclass.

        Merge logic:
        - If a field exists in both:
            - If both fields are dataclasses, recursively merge them.
            - Otherwise, the custom field overrides the base.
        - Fields only in base or only in custom are included as-is.
        """
        from dataclasses import field, fields, make_dataclass

        base_fields = {f.name: f for f in fields(base_cls)}
        custom_fields = {f.name: f for f in fields(custom_cls)}

        merged = []

        # Merge overlapping and base-only fields
        for name, base_f in base_fields.items():
            if name in custom_fields:
                custom_f = custom_fields[name]
                if is_dataclass(base_f.type) and is_dataclass(custom_f.type):
                    merged_type = TorchTitanPretrainTrainer.merge_configs(base_f.type, custom_f.type)
                    merged.append((name, merged_type, field(default_factory=merged_type)))
                else:
                    merged.append((name, custom_f.type, custom_f))
            else:
                merged.append((name, base_f.type, base_f))

        # Add custom-only fields
        for name, custom_f in custom_fields.items():
            if name not in base_fields:
                merged.append((name, custom_f.type, custom_f))

        return make_dataclass(f"Merged{base_cls.__name__}", merged, bases=(base_cls,))

    def _dict_to_dataclass(self, cls, data: dict[str, Any]) -> Any:
        """Recursively convert dictionary to dataclass, handling nested and custom fields."""
        from dataclasses import fields, is_dataclass

        if not is_dataclass(cls):
            return data

        # collect valid field names
        field_names = {f.name for f in fields(cls)}
        init_values = {}

        # only use known fields for constructor
        for f in fields(cls):
            if f.name in data:
                val = data[f.name]
                if is_dataclass(f.type) and isinstance(val, dict):
                    init_values[f.name] = self._dict_to_dataclass(f.type, val)
                else:
                    init_values[f.name] = val

        # instantiate dataclass
        obj = cls(**init_values)

        # attach unknown fields dynamically
        for k, v in data.items():
            if k not in field_names:
                setattr(obj, k, v)

        return obj
