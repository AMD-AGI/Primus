###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
TorchTitanPretrainTrainer: Primus wrapper for TorchTitan pre-training.

This trainer bridges Primus's configuration system with TorchTitan's training
loop, following the same pattern as ``MegatronPretrainTrainer`` does for
Megatron-LM.

The trainer inherits from ``TorchTitanBaseTrainer`` which handles:
    - Integration with the unified BaseTrainer workflow (run_patches)
    - Version detection and common logging

This class only needs to implement:
    - setup(): optional pre-initialization
    - init(): construct the underlying TorchTitan Trainer
    - run_train(): call into TorchTitan's training loop
"""

from types import SimpleNamespace
from typing import Any, Optional

from primus.backends.torchtitan.torchtitan_base_trainer import TorchTitanBaseTrainer
from primus.modules.module_utils import log_rank_0


class TorchTitanPretrainTrainer(TorchTitanBaseTrainer):
    """
    Trainer class for TorchTitan pre-training.
    """

    def __init__(self, primus_config: Any, module_config: Any, backend_args: Any):
        """
        Initialize TorchTitan pretrain trainer.

        Args:
            primus_config: Full Primus configuration
            module_config: Module-specific configuration
            backend_args: TorchTitan configuration as SimpleNamespace (from TorchTitanAdapter)
        """
        super().__init__(
            primus_config=primus_config,
            module_config=module_config,
            backend_args=backend_args,
        )

        self._trainer: Optional["Trainer"] = None  # type: ignore[name-defined]

        log_rank_0(f"Initialized TorchTitanPretrainTrainer for model: {module_config.model or 'custom'}")

    # --------------------------------------------------------------------- #
    # Lifecycle hooks
    # --------------------------------------------------------------------- #

    def setup(self):
        """
        Optional setup phase (kept for API symmetry with other trainers).
        """
        log_rank_0("TorchTitanPretrainTrainer.setup()")

    def init(self):
        """
        Construct the underlying TorchTitan Trainer using the JobConfig.
        """
        log_rank_0("TorchTitanPretrainTrainer.init() - building TorchTitan Trainer")

        from torchtitan.train import Trainer  # type: ignore[import]

        # Note: TorchTitan's logger has already been patched in __init__.py
        # to use a named logger instead of root logger for proper source tracking.
        # backend_args is a SimpleNamespace produced by TorchTitanJobConfigBuilder
        # Convert it to JobConfig for TorchTitan's Trainer (handles custom extensions)
        job_config = self._build_job_config_from_namespace(self.backend_args)
        self._trainer = Trainer(job_config)

    # --------------------------------------------------------------------- #
    # Training entrypoint
    # --------------------------------------------------------------------- #

    def run_train(self):
        """
        Execute TorchTitan pre-training using its Trainer.train() loop.

        This method is called by BaseTrainer.run() after applying patches.
        """
        if self._trainer is None:
            raise RuntimeError("TorchTitanPretrainTrainer.init() must be called before run_train().")

        log_rank_0("Executing TorchTitan pretrain...")
        self._trainer.train()
        log_rank_0("TorchTitan pretrain execution completed.")

    # --------------------------------------------------------------------- #
    # Helper methods
    # --------------------------------------------------------------------- #

    def _build_job_config_from_namespace(self, ns: SimpleNamespace) -> "JobConfig":  # type: ignore[name-defined]
        """
        Convert a nested SimpleNamespace back to TorchTitan's JobConfig.

        This method properly handles:
            1. TorchTitan's experimental.custom_args_module extension mechanism
            2. Merging custom JobConfig extensions with the base JobConfig
            3. Recursive dataclass construction with dynamic field attachment
            4. Preserving Primus-specific configurations under `primus` attribute

        This mirrors the logic from primus.modules.trainer.torchtitan.pre_trainer.build_job_config

        Args:
            ns: Nested SimpleNamespace with TorchTitan configuration

        Returns:
            JobConfig dataclass instance (potentially extended with custom and Primus fields)
        """
        import importlib

        from torchtitan.config.job_config import Experimental, JobConfig

        from primus.core.utils.yaml_utils import dict_to_nested_namespace

        # Step 1: Convert namespace to dict
        cfg_dict = self._namespace_to_dict(ns)

        # Step 2: Extract and preserve Primus-specific configuration
        primus_config = cfg_dict.pop("primus", None)

        # Step 3: Parse the experimental section to check for a custom JobConfig extension
        experimental_cfg = cfg_dict.get("experimental", {})
        experimental = Experimental(**experimental_cfg)

        # Step 4: If a custom_args_module is defined, import and merge with JobConfig
        custom_job_config_cls = JobConfig
        if experimental and getattr(experimental, "custom_args_module", None):
            try:
                module = importlib.import_module(experimental.custom_args_module)
                extended_job_config_cls = getattr(module, "JobConfig")
                custom_job_config_cls = self._merge_configs(JobConfig, extended_job_config_cls)
                log_rank_0(f"Loaded and merged custom JobConfig from {experimental.custom_args_module}")
            except Exception as e:
                log_rank_0(
                    f"Warning: Failed to load custom_args_module '{experimental.custom_args_module}': {e}"
                )

        # Step 5: Parse config dict (including custom fields) into dataclass recursively
        job_config = self._dict_to_dataclass(custom_job_config_cls, cfg_dict)

        # Step 6: Attach Primus configuration as a dynamic attribute if present
        if primus_config:
            job_config.primus = dict_to_nested_namespace(primus_config)
            log_rank_0(f"Attached Primus configuration to JobConfig ({len(primus_config)} top-level keys)")

        return job_config

    @staticmethod
    def _namespace_to_dict(obj: Any) -> Any:
        """Recursively convert SimpleNamespace to dict."""
        if isinstance(obj, SimpleNamespace):
            return {k: TorchTitanPretrainTrainer._namespace_to_dict(v) for k, v in vars(obj).items()}
        return obj

    @staticmethod
    def _merge_configs(base_cls: type, custom_cls: type) -> type:
        """
        Merges two dataclass types into one unified dataclass.

        Merge logic:
        - If a field exists in both:
            - If both fields are dataclasses, recursively merge them.
            - Otherwise, the custom field overrides the base.
        - Fields only in base or only in custom are included as-is.

        This mirrors the logic from primus.modules.trainer.torchtitan.pre_trainer.merge_configs
        """
        from dataclasses import field, fields, is_dataclass, make_dataclass

        base_fields = {f.name: f for f in fields(base_cls)}
        custom_fields = {f.name: f for f in fields(custom_cls)}

        merged = []

        # Merge overlapping and base-only fields
        for name, base_f in base_fields.items():
            if name in custom_fields:
                custom_f = custom_fields[name]
                if is_dataclass(base_f.type) and is_dataclass(custom_f.type):
                    merged_type = TorchTitanPretrainTrainer._merge_configs(base_f.type, custom_f.type)
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

    @staticmethod
    def _dict_to_dataclass(cls: type, data: dict) -> Any:
        """
        Recursively convert dict to dataclass, handling nested and custom fields.

        This mirrors the logic from primus.modules.trainer.torchtitan.pre_trainer._dict_to_dataclass
        """
        from dataclasses import fields, is_dataclass

        if not is_dataclass(cls):
            return data

        # Collect valid field names
        field_names = {f.name for f in fields(cls)}
        init_values: dict = {}

        # Only use known fields for constructor
        for f in fields(cls):
            if f.name in data:
                val = data[f.name]
                if is_dataclass(f.type) and isinstance(val, dict):
                    init_values[f.name] = TorchTitanPretrainTrainer._dict_to_dataclass(f.type, val)
                else:
                    init_values[f.name] = val

        # Instantiate dataclass
        obj = cls(**init_values)

        # Attach unknown fields dynamically
        for k, v in data.items():
            if k not in field_names:
                setattr(obj, k, v)

        return obj
