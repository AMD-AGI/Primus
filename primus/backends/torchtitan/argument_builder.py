###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
TorchTitan argument builder utilities.

This module is the TorchTitan counterpart of
``primus.backends.megatron.argument_builder``.

It provides a small helper to:
    - load TorchTitan's default JobConfig values
    - apply nested overrides from Primus config or CLI
    - materialize a final ``JobConfig`` dataclass instance
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from types import SimpleNamespace
from typing import Any, Dict

from torchtitan.config.job_config import JobConfig

# -----------------------------------------------------------------------------
# Load TorchTitan's default JobConfig as a nested dict
# -----------------------------------------------------------------------------


def _load_torchtitan_defaults() -> Dict[str, Any]:
    """
    Load TorchTitan's default JobConfig values as a nested dictionary.

    This is analogous to Megatron's ``_load_megatron_defaults`` helper, but
    for TorchTitan's dataclass-based configuration.
    """
    return JobConfig().to_dict()


# -----------------------------------------------------------------------------
# Helper: deep-merge nested dictionaries
# -----------------------------------------------------------------------------


def _namespace_to_dict(obj: Any) -> Any:
    """
    Recursively convert SimpleNamespace to dict.

    This is needed because TorchTitan configs can have nested SimpleNamespace objects.
    """
    if isinstance(obj, SimpleNamespace):
        return {k: _namespace_to_dict(v) for k, v in vars(obj).items()}
    elif isinstance(obj, dict):
        return {k: _namespace_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_namespace_to_dict(item) for item in obj)
    else:
        return obj


def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge ``overrides`` into ``base`` and return a new dict.

    - Nested dicts are merged recursively.
    - Non-dict values in overrides replace base values.
    """
    result: Dict[str, Any] = dict(base)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


# -----------------------------------------------------------------------------
# Helper: convert nested dict → nested JobConfig dataclasses
# -----------------------------------------------------------------------------


def _dict_to_dataclass(cls: type, data: Dict[str, Any]) -> Any:
    """
    Recursively convert a nested dictionary into a dataclass instance.

    This mirrors the logic used in ``TorchTitanPretrainTrainer._dict_to_dataclass``,
    but is implemented here as a standalone helper for reuse.
    """
    if not is_dataclass(cls):
        return data

    dataclass_fields = fields(cls)
    field_names = {f.name for f in dataclass_fields}
    init_values: Dict[str, Any] = {}

    # Use only known fields for constructor
    for f in dataclass_fields:
        if f.name in data:
            val = data[f.name]
            if is_dataclass(f.type) and isinstance(val, dict):
                init_values[f.name] = _dict_to_dataclass(f.type, val)
            else:
                init_values[f.name] = val

    # Instantiate dataclass
    obj = cls(**init_values)

    # Attach unknown fields dynamically (if any)
    for k, v in data.items():
        if k not in field_names:
            setattr(obj, k, v)

    return obj


# -----------------------------------------------------------------------------
# TorchTitanJobConfigBuilder
# -----------------------------------------------------------------------------


class TorchTitanJobConfigBuilder:
    """
    A lightweight utility to build final TorchTitan ``JobConfig`` for Primus.

    It merges:
        1. Primus CLI arguments
        2. Primus config arguments
        3. TorchTitan's default JobConfig values

    WITHOUT defining any manual mapping and WITHOUT maintaining version compatibility
    manually — because we rely entirely on TorchTitan's own JobConfig dataclass.

    Usage:
        builder = TorchTitanJobConfigBuilder()
        builder.update(cli_args)
        builder.update(config_args)
        job_cfg = builder.to_job_config()  # or builder.finalize()

    'job_cfg' is a JobConfig dataclass containing all fields TorchTitan expects.
    """

    def __init__(self) -> None:
        # Load TorchTitan defaults once during initialization
        # and store as the working configuration that will be updated
        self.config: Dict[str, Any] = _load_torchtitan_defaults()

    # ------------------------------------------------------------------
    # Add values to the configuration
    # ------------------------------------------------------------------
    def update(self, values: SimpleNamespace) -> "TorchTitanJobConfigBuilder":
        """
        Merge a SimpleNamespace into the current configuration.

        - Only accepts SimpleNamespace inputs (with nested structure)
        - All parameters are accepted (TorchTitan's JobConfig is flexible)
        - Values are directly merged into the working configuration

        The structure of ``values`` should follow TorchTitan's JobConfig layout,
        with nested SimpleNamespace objects, e.g.:

            SimpleNamespace(
                model=SimpleNamespace(name="llama3", flavor="debugmodel"),
                training=SimpleNamespace(steps=1000),
                parallelism=SimpleNamespace(tensor_parallel_degree=4)
            )
        """
        # Convert SimpleNamespace to dict
        values_dict = _namespace_to_dict(values)

        # Directly merge into the working configuration
        self.config = _deep_merge(self.config, values_dict)
        return self

    # ------------------------------------------------------------------
    # Produce the final TorchTitan JobConfig
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """
        Return a copy of the current configuration as a nested dictionary.

        The configuration already contains:
            - TorchTitan default JobConfig values (loaded during __init__)
            - Primus overrides (applied via update() calls)

        This is an intermediate representation before materializing
        the final JobConfig dataclass or SimpleNamespace.

        Note: Returns a deep copy to prevent external modifications.
        """
        import copy

        return copy.deepcopy(self.config)

    def to_namespace(self) -> SimpleNamespace:
        """
        Produce the final TorchTitan configuration as a SimpleNamespace.

        This method ensures API consistency with MegatronArgBuilder.to_namespace().
        The namespace contains a nested structure matching TorchTitan's JobConfig.

        Fields not provided by Primus are automatically filled with TorchTitan's defaults.

        Returns:
            SimpleNamespace with nested TorchTitan configuration that can be passed
            to convert back to JobConfig when needed
        """
        merged = self.to_dict()
        return self._dict_to_namespace(merged)

    def to_job_config(self) -> JobConfig:
        """
        Materialize the final TorchTitan ``JobConfig`` dataclass instance.

        Fields not provided by Primus are automatically filled with TorchTitan's defaults.

        Returns:
            JobConfig dataclass ready to be passed to TorchTitan's Trainer
        """
        merged = self.to_dict()
        return _dict_to_dataclass(JobConfig, merged)

    # Alias for usage style consistency with MegatronArgBuilder:
    # builder.finalize()
    finalize = to_namespace

    # ------------------------------------------------------------------
    # Helper: convert nested dict to nested SimpleNamespace
    # ------------------------------------------------------------------
    @staticmethod
    def _dict_to_namespace(data: Dict[str, Any]) -> SimpleNamespace:
        """
        Recursively convert a nested dictionary to a nested SimpleNamespace.

        This is used to provide a consistent interface with MegatronArgBuilder.
        """
        namespace_dict = {}
        for key, value in data.items():
            if isinstance(value, dict):
                namespace_dict[key] = TorchTitanJobConfigBuilder._dict_to_namespace(value)
            else:
                namespace_dict[key] = value
        return SimpleNamespace(**namespace_dict)
