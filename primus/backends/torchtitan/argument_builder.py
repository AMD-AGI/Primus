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
from typing import Any, Dict, Mapping

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


def _deep_merge(base: Dict[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge ``overrides`` into ``base`` and return a new dict.

    - Nested mappings are merged recursively.
    - Non-mapping values in overrides replace base values.
    """
    result: Dict[str, Any] = dict(base)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, Mapping):
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
    Utility to build a final TorchTitan ``JobConfig`` for Primus.

    It merges:
        1. TorchTitan's default ``JobConfig`` values
        2. Nested overrides from Primus config / CLI (as dicts)

    Usage:
        builder = TorchTitanJobConfigBuilder()
        builder.update(cfg_dict_from_primus)
        job_cfg = builder.to_job_config()
    """

    def __init__(self) -> None:
        # Nested overrides to be merged into JobConfig defaults
        self._overrides: Dict[str, Any] = {}

    def update(self, values: Mapping[str, Any]) -> "TorchTitanJobConfigBuilder":
        """
        Merge a nested mapping of overrides into the current builder state.

        The structure of ``values`` should follow TorchTitan's JobConfig layout,
        e.g.:

            {
                "model": {"name": "llama3", "flavor": "debugmodel"},
                "training": {"steps": 1000},
                "parallelism": {"tensor_parallel_degree": 4},
            }
        """
        self._overrides = _deep_merge(self._overrides, values)
        return self

    def to_dict(self) -> Dict[str, Any]:
        """
        Return the merged nested dictionary:
            defaults(JobConfig) + overrides
        """
        defaults = _load_torchtitan_defaults()
        return _deep_merge(defaults, self._overrides)

    def to_job_config(self) -> JobConfig:
        """
        Materialize the final TorchTitan ``JobConfig`` dataclass instance.
        """
        merged = self.to_dict()
        return _dict_to_dataclass(JobConfig, merged)
