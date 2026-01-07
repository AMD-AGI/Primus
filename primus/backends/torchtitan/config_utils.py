###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
TorchTitan configuration utilities.

This module provides utility functions for converting between different
configuration representations used in TorchTitan integration:
    - SimpleNamespace ↔ dict conversions
    - dict → JobConfig dataclass construction
    - Merging custom JobConfig extensions
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

from primus.core.utils.yaml_utils import dict_to_nested_namespace
from primus.modules.module_utils import log_rank_0


def build_job_config_from_namespace(ns: SimpleNamespace) -> Any:
    """
    Convert a nested SimpleNamespace to TorchTitan's JobConfig.

    This function properly handles:
        1. TorchTitan's experimental.custom_args_module extension mechanism
        2. Merging custom JobConfig extensions with the base JobConfig
        3. Recursive dataclass construction with dynamic field attachment
        4. Preserving Primus-specific configurations under `primus` attribute

    Args:
        ns: Nested SimpleNamespace with TorchTitan configuration

    Returns:
        JobConfig dataclass instance (potentially extended with custom and Primus fields)
    """
    from torchtitan.config.job_config import Experimental, JobConfig

    # Step 1: Convert namespace to dict
    cfg_dict = namespace_to_dict(ns)

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
            custom_job_config_cls = merge_dataclass_configs(JobConfig, extended_job_config_cls)
            log_rank_0(f"Loaded and merged custom JobConfig from {experimental.custom_args_module}")
        except Exception as e:
            log_rank_0(f"Warning: Failed to load custom_args_module '{experimental.custom_args_module}': {e}")

    # Step 5: Parse config dict (including custom fields) into dataclass recursively
    job_config = dict_to_dataclass(custom_job_config_cls, cfg_dict)

    # Step 6: Attach Primus configuration as a dynamic attribute if present
    if primus_config:
        job_config.primus = dict_to_nested_namespace(primus_config)
        log_rank_0(f"Attached Primus configuration to JobConfig ({len(primus_config)} top-level keys)")

    return job_config


def namespace_to_dict(obj: Any) -> Any:
    """
    Recursively convert SimpleNamespace to dict.

    Args:
        obj: Object to convert (can be SimpleNamespace, dict, list, or primitive)

    Returns:
        Converted object with all SimpleNamespace instances replaced by dicts
    """
    if isinstance(obj, SimpleNamespace):
        return {k: namespace_to_dict(v) for k, v in vars(obj).items()}
    return obj


def merge_dataclass_configs(base_cls: type, custom_cls: type) -> type:
    """
    Merge two dataclass types into one unified dataclass.

    Merge logic:
        - If a field exists in both:
            - If both fields are dataclasses, recursively merge them.
            - Otherwise, the custom field overrides the base.
        - Fields only in base or only in custom are included as-is.

    Args:
        base_cls: Base dataclass type (e.g., TorchTitan's JobConfig)
        custom_cls: Custom dataclass type with extensions

    Returns:
        New dataclass type with merged fields
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
                merged_type = merge_dataclass_configs(base_f.type, custom_f.type)
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


def dict_to_dataclass(cls: type, data: dict) -> Any:
    """
    Recursively convert dict to dataclass, handling nested and custom fields.

    This function:
        - Constructs dataclass instances from dictionaries
        - Recursively processes nested dataclass fields
        - Attaches unknown fields dynamically as attributes

    Args:
        cls: Target dataclass type
        data: Dictionary to convert

    Returns:
        Instance of the dataclass with all fields populated
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
                init_values[f.name] = dict_to_dataclass(f.type, val)
            else:
                init_values[f.name] = val

    # Instantiate dataclass
    obj = cls(**init_values)

    # Attach unknown fields dynamically
    for k, v in data.items():
        if k not in field_names:
            setattr(obj, k, v)

    return obj
