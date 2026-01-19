###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron-Bridge configuration utilities.

This module provides utility functions for converting between different
configuration representations used in Megatron-Bridge integration:
    - SimpleNamespace ↔ dict conversions
    - dict → ConfigContainer dataclass construction
    - Handling Megatron-Bridge's InstantiationMode
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from primus.core.utils.yaml_utils import dict_to_nested_namespace
from primus.modules.module_utils import log_rank_0


def build_job_config_from_namespace(ns: SimpleNamespace) -> Any:
    """
    Convert a nested SimpleNamespace to Megatron-Bridge's ConfigContainer.

    This function properly handles:
        1. Converting SimpleNamespace to dict recursively
        2. Using ConfigContainer.from_dict() with proper InstantiationMode
        3. Preserving Primus-specific configurations under `primus` attribute
        4. Adding required _target_ field for Megatron-Bridge instantiation

    Args:
        ns: Nested SimpleNamespace with Megatron-Bridge configuration

    Returns:
        ConfigContainer dataclass instance (potentially extended with Primus fields)
    """
    from megatron.bridge.training.config import ConfigContainer
    from megatron.bridge.training.utils.config_utils import InstantiationMode

    # Step 1: Convert namespace to dict
    cfg_dict = namespace_to_dict(ns)

    # Step 2: Extract and preserve Primus-specific configuration
    primus_config = cfg_dict.pop("primus", None)

    # Step 3: Add _target_ field required by Megatron-Bridge's from_dict
    # This tells the instantiate() function which class to create
    cfg_dict["_target_"] = "megatron.bridge.training.config.ConfigContainer"

    # Step 4: Use ConfigContainer.from_dict() with LENIENT mode
    # LENIENT mode allows extra keys and is more flexible during development
    try:
        config_container = ConfigContainer.from_dict(cfg_dict, mode=InstantiationMode.LENIENT)
        log_rank_0("ConfigContainer created successfully from namespace")
    except Exception as e:
        log_rank_0(f"Warning: Failed to create ConfigContainer with LENIENT mode: {e}")
        log_rank_0("Falling back to direct instantiation...")
        
        # Fallback: remove _target_ and try direct instantiation
        cfg_dict.pop("_target_", None)
        config_container = _dict_to_dataclass(ConfigContainer, cfg_dict)

    # Step 5: Attach Primus configuration as a dynamic attribute if present
    if primus_config:
        config_container.primus = dict_to_nested_namespace(primus_config)
        log_rank_0(f"Attached Primus configuration to ConfigContainer ({len(primus_config)} top-level keys)")

    return config_container


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
    elif isinstance(obj, dict):
        return {k: namespace_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(namespace_to_dict(item) for item in obj)
    return obj


def _dict_to_dataclass(cls: type, data: dict) -> Any:
    """
    Recursively convert dict to dataclass (fallback implementation).

    This is a simplified fallback when ConfigContainer.from_dict() fails.
    It handles nested dataclass fields but may not support all Megatron-Bridge features.

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
                init_values[f.name] = _dict_to_dataclass(f.type, val)
            else:
                init_values[f.name] = val

    # Instantiate dataclass
    obj = cls(**init_values)

    # Attach unknown fields dynamically
    for k, v in data.items():
        if k not in field_names:
            setattr(obj, k, v)

    return obj
