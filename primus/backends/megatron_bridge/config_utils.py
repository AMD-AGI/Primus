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

import importlib
from types import SimpleNamespace
from typing import Any, Optional

from primus.core.utils.yaml_utils import dict_to_nested_namespace
from primus.modules.module_utils import log_rank_0


def build_job_config_from_namespace(ns: SimpleNamespace) -> Any:
    """
    Convert a nested SimpleNamespace to Megatron-Bridge's ConfigContainer.

    This function properly handles:
        1. Loading recipe configuration if specified (ns.recipe + ns.flavor)
        2. Merging recipe config with user overrides (user config has higher priority)
        3. Converting SimpleNamespace to dict recursively
        4. Using ConfigContainer.from_dict() with proper InstantiationMode
        5. Preserving Primus-specific configurations under `primus` attribute

    Args:
        ns: Nested SimpleNamespace with Megatron-Bridge configuration

    Returns:
        ConfigContainer dataclass instance (potentially extended with Primus fields)
    """
    from megatron.bridge.training.config import ConfigContainer
    from megatron.bridge.training.utils.config_utils import InstantiationMode

    # Step 1: Load recipe configuration if specified
    recipe_config = _load_recipe_config(ns)

    # Step 2: Convert namespace to dict
    cfg_dict = namespace_to_dict(ns)

    # Step 3: Merge recipe config with user config (user config has priority)
    if recipe_config is not None:
        # Convert recipe ConfigContainer to dict and merge
        try:
            from omegaconf import OmegaConf
            from megatron.bridge.training.utils.omegaconf_utils import create_omegaconf_dict_config

            # Convert recipe ConfigContainer to OmegaConf DictConfig
            recipe_omega, excluded_fields = create_omegaconf_dict_config(recipe_config)
            recipe_dict = OmegaConf.to_container(recipe_omega, resolve=True)

            # Merge: recipe as base, user overrides on top
            cfg_dict = _deep_merge_dicts(recipe_dict, cfg_dict)
            log_rank_0("Merged recipe configuration with user overrides (user config has priority)")
        except Exception as e:
            log_rank_0(f"Error: Failed to merge recipe config: {e}")
            raise RuntimeError(f"Recipe merging failed: Cannot convert or merge recipe configuration") from e

    # Step 4: Extract and preserve Primus-specific configuration
    primus_config = cfg_dict.pop("primus", None)

    # Remove recipe-related fields from cfg_dict as they are not part of ConfigContainer
    cfg_dict.pop("recipe", None)
    cfg_dict.pop("flavor", None)
    cfg_dict.pop("recipe_kwargs", None)

    # Step 5: Add _target_ field required by Megatron-Bridge's from_dict
    # This tells the instantiate() function which class to create
    cfg_dict["_target_"] = "megatron.bridge.training.config.ConfigContainer"

    # Step 6: Use ConfigContainer.from_dict() with LENIENT mode
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

    # Step 7: Attach Primus configuration as a dynamic attribute if present
    if primus_config:
        config_container.primus = dict_to_nested_namespace(primus_config)
        log_rank_0(f"Attached Primus configuration to ConfigContainer ({len(primus_config)} top-level keys)")

    return config_container


def _load_recipe_config(ns: SimpleNamespace) -> Optional[Any]:
    """
    Load Megatron-Bridge recipe configuration if specified.

    Recipe format:
        ns.recipe: Module path (e.g., "qwen.qwen3")
        ns.flavor: Function name (e.g., "qwen3_32b_finetune_config")
        ns.recipe_kwargs: Optional dict/namespace with recipe-specific parameters
        Full function: megatron.bridge.recipes.{recipe}.{flavor}(**recipe_kwargs)

    Example 1 (basic):
        ns.recipe = "qwen.qwen3"
        ns.flavor = "qwen3_32b_finetune_config"
        → Calls megatron.bridge.recipes.qwen.qwen3.qwen3_32b_finetune_config()

    Example 2 (with kwargs):
        ns.recipe = "qwen.qwen3"
        ns.flavor = "qwen3_32b_finetune_config"
        ns.recipe_kwargs = {
            "hf_path": "Qwen/Qwen3-32B",
            "tensor_model_parallel_size": 8,
            "pipeline_model_parallel_size": 2,
            "peft": "lora",
        }
        → Calls megatron.bridge.recipes.qwen.qwen3.qwen3_32b_finetune_config(**kwargs)

    Args:
        ns: SimpleNamespace that may contain 'recipe', 'flavor', and 'recipe_kwargs'

    Returns:
        ConfigContainer from recipe, or None if recipe not specified

    Raises:
        RuntimeError: If recipe is specified but loading fails (import error, function not found, etc.)
    """
    recipe = getattr(ns, "recipe", None)
    flavor = getattr(ns, "flavor", None)

    if not recipe or not flavor:
        return None

    try:
        # Construct full module path and function name
        full_module_path = f"megatron.bridge.recipes.{recipe}"
        function_name = flavor

        log_rank_0(f"Loading recipe: {full_module_path}.{function_name}()")

        # Import module and get function
        module = importlib.import_module(full_module_path)
        recipe_func = getattr(module, function_name)

        # Extract recipe kwargs if provided
        # User can provide recipe-specific arguments via ns.recipe_kwargs
        recipe_kwargs = getattr(ns, "recipe_kwargs", {})
        if isinstance(recipe_kwargs, SimpleNamespace):
            recipe_kwargs = namespace_to_dict(recipe_kwargs)
        
        if recipe_kwargs:
            log_rank_0(f"Recipe kwargs: {list(recipe_kwargs.keys())}")

        # Call recipe function to get ConfigContainer with kwargs
        config_container = recipe_func(**recipe_kwargs)
        log_rank_0(f"Successfully loaded recipe: {full_module_path}.{function_name}()")

        return config_container

    except ImportError as e:
        log_rank_0(f"Error: Failed to import recipe module: {e}")
        raise RuntimeError(f"Recipe loading failed: Cannot import '{full_module_path}'") from e
    except AttributeError as e:
        log_rank_0(f"Error: Recipe function not found: {e}")
        raise RuntimeError(f"Recipe loading failed: Function '{function_name}' not found in '{full_module_path}'") from e
    except Exception as e:
        log_rank_0(f"Error: Failed to load recipe: {e}")
        raise RuntimeError(f"Recipe loading failed: {full_module_path}.{function_name}()") from e


def _deep_merge_dicts(base: dict, override: dict) -> dict:
    """
    Deep merge two dictionaries, with override taking priority.

    Args:
        base: Base dictionary (lower priority)
        override: Override dictionary (higher priority)

    Returns:
        Merged dictionary with override values taking precedence
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dicts
            result[key] = _deep_merge_dicts(result[key], value)
        else:
            # Override takes priority
            result[key] = value

    return result


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
