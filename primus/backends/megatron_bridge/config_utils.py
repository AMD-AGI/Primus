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
import inspect
from types import SimpleNamespace
from typing import Any, Optional

from primus.core.utils.yaml_utils import dict_to_nested_namespace
from primus.modules.module_utils import log_dict_aligned, log_rank_0


def filter_kwargs_by_signature(func: Any, kwargs_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Filter kwargs dictionary to only include parameters accepted by the function.

    This function inspects the target function's signature and returns a filtered
    dictionary containing only the keyword arguments that the function can accept.

    Args:
        func: The function whose signature to check
        kwargs_dict: Dictionary of potential keyword arguments to filter

    Returns:
        Filtered dictionary containing only parameters that:
        - Match explicit parameter names in the function signature, OR
        - Can be accepted if function has **kwargs

    Example:
        >>> def my_func(a: int, b: str, **kwargs): pass
        >>> filter_kwargs_by_signature(my_func, {'a': 1, 'b': 'x', 'c': 3, 'd': 4})
        {'a': 1, 'b': 'x', 'c': 3, 'd': 4}  # All accepted due to **kwargs

        >>> def strict_func(a: int, b: str): pass
        >>> filter_kwargs_by_signature(strict_func, {'a': 1, 'b': 'x', 'c': 3})
        {'a': 1, 'b': 'x'}  # Only 'a' and 'b' accepted
    """
    # Get function signature
    sig = inspect.signature(func)

    # Check if function accepts **kwargs (VAR_KEYWORD)
    accepts_var_keyword = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()
    )

    # Get explicit parameter names (excluding *args and **kwargs)
    explicit_params = {
        name
        for name, param in sig.parameters.items()
        if param.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
    }

    # Filter kwargs: include if explicitly in signature OR function accepts **kwargs
    filtered = {}
    for k, v in kwargs_dict.items():
        if k in explicit_params or accepts_var_keyword:
            filtered[k] = v

    return filtered


def build_job_config_from_namespace(backend_args: SimpleNamespace) -> Any:
    """
    Convert a nested SimpleNamespace to Megatron-Bridge's ConfigContainer.

    Note: Recipe loading and merging is already done in MegatronBridgeArgBuilder,
    so backend_args already contains the merged configuration (recipe + user overrides).

    This function handles:
        1. Converting SimpleNamespace to dict recursively
        2. Using ConfigContainer.from_dict() with proper InstantiationMode
        3. Preserving Primus-specific configurations under `primus` attribute

    Args:
        module_config: Module configuration (for reference, not used currently)
        backend_args: Nested SimpleNamespace with Megatron-Bridge configuration
                      (already merged with recipe if specified)

    Returns:
        ConfigContainer dataclass instance (potentially extended with Primus fields)
    """
    from megatron.bridge.training.config import ConfigContainer
    from megatron.bridge.training.utils.config_utils import InstantiationMode

    log_dict_aligned("Backend args", backend_args)

    # Step 1: Convert namespace to dict (recipe already loaded and merged in ArgBuilder)
    cfg_dict = namespace_to_dict(backend_args)

    # Step 2: Extract and preserve Primus-specific configuration
    primus_config = cfg_dict.pop("primus", None)

    # Remove recipe-related fields from cfg_dict as they are not part of ConfigContainer
    cfg_dict.pop("recipe", None)
    cfg_dict.pop("flavor", None)
    cfg_dict.pop("recipe_kwargs", None)

    # Step 3: Add _target_ field required by Megatron-Bridge's from_dict
    # This tells the instantiate() function which class to create
    cfg_dict["_target_"] = "megatron.bridge.training.config.ConfigContainer"

    # Step 4: Use ConfigContainer.from_dict() with LENIENT mode
    # LENIENT mode allows extra keys and is more flexible during development
    try:
        log_dict_aligned("Config dict", cfg_dict)
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


def load_recipe_config(ns: SimpleNamespace) -> Any:
    """
    Load Megatron-Bridge recipe configuration if specified.

    Recipe format:
        ns.recipe: Module path (e.g., "qwen.qwen3")
        ns.flavor: Function name (e.g., "qwen3_32b_finetune_config")
        ns.<recipe_params>: Simple parameters passed to recipe function
        ns.<container_fields>: Complex fields merged with recipe result later
        Full function: megatron.bridge.recipes.{recipe}.{flavor}(**recipe_params)

    Parameter Filtering:
        The function intelligently filters which ns attributes to pass to recipe:

        NOT passed to recipe (metadata):
            - recipe, flavor, recipe_kwargs, primus

        NOT passed to recipe (ConfigContainer fields for later merging):
            - train, model, optimizer, scheduler, dataset, logger, tokenizer, checkpoint
            - ddp, dist, ft, profiling, comm_overlap, mixed_precision, etc.

        PASSED to recipe (simple recipe parameters):
            - hf_path, peft, tensor_model_parallel_size, pipeline_model_parallel_size
            - train_iters, global_batch_size, micro_batch_size, seq_length
            - lr, min_lr, data_paths, pretrained_checkpoint, etc.

    Example 1 (basic - no parameters):
        ns.recipe = "qwen.qwen3"
        ns.flavor = "qwen3_32b_finetune_config"
        → Calls megatron.bridge.recipes.qwen.qwen3.qwen3_32b_finetune_config()

    Example 2 (with recipe parameters):
        ns.recipe = "qwen.qwen3"
        ns.flavor = "qwen3_32b_finetune_config"
        ns.hf_path = "Qwen/Qwen3-32B"
        ns.peft = "lora"
        ns.tensor_model_parallel_size = 8
        → Passes to recipe: hf_path, peft, tensor_model_parallel_size

    Example 3 (with container field overrides - merged later):
        ns.recipe = "qwen.qwen3"
        ns.flavor = "qwen3_32b_finetune_config"
        ns.train = TrainingConfig(...)  # Complex object
        → train is NOT passed to recipe, merged with result later

    Args:
        ns: SimpleNamespace containing recipe specification and user configuration

    Returns:
        ConfigContainer from recipe (guaranteed non-None)

    Raises:
        AssertionError: If recipe or flavor is not specified (both are mandatory)
        RuntimeError: If recipe loading fails (import error, function not found, etc.)
    """
    recipe = getattr(ns, "recipe", None)
    flavor = getattr(ns, "flavor", None)

    # Recipe and flavor are mandatory for Megatron-Bridge
    assert recipe, "Recipe must be specified for Megatron-Bridge backend"
    assert flavor, "Flavor must be specified for Megatron-Bridge backend"

    try:
        # Construct full module path and function name
        full_module_path = f"megatron.bridge.recipes.{recipe}"
        function_name = flavor

        log_rank_0(f"Loading recipe: {full_module_path}.{function_name}()")

        # Import module and get function
        module = importlib.import_module(full_module_path)
        recipe_func = getattr(module, function_name)

        # Convert ns to dict
        ns_dict = namespace_to_dict(ns)

        # Recipe metadata that should never be passed to recipe function
        recipe_metadata = ["recipe", "flavor", "recipe_kwargs", "primus"]

        # ConfigContainer fields that should not be passed to recipe (they're for merging later)
        # These are complex objects that recipe functions don't accept as input
        container_fields = [
            "_target_",
            "rng",
            "rerun_state_machine",
            "train",
            "model",
            "optimizer",
            "ddp",
            "scheduler",
            "dataset",
            "logger",
            "tokenizer",
            "checkpoint",
            "dist",
            "ft",
            "straggler",
            "nvrx_straggler",
            "profiling",
            "comm_overlap",
            "mixed_precision",
            "tensor_inspect",
            "inprocess_restart",
            "trainable",
            "sink_level",
            "file_sink_level",
            "stderr_sink_level",
        ]

        # First filter: remove metadata and complex container fields
        filtered_dict = {
            k: v for k, v in ns_dict.items() if k not in recipe_metadata and k not in container_fields
        }

        # Second filter: only keep parameters accepted by function signature
        recipe_kwargs = filter_kwargs_by_signature(recipe_func, filtered_dict)

        if recipe_kwargs:
            log_rank_0(f"Recipe kwargs: {list(recipe_kwargs.keys())}")

        # Call recipe function with filtered kwargs
        config_container = recipe_func(**recipe_kwargs)
        log_rank_0(f"Successfully loaded recipe: {full_module_path}.{function_name}()")

        return config_container

    except ImportError as e:
        log_rank_0(f"Error: Failed to import recipe module: {e}")
        raise RuntimeError(f"Recipe loading failed: Cannot import '{full_module_path}'") from e
    except AttributeError as e:
        log_rank_0(f"Error: Recipe function not found: {e}")
        raise RuntimeError(
            f"Recipe loading failed: Function '{function_name}' not found in '{full_module_path}'"
        ) from e
    except Exception as e:
        log_rank_0(f"Error: Failed to load recipe: {e}")
        raise RuntimeError(f"Recipe loading failed: {full_module_path}.{function_name}()") from e


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
