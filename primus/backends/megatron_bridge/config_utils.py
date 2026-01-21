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
from typing import Any, Callable, Dict

from primus.modules.module_utils import log_rank_0


def auto_filter_and_call(func: Callable, kwargs: Dict[str, Any], max_retries: int = 50) -> Any:
    """
    Automatically filter kwargs and call function with retry mechanism.

    This function tries to call the target function with given kwargs.
    If it fails due to unexpected keyword arguments (from func itself or
    any nested function calls), it automatically removes the problematic
    parameters and retries until success or max_retries reached.

    Strategy:
        1. Directly try calling func(**kwargs)
        2. If TypeError occurs: Parse error message to find invalid parameter
        3. Remove invalid parameter and retry
        4. Repeat until success or max_retries reached

    Note: We don't pre-filter by signature because:
        - func may call other functions internally
        - Those nested functions may have different parameter requirements
        - Only runtime errors reveal the true invalid parameters

    Args:
        func: Target function to call
        kwargs: Dictionary of keyword arguments
        max_retries: Maximum number of retry attempts (default: 50)

    Returns:
        The return value of the function call

    Raises:
        TypeError: If the function call fails after all retries

    Example:
        >>> def my_func(a, b): return a + b
        >>> result = auto_filter_and_call(my_func, {'a': 1, 'b': 2, 'c': 3, 'd': 4})
        # Automatically removes 'c' and 'd', returns 3
    """
    import re

    log_rank_0(f"Attempting to call {func.__name__}() with {len(kwargs)} parameters...")

    # Try calling directly without pre-filtering
    attempt = 0
    current_kwargs = kwargs.copy()
    removed_params = []

    while attempt < max_retries:
        try:
            result = func(**current_kwargs)

            if removed_params:
                log_rank_0(
                    f"✅ Successfully called {func.__name__}() after removing "
                    f"{len(removed_params)} invalid parameters: {removed_params}"
                )
            else:
                log_rank_0(f"✅ Successfully called {func.__name__}() with all parameters")

            return result

        except TypeError as e:
            error_msg = str(e)

            # Pattern 1: "got an unexpected keyword argument 'param_name'"
            match = re.search(r"unexpected keyword argument[s]? ['\"]([^'\"]+)['\"]", error_msg)

            if match:
                invalid_param = match.group(1)

                if invalid_param in current_kwargs:
                    removed_params.append(invalid_param)
                    del current_kwargs[invalid_param]
                    log_rank_0(
                        f"⚠️  Retry {attempt + 1}: Removing invalid parameter '{invalid_param}' "
                        f"({len(current_kwargs)} params remaining)"
                    )
                    attempt += 1
                    continue
                else:
                    # Parameter already removed, but still getting error
                    raise TypeError(
                        f"Failed to call {func.__name__}(): {error_msg}. "
                        f"Parameter '{invalid_param}' was already removed."
                    ) from e

            # Pattern 2: "missing required argument 'param_name'"
            if "missing" in error_msg.lower() and "required" in error_msg.lower():
                # This is a different error - missing required parameters
                raise TypeError(
                    f"Failed to call {func.__name__}(): {error_msg}. "
                    f"Current parameters: {list(current_kwargs.keys())}"
                ) from e

            # Unknown TypeError pattern
            raise TypeError(
                f"Failed to call {func.__name__}(): {error_msg}. "
                f"Could not parse error message to extract invalid parameter name."
            ) from e

    # Max retries exceeded
    raise TypeError(
        f"Failed to call {func.__name__}() after {max_retries} retries. "
        f"Removed parameters: {removed_params}. "
        f"Remaining parameters: {list(current_kwargs.keys())}"
    )


def load_recipe_config(backend_args: SimpleNamespace) -> Any:
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
    recipe = backend_args.recipe
    flavor = backend_args.flavor

    # Recipe and flavor are mandatory for Megatron-Bridge
    assert recipe, "Recipe must be specified for Megatron-Bridge backend"
    assert flavor, "Flavor must be specified for Megatron-Bridge backend"

    # Construct full module path and function name
    full_module_path = f"megatron.bridge.recipes.{recipe}"
    function_name = flavor

    log_rank_0(f"Loading recipe: {full_module_path}.{function_name}()")

    # Import module and get function
    try:
        module = importlib.import_module(full_module_path)
    except ImportError as e:
        assert False, f"Recipe loading failed: Cannot import '{full_module_path}': {e}"

    assert hasattr(
        module, function_name
    ), f"Recipe loading failed: Function '{function_name}' not found in '{full_module_path}'"
    recipe_func = getattr(module, function_name)

    backend_dict = namespace_to_dict(backend_args)
    config_container = auto_filter_and_call(recipe_func, backend_dict)
    log_rank_0(f"Successfully loaded recipe: {full_module_path}.{function_name}()")

    # Validate return type
    from megatron.bridge.training.config import ConfigContainer

    assert isinstance(config_container, ConfigContainer), (
        f"Recipe function '{full_module_path}.{function_name}()' must return "
        f"ConfigContainer, but returned {type(config_container).__name__}"
    )

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


# def _dict_to_dataclass(cls: type, data: dict) -> Any:
#     """
#     Recursively convert dict to dataclass (fallback implementation).

#     This is a simplified fallback when ConfigContainer.from_dict() fails.
#     It handles nested dataclass fields but may not support all Megatron-Bridge features.

#     Args:
#         cls: Target dataclass type
#         data: Dictionary to convert

#     Returns:
#         Instance of the dataclass with all fields populated
#     """
#     from dataclasses import fields, is_dataclass

#     if not is_dataclass(cls):
#         return data

#     # Collect valid field names
#     field_names = {f.name for f in fields(cls)}
#     init_values: dict = {}

#     # Only use known fields for constructor
#     for f in fields(cls):
#         if f.name in data:
#             val = data[f.name]
#             if is_dataclass(f.type) and isinstance(val, dict):
#                 init_values[f.name] = _dict_to_dataclass(f.type, val)
#             else:
#                 init_values[f.name] = val

#     # Instantiate dataclass
#     obj = cls(**init_values)

#     # Attach unknown fields dynamically
#     for k, v in data.items():
#         if k not in field_names:
#             setattr(obj, k, v)

#     return obj
