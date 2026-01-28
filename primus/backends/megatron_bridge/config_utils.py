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

            log_rank_0(f"error_msg: {error_msg} {func.__name__}")

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


def _merge_dataclass_recursive(target: Any, source: Any, path: str = "") -> None:
    """
    Recursively merge dataclass fields from source into target.

    This function performs a deep merge of dataclass objects, allowing partial
    overrides of nested configurations without replacing entire objects.

    Example:
        Recipe returns: TrainingConfig(train_iters=1000, eval_interval=500)
        User overrides: TrainingConfig(train_iters=2000)
        Result: TrainingConfig(train_iters=2000, eval_interval=500)  # eval_interval preserved!

    Args:
        target: Target dataclass to merge into (will be modified in-place)
        source: Source dataclass to merge from
        path: Current path for logging (e.g., "config_container.train.optimizer")

    Returns:
        None (modifies target in-place)
    """
    from dataclasses import fields, is_dataclass

    if not is_dataclass(source):
        return  # Source is not a dataclass, nothing to merge

    for field in fields(source):
        field_name = field.name
        source_value = getattr(source, field_name)

        if source_value is None:
            continue  # Skip None values - don't override with None

        current_path = f"{path}.{field_name}" if path else field_name

        if not hasattr(target, field_name):
            # Target doesn't have this field, just set it
            setattr(target, field_name, source_value)
            log_rank_0(f"  ↳ Setting {current_path} (new field)")
            continue

        target_value = getattr(target, field_name)

        # If both are dataclasses, recursively merge
        if is_dataclass(target_value) and is_dataclass(source_value):
            _merge_dataclass_recursive(target_value, source_value, current_path)
        else:
            # Non-dataclass or mixed types: direct override
            setattr(target, field_name, source_value)
            log_rank_0(f"  ↳ Overriding {current_path}")


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

    Configuration Override:
        After the recipe returns a ConfigContainer, any ConfigContainer fields
        present in backend_args will override the recipe's default values.
        This ensures user overrides in YAML take precedence:

        Recipe returns: ConfigContainer(train=TrainingConfig(train_iters=1000), ...)
        backend_args has: train_iters=2000
        Final result: ConfigContainer(train=TrainingConfig(train_iters=2000), ...)

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
        backend_args: SimpleNamespace containing recipe specification and user configuration

    Returns:
        ConfigContainer from recipe with user overrides applied (guaranteed non-None)

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

    # Override config_container fields with values from backend_args
    # This ensures user overrides in YAML take precedence over recipe defaults
    # Uses recursive merge for nested dataclass fields to allow partial overrides
    from dataclasses import fields, is_dataclass

    log_rank_0("Applying backend_args overrides to config_container...")

    # Fields that should NOT be merged from backend_args because they have
    # different meanings in Primus vs Megatron-Bridge ConfigContainer
    # - model: In Primus it's a string path to YAML, in ConfigContainer it's GPTModelProvider object
    # - recipe/flavor: Metadata fields for recipe selection, not part of ConfigContainer
    primus_metadata_fields = {
        "model",  # Primus: YAML path string, ConfigContainer: GPTModelProvider object
        "recipe",  # Primus metadata: which recipe module to use
        "flavor",  # Primus metadata: which recipe function to call
        "primus",  # Primus internal metadata
    }

    for field in fields(config_container):
        field_name = field.name

        # Skip Primus metadata fields
        if field_name in primus_metadata_fields:
            continue

        if hasattr(backend_args, field_name):
            field_value = getattr(backend_args, field_name)
            if field_value is not None:  # Only override if explicitly set
                target_value = getattr(config_container, field_name)

                # If both are dataclasses, recursively merge to preserve unset fields
                if is_dataclass(target_value) and is_dataclass(field_value):
                    _merge_dataclass_recursive(target_value, field_value, f"config_container.{field_name}")
                else:
                    # Non-dataclass or None target: direct replacement
                    setattr(config_container, field_name, field_value)
                    log_rank_0(f"  ↳ Replacing config_container.{field_name}")

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
