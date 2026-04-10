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

import ast
import importlib
from types import SimpleNamespace
from typing import Any, Callable, Dict

from primus.modules.module_utils import log_rank_0

DATASET_PATH_KEYS = (
    "data_paths",
    "train_data_path",
    "valid_data_path",
    "test_data_path",
)
DATASET_CONFIG_FILE_KEYS = (
    "data_args_path",
    "per_split_data_args_path",
)
_BOOL_MAP = {"true": True, "false": False}
_is_dict = lambda o: isinstance(o, dict)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    return obj.get(key, default) if _is_dict(obj) else getattr(obj, key, default)


def _has(obj: Any, key: str) -> bool:
    return key in obj if _is_dict(obj) else hasattr(obj, key)


def _set(obj: Any, key: str, value: Any) -> None:
    if _is_dict(obj):
        obj[key] = value
    else:
        setattr(obj, key, value)


def _delete(obj: Any, key: str) -> None:
    if _is_dict(obj):
        obj.pop(key, None)
    elif hasattr(obj, key):
        delattr(obj, key)


def _normalize_optional_string(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    stripped = value.strip()
    if not stripped or stripped.lower() == "none":
        return None

    return stripped


def _parse_optional_bool(value: Any, field_name: str) -> bool | None:
    """
    Parse optional boolean values from bool/str/None with strict semantics.
    """
    if value is None or isinstance(value, bool):
        return value

    if isinstance(value, str):
        normalized = _normalize_optional_string(value)
        if normalized is None:
            return None
        lowered = normalized.lower()
        if lowered in _BOOL_MAP:
            return _BOOL_MAP[lowered]
        raise ValueError(
            f"Invalid boolean string for '{field_name}': {value!r}. Expected one of: true/false."
        )

    raise TypeError(f"Invalid type for '{field_name}': {type(value).__name__}. Expected bool, str, or None.")


def _to_str_list(value: Any) -> list[str]:
    """Coerce a single value or collection into a flat list of non-empty strings."""
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except Exception:
            parsed = None
        raw = list(parsed) if isinstance(parsed, (list, tuple)) else [value]
    elif isinstance(value, (list, tuple, set)):
        raw = list(value)
    else:
        raw = [value]
    return [str(v) for v in raw if _normalize_optional_string(v) is not None]


def _normalize_dataset_path_value(value: Any) -> list[str] | None:
    """Normalize dataset path fields to ``list[str] | None`` for Bridge recipes."""
    value = _normalize_optional_string(value)
    if value is None:
        return None
    return _to_str_list(value) or None


def _has_real_data_config(obj: Any) -> bool:
    for key in (*DATASET_PATH_KEYS, *DATASET_CONFIG_FILE_KEYS):
        value = _get(obj, key)
        if isinstance(value, str):
            if _normalize_optional_string(value) is not None:
                return True
        elif value:
            return True
    return False


def normalize_megatron_bridge_dataset_args(args: Any) -> Any:
    """
    Normalize Primus/CLI dataset arguments into the shapes expected by Bridge recipes.

    This keeps Primus config authoring flexible while making runtime behavior
    predictable for Megatron-Bridge:
      - accepts legacy `mock_data` as input alias, then converges to `mock`
      - string dataset prefixes are wrapped into lists
      - Python list literals passed through CLI remain supported
      - any explicit real-data config forces `mock=false` to avoid silent fallback
    """
    if args is None:
        return args

    for key in DATASET_PATH_KEYS:
        if _has(args, key):
            _set(args, key, _normalize_dataset_path_value(_get(args, key)))

    for key in DATASET_CONFIG_FILE_KEYS:
        if _has(args, key):
            _set(args, key, _normalize_optional_string(_get(args, key)))

    mock = _parse_optional_bool(_get(args, "mock"), "mock")
    if mock is None:
        mock = _parse_optional_bool(_get(args, "mock_data"), "mock_data")

    _delete(args, "mock_data")

    if _has_real_data_config(args):
        mock = False

    if mock is not None:
        _set(args, "mock", mock)

    return args


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
                        f"({len(current_kwargs)} params remaining) error_msg: {error_msg}"
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


def _merge_dict_to_dataclass(target: Any, source_dict: dict, path: str = "") -> None:
    """
    Recursively merge dict values into a dataclass target.

    Args:
        target: Target dataclass to merge into (modified in-place)
        source_dict: Source dict containing values to merge
        path: Current path for logging (e.g., "config_container.train")

    Returns:
        None (modifies target in-place)
    """
    from dataclasses import fields, is_dataclass

    if not is_dataclass(target):
        return  # Target is not a dataclass, nothing to merge

    for field in fields(target):
        field_name = field.name

        # Skip if source doesn't have this field
        if field_name not in source_dict:
            continue

        source_value = source_dict[field_name]

        # Skip None values
        if source_value is None:
            continue

        current_path = f"{path}.{field_name}" if path else field_name
        target_value = getattr(target, field_name)

        # If target field is dataclass and source value is dict, recursively merge
        if is_dataclass(target_value) and isinstance(source_value, dict):
            _merge_dict_to_dataclass(target_value, source_value, current_path)
            log_rank_0(f"  ↳ Merged {current_path} (recursive)")
        else:
            # For non-dataclass fields, check type compatibility before assignment
            # Get expected type from target field
            target_type = type(target_value)
            source_type = type(source_value)

            # Allow assignment if types match or target is None (uninitialized)
            # if target_value is None or source_type == target_type or isinstance(source_value, target_type):
            if source_type == target_type or isinstance(source_value, target_type):
                setattr(target, field_name, source_value)
                log_rank_0(f"  ↳ Set {current_path} = {source_value}")
            else:
                # Type mismatch - log warning and skip
                log_rank_0(
                    f"  ⚠️  Skipping {current_path}: type mismatch "
                    f"(target={target_type.__name__}, source={source_type.__name__})"
                )


def load_recipe_config(backend_args: SimpleNamespace) -> Any:
    normalize_megatron_bridge_dataset_args(backend_args)

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

    # Convert backend_args to dict once (used for both recipe call and config override)
    backend_dict = namespace_to_dict(backend_args)

    # Call recipe function with filtered dict
    config_container = auto_filter_and_call(recipe_func, backend_dict)
    log_rank_0(f"Successfully loaded recipe: {full_module_path}.{function_name}()")
    # log_dict_aligned("[debug]ConfigContainer", config_container.to_dict())

    # Validate return type
    from megatron.bridge.training.config import ConfigContainer

    assert isinstance(config_container, ConfigContainer), (
        f"Recipe function '{full_module_path}.{function_name}()' must return "
        f"ConfigContainer, but returned {type(config_container).__name__}"
    )

    log_rank_0("Applying backend_args overrides to config_container...")
    _merge_dict_to_dataclass(config_container, backend_dict, "config_container")

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
