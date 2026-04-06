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


def _call_recipe_function(recipe_func: Callable[..., Any], backend_dict: Dict[str, Any]) -> Any:
    """
    Invoke a recipe config factory without passing the entire Primus module dict when the
    recipe takes no (or few) explicit parameters.

    Megatron-Bridge diffusion recipes in Primus often return a default ``ConfigContainer`` and
    rely on ``_apply_flat_config_knobs`` afterward. Passing 50+ merged module keys into
    ``auto_filter_and_call`` hits the retry cap and fails with leftover keys (e.g. ``model``,
    ``num_workers``).
    """
    import inspect

    sig = inspect.signature(recipe_func)
    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if has_varkw:
        return auto_filter_and_call(recipe_func, backend_dict)

    call_kwargs: Dict[str, Any] = {}
    missing_required: list[str] = []
    for name, param in sig.parameters.items():
        if param.kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            continue
        if name in backend_dict:
            call_kwargs[name] = backend_dict[name]
        elif param.default is inspect.Parameter.empty:
            missing_required.append(name)

    if missing_required:
        log_rank_0(
            f"Recipe {recipe_func.__name__} missing required params {missing_required!r}; "
            "falling back to auto_filter_and_call."
        )
        return auto_filter_and_call(recipe_func, backend_dict)

    return recipe_func(**call_kwargs)


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


def _dataclass_field_names(obj: Any) -> frozenset[str]:
    from dataclasses import fields, is_dataclass

    if not is_dataclass(obj):
        return frozenset()
    return frozenset(f.name for f in fields(obj))


def _apply_flat_config_knobs(config_container: Any, backend_dict: dict) -> None:
    """
    Pop Primus YAML "flat knobs" and write them into the nested Megatron-Bridge
    ``ConfigContainer`` (model / train / checkpoint / ddp / dataset / …).

    This keeps experiment overrides ergonomic (same style as SFT posttrain configs)
    without requiring users to nest under ``model:``, ``train:``, etc.
    """
    d = backend_dict

    def _pop_set(obj: Any, key: str, log_prefix: str) -> bool:
        if key not in d:
            return False
        if key not in _dataclass_field_names(obj):
            return False
        value = d.pop(key)
        setattr(obj, key, value)
        log_rank_0(f"  ↳ flat knob {key!r} → {log_prefix}.{key} = {value!r}")
        return True

    model = config_container.model
    train = config_container.train
    checkpoint = config_container.checkpoint
    ddp = config_container.ddp
    dataset = getattr(config_container, "dataset", None)
    optimizer = config_container.optimizer
    scheduler = config_container.scheduler
    logger_cfg = config_container.logger

    # --- Model (parallelism, recompute, sequence length) ---
    model_keys = (
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "pipeline_dtype",
        "virtual_pipeline_model_parallel_size",
        "context_parallel_size",
        "sequence_parallel",
        "recompute_granularity",
        "recompute_method",
        "recompute_num_layers",
    )
    for k in model_keys:
        _pop_set(model, k, "model")

    if "seq_length" in d and "seq_length" in _dataclass_field_names(model):
        v = d.pop("seq_length")
        model.seq_length = v
        log_rank_0(f"  ↳ flat knob 'seq_length' → model.seq_length = {v!r}")
        if dataset is not None and "seq_length" in _dataclass_field_names(dataset):
            dataset.seq_length = v
            log_rank_0(f"  ↳ flat knob 'seq_length' → dataset.seq_length = {v!r}")

    # --- Training loop (batch sizes handled below for train + dataset) ---
    train_keys = (
        "train_iters",
        "train_samples",
        "skip_train",
        "eval_interval",
        "eval_iters",
        "rampup_batch_size",
        "decrease_batch_size_if_needed",
        "manual_gc",
        "manual_gc_interval",
        "manual_gc_eval",
    )
    for k in train_keys:
        _pop_set(train, k, "train")

    # --- Primus: validation logging / early-stop (extra attrs on TrainingConfig) ---
    for k in (
        "val_stop_loss",
        "val_stop_loss_key",
        "val_stop_mode",
        "eval_verbose",
        "eval_skip_posttrain_test",
    ):
        if k not in d:
            continue
        v = d.pop(k)
        setattr(train, k, v)
        log_rank_0(f"  ↳ flat knob {k!r} → train.{k} = {v!r}")

    # Batch sizes: one knob updates both train and dataset when both define the field
    for bk in ("global_batch_size", "micro_batch_size"):
        if bk not in d:
            continue
        v = d.pop(bk)
        if bk in _dataclass_field_names(train):
            setattr(train, bk, v)
            log_rank_0(f"  ↳ flat knob {bk!r} → train.{bk} = {v!r}")
        if dataset is not None and bk in _dataclass_field_names(dataset):
            setattr(dataset, bk, v)
            log_rank_0(f"  ↳ flat knob {bk!r} → dataset.{bk} = {v!r}")

    # --- Checkpointing ---
    for k in (
        "save_interval",
        "save",
        "load",
        "pretrained_checkpoint",
        "ckpt_format",
        "fully_parallel_save",
        "save_optim",
        "save_rng",
        "load_optim",
        "load_rng",
        "finetune",
        "most_recent_k",
    ):
        _pop_set(checkpoint, k, "checkpoint")

    # --- DDP ---
    for k in (
        "use_megatron_fsdp",
        "check_for_nan_in_grad",
        "grad_reduce_in_fp32",
        "overlap_grad_reduce",
        "overlap_param_gather",
        "average_in_collective",
        "use_distributed_optimizer",
    ):
        _pop_set(ddp, k, "ddp")

    # --- Optimizer / scheduler (common Primus knob names) ---
    if "finetune_lr" in d and "lr" in _dataclass_field_names(optimizer) and "lr" not in d:
        d["lr"] = d.pop("finetune_lr")
    for k in ("lr", "min_lr", "weight_decay", "adam_beta1", "adam_beta2"):
        _pop_set(optimizer, k, "optimizer")

    for k in ("lr_warmup_iters", "lr_decay_iters", "lr_decay_style", "lr_warmup_fraction", "lr_warmup_init"):
        _pop_set(scheduler, k, "scheduler")

    # --- Logger / W&B ---
    for k in ("wandb_project", "wandb_entity", "wandb_exp_name", "log_interval", "tensorboard_dir"):
        _pop_set(logger_cfg, k, "logger")

    # --- Dataset (Energon / WebDataset root, workers) ---
    if "data_path" in d and dataset is not None and "path" in _dataclass_field_names(dataset):
        v = d.pop("data_path")
        dataset.path = v
        log_rank_0(f"  ↳ flat knob 'data_path' → dataset.path = {v!r}")
    if dataset is not None:
        _pop_set(dataset, "num_workers", "dataset")

    # --- Mixed precision (Primus alias + top-level Megatron-Bridge field) ---
    if "precision_config" in d:
        spec = d.pop("precision_config")
        if spec is not None:
            from megatron.bridge.training.mixed_precision import get_mixed_precision_config

            if isinstance(spec, str):
                config_container.mixed_precision = get_mixed_precision_config(spec)
            else:
                config_container.mixed_precision = spec
            log_rank_0(f"  ↳ flat knob 'precision_config' → mixed_precision = {spec!r}")

    if "mixed_precision" in d and "mixed_precision" in _dataclass_field_names(config_container):
        v = d.pop("mixed_precision")
        if isinstance(v, str):
            from megatron.bridge.training.mixed_precision import get_mixed_precision_config

            config_container.mixed_precision = get_mixed_precision_config(v)
        else:
            config_container.mixed_precision = v
        log_rank_0(f"  ↳ flat knob 'mixed_precision' → mixed_precision = {v!r}")

    if "comm_overlap_config" in d:
        v = d.pop("comm_overlap_config")
        if v is not None and "comm_overlap" in _dataclass_field_names(config_container):
            config_container.comm_overlap = v
            log_rank_0("  ↳ flat knob 'comm_overlap_config' → comm_overlap")


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
    flavor = getattr(backend_args, "flavor", None)
    assert flavor, "Flavor must be specified for Megatron-Bridge backend"

    recipe_module = getattr(backend_args, "recipe_module", None)
    recipe = getattr(backend_args, "recipe", None)

    # Standard LLM recipes live under megatron.bridge.recipes.<recipe>.
    # Diffusion (FLUX/WAN) recipes shipped in Primus use primus.diffusion.recipes.*
    # via recipe_module (full dotted module path).
    if recipe_module:
        full_module_path = recipe_module
    else:
        assert recipe, (
            "Recipe must be specified for Megatron-Bridge backend, or set "
            "`recipe_module` to a full module path (e.g. primus.diffusion.recipes.flux.flux)."
        )
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

    # Call recipe function (do not pass full module merge into zero-arg factories)
    config_container = _call_recipe_function(recipe_func, backend_dict)
    log_rank_0(f"Successfully loaded recipe: {full_module_path}.{function_name}()")
    # log_dict_aligned("[debug]ConfigContainer", config_container.to_dict())

    # Validate return type
    from megatron.bridge.training.config import ConfigContainer

    assert isinstance(config_container, ConfigContainer), (
        f"Recipe function '{full_module_path}.{function_name}()' must return "
        f"ConfigContainer, but returned {type(config_container).__name__}"
    )

    log_rank_0("Applying backend_args overrides to config_container...")
    # Merge nested structures first, then apply flat knobs. If we did the reverse, a nested
    # ``train`` dict from deep-merged module defaults (e.g. ``eval_iters: 0``) would overwrite
    # flat experiment keys like ``eval_iters`` / ``eval_interval`` and disable validation.
    _merge_dict_to_dataclass(config_container, backend_dict, "config_container")
    _apply_flat_config_knobs(config_container, backend_dict)

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
