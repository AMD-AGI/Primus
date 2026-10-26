###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
import importlib
from functools import partial

from primus.core.utils.module_utils import log_rank_0


def lazy_import(paths, symbol, log_prefix="[Primus]"):
    """
    Try to import a symbol from a list of module paths.

    Args:
        paths (list[str]): candidate module paths
        symbol (str): the attribute/class/function name
        log_prefix (str): prefix for logging

    Returns:
        The imported symbol

    Raises:
        ImportError: if symbol not found in any given path
    """
    for path in paths:
        try:
            mod = importlib.import_module(path)
            log_rank_0(f"{log_prefix} Loaded {symbol} from {path}")
            return getattr(mod, symbol)
        except ImportError:
            continue
    raise ImportError(f"{log_prefix} {symbol} not found in any of: {paths}")


def get_model_provider(model_type="gpt"):
    """
    Resolve the model_provider for a ``model_type`` registered in
    ``primus/backends/megatron/core/models/registry.py``.

    Primus-owned families bind the shared Primus ``model_provider`` to the shared
    builder. Upstream families (``gpt``, ``mamba``) resolve across Megatron versions:

    - New:   model_provider + gpt_builder/mamba_builder
    - Mid:   model_provider only
    - Old:   pretrain_gpt.model_provider / pretrain_mamba.model_provider

    An unregistered ``model_type`` raises rather than building ``gpt``: a misspelt
    family would otherwise train a different model. Trainers for unregistered
    values (the diffusion trainers) supply their own ``model_provider``.
    """
    from primus.backends.megatron.core.models.registry import (
        DEFAULT_MODEL_TYPE,
        MODEL_FAMILIES,
    )

    log_prefix = "[Primus][MegatronCompat]"
    model_type = model_type or DEFAULT_MODEL_TYPE
    family = MODEL_FAMILIES.get(model_type)
    if family is None:
        raise ValueError(
            f"{log_prefix} model_type {model_type!r} is not registered in "
            "primus/backends/megatron/core/models/registry.py; "
            f"registered: {', '.join(sorted(MODEL_FAMILIES))}."
        )

    if not family.builder:
        builder_module = importlib.import_module("primus.backends.megatron.core.models.builder")
        log_rank_0(f"{log_prefix} Loaded Primus model_provider + builder for model_type {model_type!r}")
        return partial(builder_module.model_provider, partial(builder_module.build_model, model_type))

    model_provider = lazy_import(["model_provider", family.training], "model_provider", log_prefix=log_prefix)
    builder_module, _, builder_name = family.builder.partition(":")
    try:
        return partial(model_provider, lazy_import([builder_module], builder_name, log_prefix=log_prefix))
    except ImportError:
        return model_provider


def get_custom_fsdp():
    """
    Resolve FullyShardedDataParallel across Megatron versions.

    - New: megatron.core.distributed.fsdp.mcore_fsdp_adapter.FullyShardedDataParallel
    - Old: megatron.core.distributed.custom_fsdp.FullyShardedDataParallel
    """
    return lazy_import(
        [
            "megatron.core.distributed.fsdp.mcore_fsdp_adapter",
            "megatron.core.distributed.custom_fsdp",
        ],
        "FullyShardedDataParallel",
        log_prefix="[Primus][MegatronCompat]",
    )
