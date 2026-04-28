###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
import importlib
from functools import partial

from primus.modules.module_utils import log_rank_0


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
    Resolve model_provider across Megatron versions and model types.

    Args:
        model_type (str): Type of model - 'gpt', 'mamba', or 'deepseek_v4'. Defaults to 'gpt'.

    - New:   model_provider + gpt_builder/mamba_builder
    - Mid:   model_provider only
    - Old:   pretrain_gpt.model_provider / pretrain_mamba.model_provider
    """
    # Primus-owned: DeepSeek-V4 (Phase 2 stub; full V4 wiring lands in Phase 3+)
    if model_type == "deepseek_v4":
        deepseek_v4_module = importlib.import_module(
            "primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_builders"
        )
        log_rank_0(
            "[Primus][MegatronCompat] Loaded DeepSeek-V4 model_provider + builder "
            f"from {deepseek_v4_module.__name__}"
        )
        return partial(
            deepseek_v4_module.model_provider,
            deepseek_v4_module.deepseek_v4_builder,
        )

    # Upstream Megatron-LM model types
    if model_type == "mamba":
        model_provider = lazy_import(
            ["model_provider", "pretrain_mamba"], "model_provider", log_prefix="[Primus][MegatronCompat]"
        )
        # Try to import mamba_builder (for Mamba models)
        try:
            mamba_builder = lazy_import(
                ["mamba_builders"], "mamba_builder", log_prefix="[Primus][MegatronCompat]"
            )
            return partial(model_provider, mamba_builder)
        except ImportError:
            return model_provider
    else:
        # Default GPT behavior
        model_provider = lazy_import(
            ["model_provider", "pretrain_gpt"], "model_provider", log_prefix="[Primus][MegatronCompat]"
        )

        # Try to import gpt_builder (only exists in newer versions)
        try:
            gpt_builder = lazy_import(["gpt_builders"], "gpt_builder", log_prefix="[Primus][MegatronCompat]")
            return partial(model_provider, gpt_builder)
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
