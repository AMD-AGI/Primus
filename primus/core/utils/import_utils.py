###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
import importlib
from functools import partial
from typing import NamedTuple

from primus.core.utils.module_utils import log_rank_0


class PrimusModelProvider(NamedTuple):
    """One Primus-owned model type: where its builders live and the name the logs use."""

    module: str  # builders module, defining model_provider and the builder
    builder: str  # name of the builder function in that module
    display_name: str


# Primus-owned causal-LM model types: model_type -> PrimusModelProvider (builders module, builder
# name, display name). get_model_provider binds the module's model_provider to its builder; the Megatron
# pretrain trainer reuses pretrain_gpt's forward_step and datasets for every type listed here, so a
# type that needs its own forward_step or datasets must not be listed (it needs its own trainer
# branch). DeepSeek-V4 is still a Phase 2 stub: full V4 wiring lands in Phase 3+.
# Register a new Primus-owned model type here only. Other modules use is_primus_model_type and
# primus_model_display_name, not the table.
_PRIMUS_MODEL_PROVIDERS = {
    "deepseek_v4": PrimusModelProvider(
        "primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_builders",
        "deepseek_v4_builder",
        "DeepSeek-V4",
    ),
    "glm5_next": PrimusModelProvider(
        "primus.backends.megatron.core.models.glm5_next.glm5_next_builders",
        "glm5_next_builder",
        "GLM5-Next",
    ),
    "inkling": PrimusModelProvider(
        "primus.backends.megatron.core.models.inkling.inkling_builders",
        "inkling_builder",
        "Inkling",
    ),
    "kimi_k3": PrimusModelProvider(
        "primus.backends.megatron.core.models.kimi_k3.kimi_k3_builders",
        "kimi_k3_builder",
        "Kimi-K3",
    ),
}


def is_primus_model_type(model_type) -> bool:
    """True for a Primus-owned model type (a key of the provider table)."""
    return model_type in _PRIMUS_MODEL_PROVIDERS


def primus_model_display_name(model_type: str) -> str:
    """The display name of a Primus-owned model type (KeyError for any other type)."""
    return _PRIMUS_MODEL_PROVIDERS[model_type].display_name


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
        model_type (str): 'gpt', 'mamba', or any Primus-owned type in _PRIMUS_MODEL_PROVIDERS.
            Defaults to 'gpt'.

    - New:   model_provider + gpt_builder/mamba_builder
    - Mid:   model_provider only
    - Old:   pretrain_gpt.model_provider / pretrain_mamba.model_provider
    """
    # Primus-owned types (such as model_type == "inkling") resolve from _PRIMUS_MODEL_PROVIDERS.
    if is_primus_model_type(model_type):
        entry = _PRIMUS_MODEL_PROVIDERS[model_type]
        module = importlib.import_module(entry.module)
        missing = [n for n in ("model_provider", entry.builder) if not hasattr(module, n)]
        if missing:
            raise AttributeError(
                f"[Primus] model_type={model_type!r}: _PRIMUS_MODEL_PROVIDERS points at {entry.module}, "
                f"which does not define {missing}"
            )
        log_rank_0(
            f"[Primus][MegatronCompat] Loaded {entry.display_name} model_provider + builder "
            f"from {module.__name__}"
        )
        return partial(module.model_provider, getattr(module, entry.builder))

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
