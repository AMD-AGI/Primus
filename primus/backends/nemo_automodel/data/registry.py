###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Registry of offline cache builders for the AutoModel diffusion backend.

AutoModel diffusion models can train from a **pre-encoded flat cache** — the
VAE and text encoder run once, offline, so training needs neither their weights
nor their memory. Each model that supports this registers a builder here, and
``primus data automodel-cache --model <name>`` dispatches through it.

Entries are **dotted strings, not imports**, for two reasons: the CLI can list
the available models without importing any of them, and a builder that needs an
optional dependency only fails when it is actually asked for.

This is deliberately separate from ``primus data diffusion-*``, which prepares
Energon WebDataset shards for the Megatron backend. Different backend, different
on-disk format; sharing one code path would couple two unrelated pipelines.
"""
from __future__ import annotations

import importlib
from typing import Callable, Dict

# model name -> "<module>:<callable>"
CACHE_BUILDERS: Dict[str, str] = {
    "ideogram4": "primus.backends.nemo_automodel.models.ideogram4.data.build:build_cache",
}


def available_models() -> list[str]:
    """Model names that can build an offline cache."""
    return sorted(CACHE_BUILDERS)


def get_cache_builder(model: str) -> Callable[..., dict]:
    """Import and return the cache builder registered for ``model``."""
    try:
        target = CACHE_BUILDERS[model]
    except KeyError:
        raise ValueError(
            f"no cache builder registered for model {model!r}; available: {', '.join(available_models())}"
        ) from None

    module_path, _, attribute = target.partition(":")
    module = importlib.import_module(module_path)
    return getattr(module, attribute)
