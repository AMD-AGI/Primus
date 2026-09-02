###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Registry of offline cache builders for the AutoModel diffusion backend.

These models can train from a pre-encoded flat cache: the autoencoder and text
encoder run once, offline, so training itself needs neither their weights nor
their memory. Each model that supports this registers a builder here, and
``primus data automodel-cache --model <name>`` dispatches through it.

ENTRIES ARE DOTTED STRINGS, NOT IMPORTS, for two reasons that both matter at the
command line. The CLI builds its ``--model`` choices from this registry, so it has
to be able to list the models without importing any of them -- and a cache builder
pulls in an autoencoder, a text encoder and an image library. And because the
import is deferred to the moment a builder is actually asked for, a model whose
optional dependencies are missing does not break ``--help`` or any other model's
build.

This is deliberately separate from the ``primus data diffusion-*`` commands, which
prepare sharded datasets for the Megatron backend. Different backend, different
on-disk format, no shared code path; one registry spanning both would only couple
two pipelines that have nothing to say to each other.
"""
from __future__ import annotations

import importlib
from typing import Callable, Dict, List

# model name -> "<module>:<callable>"
CACHE_BUILDERS: Dict[str, str] = {
    "ideogram4": "primus.backends.nemo_automodel.models.ideogram4.data.build:build_cache",
}


def available_models() -> List[str]:
    """The model names that can build an offline cache.

    Sorted, because this is what the CLI shows in its help and its ``--model``
    choices, and a listing whose order depends on dictionary insertion is a
    gratuitously unstable thing to put in user-facing output.
    """
    return sorted(CACHE_BUILDERS)


def get_cache_builder(model: str) -> Callable[..., dict]:
    """Import and return the builder registered for ``model``."""
    try:
        target = CACHE_BUILDERS[model]
    except KeyError:
        raise ValueError(
            f"no cache builder is registered for model {model!r}. Available: "
            f"{', '.join(available_models())}"
        ) from None

    module_path, _, attribute = target.partition(":")
    module = importlib.import_module(module_path)
    return getattr(module, attribute)
