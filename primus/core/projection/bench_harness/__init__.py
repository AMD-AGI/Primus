###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Per-backend benchmark adapters for the projection tool.

The projection can *read* any backend whose config adapter is registered in
:mod:`primus.core.projection.frameworks`.  Benchmark-anchored projection asks
for more: a real model, built at the bench topology, whose layers the harness
can find and call.  A :class:`~.base.BenchModelAdapter` supplies exactly that,
and this module is where backends are looked up.

Registration is lazy and mirrors the config-adapter registry, so importing the
projection costs nothing until a benchmark actually runs.
"""

from typing import Optional, Tuple

from primus.core.projection.bench_harness.base import (
    ALL_KINDS,
    ATTENTION,
    EMBEDDING,
    LAYER,
    MLP,
    MOE,
    OUTPUT,
    BenchContext,
    BenchInputs,
    BenchModelAdapter,
    LayerParts,
    ModelParts,
)

__all__ = [
    "ALL_KINDS",
    "ATTENTION",
    "EMBEDDING",
    "LAYER",
    "MLP",
    "MOE",
    "OUTPUT",
    "BenchContext",
    "BenchInputs",
    "BenchModelAdapter",
    "LayerParts",
    "ModelParts",
    "available_bench_frameworks",
    "get_model_adapter",
    "register_model_adapter",
    "resolve_model_adapter",
]

_MODEL_ADAPTER_REGISTRY: "dict[str, BenchModelAdapter]" = {}


def register_model_adapter(name: str, adapter: BenchModelAdapter) -> None:
    """Register *adapter* under framework *name* (case-insensitive)."""
    if not name or not isinstance(name, str):
        raise ValueError(f"Framework name must be a non-empty string, got {name!r}")
    if not isinstance(adapter, BenchModelAdapter):
        raise TypeError(f"Bench adapter for '{name}' must be a BenchModelAdapter, got {adapter!r}")
    _MODEL_ADAPTER_REGISTRY[name.lower().strip()] = adapter


def get_model_adapter(name: str) -> Optional[BenchModelAdapter]:
    """Return the adapter registered for *name*, or ``None``."""
    if not name:
        return None
    _ensure_builtins_registered()
    return _MODEL_ADAPTER_REGISTRY.get(name.lower().strip())


def available_bench_frameworks() -> Tuple[str, ...]:
    """Return the sorted names of every backend the projection can benchmark."""
    _ensure_builtins_registered()
    return tuple(sorted(_MODEL_ADAPTER_REGISTRY))


def _ensure_builtins_registered() -> None:
    """Register the in-tree bench adapters on first use (idempotent)."""
    if getattr(_ensure_builtins_registered, "_done", False):
        return
    _ensure_builtins_registered._done = True

    from primus.core.projection.bench_harness.megatron import MegatronBenchAdapter
    from primus.core.projection.bench_harness.torchtitan import TorchTitanBenchAdapter

    register_model_adapter("megatron", MegatronBenchAdapter())
    register_model_adapter("torchtitan", TorchTitanBenchAdapter())


def resolve_model_adapter(name: str) -> BenchModelAdapter:
    """Return the bench adapter for framework *name*.

    Raises:
        NotImplementedError: if *name* has no adapter, i.e. the projection can
            read that backend's config but cannot drive its model.
    """
    adapter = get_model_adapter(name)
    if adapter is None:
        supported = ", ".join(available_bench_frameworks())
        raise NotImplementedError(
            f"No projection benchmark adapter for framework {name!r}. "
            f"Benchmarkable frameworks: {supported}. Register another by calling "
            "primus.core.projection.bench_harness.register_model_adapter(name, adapter)."
        )
    return adapter
