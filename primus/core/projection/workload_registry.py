###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Workload (framework) registry for the projection tool.

The projection driver used to hardcode ``get_language_model_profiler_spec`` as
the one and only top-level profiler tree, which silently assumed every workload
is a Megatron-style dense/MoE language model.  That coupling is the single thing
that blocks projecting any non-language-model workload (e.g. a TorchRec/DLRM
ranker) even when every underlying compute/memory op is already modelled.

This module breaks that coupling: selection is registry-driven and no workload
name is hardcoded at the call sites.  A workload maps a ``framework`` string to
a spec factory -- a callable ``(config: TrainingConfig) -> ModuleProfilerSpec``
that returns the root of the profiler tree for that workload.  The driver builds
the profiler from whatever spec the registry returns; the rest of the pipeline
(``build_profiler`` and the ``ModuleProfilerSpec`` walk) is already workload
agnostic.

Built-in workloads (``megatron``, ``torchrec_dlrm``) self-register on first use.
Out-of-tree code can register additional workloads by importing this module and
calling ``register_workload(name, spec_factory)`` at its own import time.

Selection at call time: explicit ``framework`` arg -> ``config.framework`` ->
``PRIMUS_WORKLOAD`` env var -> ``"megatron"``.
"""

import os
from typing import Callable, Dict, Optional, Tuple

# A workload spec factory turns a projection TrainingConfig into the root
# ModuleProfilerSpec of that workload's profiler tree.
WorkloadSpecFactory = Callable[..., "object"]

_WORKLOAD_REGISTRY: Dict[str, WorkloadSpecFactory] = {}


def register_workload(name: str, spec_factory: WorkloadSpecFactory) -> None:
    """Register a workload spec factory under *name* (case-insensitive).

    Safe to call multiple times; the last registration for a name wins.
    """
    if not name or not isinstance(name, str):
        raise ValueError(f"Workload framework name must be a non-empty string, got {name!r}")
    if not callable(spec_factory):
        raise TypeError(f"Workload spec factory for '{name}' must be callable, got {spec_factory!r}")
    _WORKLOAD_REGISTRY[name.lower().strip()] = spec_factory


def get_workload_spec_factory(name: str) -> Optional[WorkloadSpecFactory]:
    """Return the registered spec factory for *name*, or ``None`` if not registered."""
    if not name:
        return None
    _ensure_builtins_registered()
    return _WORKLOAD_REGISTRY.get(name.lower().strip())


def available_workloads() -> Tuple[str, ...]:
    """Return the sorted names of all registered workloads."""
    _ensure_builtins_registered()
    return tuple(sorted(_WORKLOAD_REGISTRY))


def _ensure_builtins_registered() -> None:
    """Register the in-tree workloads on first use (idempotent).

    Imports are done lazily here (not at module top level) so importing this
    registry has no hard dependency on the profiler package, which pulls in the
    training config and every module profiler.  A function attribute (rather than
    a module global) tracks the one-time registration.
    """
    if getattr(_ensure_builtins_registered, "_done", False):
        return
    _ensure_builtins_registered._done = True

    # Language model (Megatron) -- the default; preserves historical behaviour.
    from primus.core.projection.module_profilers.language_model import (
        get_language_model_profiler_spec,
    )

    register_workload("megatron", get_language_model_profiler_spec)

    # DLRM-v4 (TorchRec / HSTU ranker), under every alias the config converter
    # accepts so ``framework: dlrm`` / ``torchrec`` all resolve.
    from primus.core.projection.module_profilers.dlrm import get_dlrm_profiler_spec

    for alias in ("torchrec_dlrm", "torchrec", "dlrm", "dlrm_v4"):
        register_workload(alias, get_dlrm_profiler_spec)


def resolve_top_level_spec(config, framework: Optional[str] = None):
    """Resolve and build the top-level ``ModuleProfilerSpec`` for *config*.

    This is the single seam the projection drivers call instead of hardcoding
    ``get_language_model_profiler_spec``.

    Selection order: explicit ``framework`` arg -> ``config.framework`` ->
    ``PRIMUS_WORKLOAD`` env var -> ``"megatron"``.

    Raises:
        ValueError: if the selected framework has no registered workload.
    """
    _ensure_builtins_registered()

    name = framework or getattr(config, "framework", None) or os.getenv("PRIMUS_WORKLOAD", None)
    if name is not None:
        name = str(name).lower().strip()
    if not name:
        name = "megatron"

    factory = get_workload_spec_factory(name)
    if factory is None:
        supported = ", ".join(available_workloads()) or "megatron"
        raise ValueError(
            f"Unknown projection workload framework: '{name}'. Supported workloads: {supported}. "
            "Register additional workloads by calling register_workload(name, spec_factory)."
        )

    if int(os.getenv("RANK", "0")) == 0:
        print(f"[Primus:Projection] Using workload: {name}")
    return factory(config)
