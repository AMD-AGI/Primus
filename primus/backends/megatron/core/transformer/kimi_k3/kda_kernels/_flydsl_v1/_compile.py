###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Inductor wrapper for the torch glue between this backend's FlyDSL kernels.

Profiling the production backward (`[1, 4096, 96, 128, 128]` bf16) put **49 % of
it — 5083 µs of 10434 — in torch elementwise glue spread over ~80 launches**,
against `fla`'s 13 hand-written kernels for the whole pass. Every one of those
ops is a full HBM round trip on a 100–200 MB tensor for one multiply of real
arithmetic, and they are all elementwise chains, which is precisely what
Inductor fuses. Measured on the chunk-prep adjoint in isolation
(`bench/probe_torch_tricks.py`): **17 launches -> 3**, worst output difference
1.9e-06 against the eager chain, which is fp32 reassociation and not a
different computation.

So the glue is wrapped here rather than hand-written as more FlyDSL kernels.
The kernels that *are* hand-written stay hand-written: they do things Inductor
cannot express (the block-referenced decay exponents, the on-chip triangular
solve, the sequential state recurrence).

Three things this module is responsible for.

``compiled``
    Wraps a pure function. ``dynamic=False`` because every call site here is
    shape-specialised anyway and a dynamic graph loses the vectorisation that
    makes the fusion worth having.

The recompile limit
    Dynamo recompiles per input shape **and per grad mode** — every one of these
    regions runs once under ``no_grad`` in the forward and once under
    ``enable_grad`` in the backward's recompute, which is a
    ``GLOBAL_STATE changed: grad_mode`` guard failure and therefore a second
    entry. Seven geometries at two dtypes is then 28 variants against a default
    limit of 8, and exceeding it falls back to eager *silently* — measured as
    `curve_mbs16` keeping 83 backward launches where `prod_T4096` got 67. Raised
    once, here, under every name the knob has had (`recompile_limit` in current
    torch, `cache_size_limit` before it).

The kill switch
    ``K3P_KDA_FLYDSL_COMPILE=0`` returns the eager function untouched. Inductor
    is a large dependency to put on a training hot path; being able to bisect
    against it without editing code is worth one environment variable.
"""

from __future__ import annotations

import os
from typing import Any, Callable, TypeVar

import torch

__all__ = ["compiled", "compile_enabled"]

_F = TypeVar("_F", bound=Callable[..., Any])

_MIN_CACHE_SIZE = 1024
_LIMIT_FIELDS = (
    "recompile_limit",
    "accumulated_recompile_limit",
    "cache_size_limit",
    "accumulated_cache_size_limit",
)
_configured = False


def compile_enabled() -> bool:
    """``False`` when ``K3P_KDA_FLYDSL_COMPILE`` is set to a falsey value."""
    return os.environ.get("K3P_KDA_FLYDSL_COMPILE", "1").lower() not in ("0", "false", "no", "off")


def _configure() -> None:
    global _configured
    if _configured:
        return
    _configured = True
    try:
        import torch._dynamo.config as dynamo_config

        for field in _LIMIT_FIELDS:
            if hasattr(dynamo_config, field) and getattr(dynamo_config, field) < _MIN_CACHE_SIZE:
                setattr(dynamo_config, field, _MIN_CACHE_SIZE)
    except Exception:  # noqa: BLE001 - a missing knob must not break the backend
        pass


def compiled(fn: _F) -> _F:
    """Return ``fn`` compiled by Inductor, or ``fn`` itself if compilation is off.

    The compile is lazy: ``torch.compile`` is applied on first call, so importing
    this backend does not drag Inductor in, and a build where ``torch.compile``
    is unavailable degrades to eager instead of failing to import.
    """
    if not compile_enabled():
        return fn

    holder: dict = {}

    def wrapper(*args, **kwargs):
        target = holder.get("fn")
        if target is None:
            _configure()
            try:
                target = torch.compile(fn, dynamic=False)
            except Exception:  # noqa: BLE001
                target = fn
            holder["fn"] = target
        return target(*args, **kwargs)

    wrapper.__name__ = getattr(fn, "__name__", "compiled")
    wrapper.__doc__ = fn.__doc__
    wrapper._eager_fn = fn  # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]
