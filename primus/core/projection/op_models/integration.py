###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Where registered op models enter the built-in profilers.

The projection prices one representative layer per kind and multiplies by the
layer count, so an op that runs on a fraction ``f`` of the layers enters as a
per-layer average: ``(1 - f) * built_in + f * op`` for a replacing slot, and
``+ f * op`` for an additive one.
"""

from __future__ import annotations

import os
from typing import Callable, List, Optional, Tuple

from .base import (
    SLOT_ATTENTION,
    SLOT_ATTENTION_EXTRA,
    SLOT_LAYER_EXTRA,
    SLOT_MLP,
    OpContext,
    OpCost,
    OpModel,
)
from .registry import active_op_models

_ANNOUNCED: set = set()


def _announce(op: OpModel, frac: float, cost: OpCost) -> None:
    if int(os.getenv("RANK", "0")) != 0:
        return
    key = (op.name, round(frac, 4))
    if key in _ANNOUNCED and not os.getenv("PRIMUS_PRINT_LAYER_BREAKDOWN"):
        return
    _ANNOUNCED.add(key)
    parts = " ".join(f"{k}={f:.3f}/{b:.3f}" for k, (f, b) in cost.breakdown.items())
    print(
        f"[Primus:OpModel] {op.name} ({op.slot}) on {frac:.0%} of layers: "
        f"fwd={cost.fwd_ms:.3f} ms bwd={cost.bwd_ms:.3f} ms" + (f"  [{parts}]" if parts else "")
    )


def _fractioned(ops: List[OpModel], model_config) -> List[Tuple[OpModel, float]]:
    out = []
    for op in ops:
        f = max(0.0, min(1.0, float(op.layer_fraction(model_config))))
        if f > 0.0:
            out.append((op, f))
    return out


def _blend(
    replacing: List[Tuple[OpModel, float]],
    extra: List[Tuple[OpModel, float]],
    ctx: OpContext,
    built_in: Callable[[], Tuple[float, float]],
) -> Tuple[float, float]:
    f_rep = min(1.0, sum(f for _, f in replacing))
    fwd = bwd = 0.0
    if f_rep < 1.0:
        b_fwd, b_bwd = built_in()
        fwd, bwd = (1.0 - f_rep) * b_fwd, (1.0 - f_rep) * b_bwd
    for op, f in replacing + extra:
        c = op.cost(ctx)
        _announce(op, f, c)
        fwd += f * c.fwd_ms
        bwd += f * c.bwd_ms
    return fwd, bwd


def attention_op_models(model_config) -> List[OpModel]:
    return active_op_models(model_config, slots=(SLOT_ATTENTION, SLOT_ATTENTION_EXTRA))


def simulate_attention_with_op_models(
    ops: List[OpModel],
    config,
    batch_size: int,
    seq_len: int,
    gemm_backend,
    sdpa_backend,
    built_in: Callable[[], Tuple[float, float]],
) -> Tuple[float, float]:
    """Attention-block ``(fwd, bwd)`` ms with the op models blended in."""
    mc = config.model_config
    ctx = OpContext(config, batch_size, seq_len, gemm_backend, sdpa_backend)
    rep = _fractioned([o for o in ops if o.slot == SLOT_ATTENTION], mc)
    ext = _fractioned([o for o in ops if o.slot == SLOT_ATTENTION_EXTRA], mc)
    return _blend(rep, ext, ctx, built_in)


def apply_layer_op_models(
    config,
    batch_size: int,
    seq_len: int,
    gemm_backend,
    sdpa_backend,
    mlp_fwd: float,
    mlp_bwd: float,
    layer_kind: str,
) -> Tuple[float, float, float, float]:
    """``(mlp_fwd, mlp_bwd, extra_fwd, extra_bwd)`` after MLP / layer-extra ops."""
    mc = config.model_config
    ops = active_op_models(mc, slots=(SLOT_MLP, SLOT_LAYER_EXTRA))
    if not ops:
        return mlp_fwd, mlp_bwd, 0.0, 0.0
    ctx = OpContext(config, batch_size, seq_len, gemm_backend, sdpa_backend, layer_kind=layer_kind)
    rep = _fractioned([o for o in ops if o.slot == SLOT_MLP], mc)
    if rep:
        mlp_fwd, mlp_bwd = _blend(rep, [], ctx, lambda: (mlp_fwd, mlp_bwd))
    extra_fwd, extra_bwd = _blend(
        [], _fractioned([o for o in ops if o.slot == SLOT_LAYER_EXTRA], mc), ctx, lambda: (0.0, 0.0)
    )
    return mlp_fwd, mlp_bwd, extra_fwd, extra_bwd


def blended_memory_term(
    ops: List[OpModel], config, batch_size: int, seq_len: int, built_in: float, attr: str
) -> float:
    """Blend an optional :class:`OpCost` memory term (``activation_bytes`` / ``num_params``)."""
    mc = config.model_config
    rep = _fractioned([o for o in ops if o.slot in (SLOT_ATTENTION, SLOT_MLP)], mc)
    ext = _fractioned([o for o in ops if o.slot in (SLOT_ATTENTION_EXTRA, SLOT_LAYER_EXTRA)], mc)
    if not rep and not ext:
        return built_in
    ctx = OpContext(config, max(1, batch_size), max(1, seq_len))
    total = float(built_in)
    for op, f in rep:
        val: Optional[float] = getattr(op.cost(ctx), attr)
        if val is not None:
            total += f * (float(val) - float(built_in))
    for op, f in ext:
        val = getattr(op.cost(ctx), attr)
        if val is not None:
            total += f * float(val)
    return total
