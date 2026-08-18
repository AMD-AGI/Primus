###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""In-place MXFP4 -> BF16 precision switch at a chosen iteration.

The fastest measured route to the Flux 12B convergence gate on one MI355X node
trains MXFP4 for 12,288 iterations and then heals in BF16, reaching val loss
0.586 at iteration ~16,000 for 4.75 h end to end, against 6.10 h for BF16
throughout. That was measured as two processes: save a checkpoint at 12,288,
then resume under a BF16 recipe. The arithmetic is right but the run is not
submittable, because a scoring run wants one contiguous wall clock, one MLLOG
stream and an unbroken data order, and a resume breaks all three.

This module performs the same transition inside a single launch. It is small
because of a property of the Flux MXFP4 path specifically: ``MXFP4LinearFunction``
holds BF16 master weights and quantizes inside forward, so switching precision
means "stop quantizing" and nothing about the parameters, the optimizer state or
the gradient buffers changes. Contrast
``megatron_bridge.recipes.mlperf_llama2_70b.mxfp4_healing``, which has to restage
quantized weights from a CPU stash and reset TE recipe state, because on that
path the quantized weight *is* the parameter.

What the switch does not do, deliberately:

* It does not touch the optimizer, the LR schedule or the data order. The BF16
  phase continues the same run; only the GEMM precision of the MXFP4 linears
  changes.
* It does not convert or re-initialize weights. There is nothing to convert.
* It does not disable the sensitive-layer overrides. Layers built in BF16 stay
  BF16, so switching is idempotent with respect to them.

Trigger arithmetic: Megatron's train loop is ``while iteration < train_iters``
and passes that ``iteration`` into ``train_step`` *before* incrementing it, so
``iteration`` equals the number of completed steps. Firing when
``iteration >= mxfp4_switch_iter`` therefore makes iterations 1..N run MXFP4 and
N+1 onward run BF16 -- byte-identical in scheduling to resuming from a checkpoint
written at step N, which is what makes the inline result comparable to the 4.75 h
measurement.

One cost worth stating: the model is compiled, and ``_mxfp4_enabled`` is a plain
Python bool read inside the compiled region, so Dynamo guards on it and the first
step after the switch recompiles. That is a one-time cost at the switch step and
is the reason the flag is a bool attribute rather than a tensor: a tensor would
avoid the recompile but would keep the quantization branch live in the graph,
which is the opposite of what the switch is for.
"""

from __future__ import annotations

from typing import Any, Iterable

from primus.core.utils.module_utils import log_rank_0

_SWITCH_APPLIED: bool = False


def _iter_modules(model: Any) -> Iterable[Any]:
    chunks = list(model) if isinstance(model, (list, tuple)) else [model]
    for chunk in chunks:
        yield from chunk.modules()


def _switchable(model: Any) -> list:
    """Every MXFP4 linear still quantizing.

    Detected by attribute rather than by class so that the two module classes
    (column- and row-parallel) need no registry here, and so a module that has
    already been switched is not counted twice.
    """
    return [m for m in _iter_modules(model) if getattr(m, "_mxfp4_enabled", False)]


def switch_iter(config: Any) -> int:
    return int(getattr(config, "mxfp4_switch_iter", 0) or 0)


def is_switched() -> bool:
    """True once the BF16 phase has begun in this process."""
    return _SWITCH_APPLIED


def reset_switch_state() -> None:
    """Test and multi-run hygiene: forget that a switch happened."""
    global _SWITCH_APPLIED
    _SWITCH_APPLIED = False


def apply_switch(model: Any) -> int:
    """Stop MXFP4 quantization on every switchable linear. Returns how many.

    Idempotent: a second call finds nothing switchable and returns 0.
    """
    global _SWITCH_APPLIED
    targets = _switchable(model)
    for module in targets:
        module._mxfp4_enabled = False
    _SWITCH_APPLIED = True
    return len(targets)


def apply_switch_if_due(model: Any, config: Any, iteration: int) -> int:
    """Switch when ``iteration`` reaches the configured step. Returns how many.

    ``iteration`` is Megatron's completed-step count as passed to ``train_step``.
    The ``>=`` rather than ``==`` matters: a resumed run can re-enter the loop at
    an iteration past the trigger, and a switch that only fired on exact equality
    would silently leave such a run in MXFP4 for its whole BF16 phase.
    """
    if _SWITCH_APPLIED:
        return 0

    target = switch_iter(config)
    if target <= 0 or iteration < target:
        return 0

    n = apply_switch(model)
    if n == 0:
        log_rank_0(
            f"[mxfp4_inline_switch] iteration={iteration} reached "
            f"mxfp4_switch_iter={target} but no MXFP4 linear was found to switch; "
            "the model was not built with the MXFP4 local spec."
        )
        return 0

    log_rank_0(
        f"[mxfp4_inline_switch] switched {n} MXFP4 linears to BF16 at "
        f"iteration={iteration} (mxfp4_switch_iter={target}); master weights, "
        "optimizer state, LR schedule and data order are untouched. Expect one "
        "torch.compile recompilation on this step."
    )
    return n
