###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Zero only the part of the DDP gradient buffer that no Turbo wgrad overwrites.

Megatron zeroes the whole ``_ParamAndGradBuffer.grad_data`` every iteration so
that the beta=1 gradient accumulation each producer performs starts from zero.
Turbo's MXFP4 expert wgrad no longer accumulates on the step's first write: it
runs its epilogue at beta=0 and replaces the whole ``main_grad`` view. Zeroing
those slices first is then pure bandwidth. On GPT-OSS 20B the expert weights are
93% of a 20,488,420,224-element buffer, so almost all of the clear is dead work.

The patch wraps two Megatron methods and modifies neither:

``DistributedDataParallel.zero_grad_buffer``
    Rotates :mod:`primus_turbo.pytorch.core.grad_ownership`'s log once per
    iteration, which also re-checks that everything skipped last iteration was
    in fact overwritten during it.

``_ParamAndGradBuffer.reset``
    Zeroes the complement of the owned slices instead of the whole buffer.

What makes the skip safe:

- A slice is owned only if the producer logged a beta=0 write to it during the
  *previous* iteration, and the log is rebuilt from scratch every iteration. The
  producer selects and logs the beta=0 writer in actual backward order, so
  activation recompute and staged forwards cannot leave a forward-time claim.
- An empty log means "zero everything". Iteration 0 has nothing logged, as do
  configurations where no overwrite-capable producer runs.
- A claim is honoured only when the parameter's ``main_grad`` is exactly the
  contiguous ``[start, end)`` the buffer's own index map assigns it, matched by
  address, element count, and dtype. Inter-parameter alignment padding and the padding
  at the end of each bucket therefore always get zeroed: the reduce-scatter
  reads them.
- ``PRIMUS_TURBO_GRAD_OWNERSHIP_POISON=1`` fills the skipped slices with NaN
  instead of leaving them, so anything that is not genuinely overwritten reaches
  the reduce-scatter as NaN rather than as a plausible stale gradient. A finite
  sentinel is not usable here: real wgrad output collides with one, while finite
  operands cannot produce NaN.

Set ``PRIMUS_TURBO_GRAD_OWNERSHIP=1`` to opt in. It defaults off, so workloads
that have not explicitly validated the producer/consumer contract continue to
zero the whole buffer.
"""

from __future__ import annotations

import os
from typing import Dict, FrozenSet, List, Tuple

from primus.backends.megatron.patches.turbo.utils import is_primus_turbo_can_patch
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

Slice = Tuple[int, int, object]

_DISABLED = os.environ.get("PRIMUS_TURBO_GRAD_OWNERSHIP", "0") != "1"
_POISON = os.environ.get("PRIMUS_TURBO_GRAD_OWNERSHIP_POISON", "0") == "1"

# Rebuilt every iteration by the wrapped ``zero_grad_buffer``; consumed by the
# wrapped ``reset`` of every buffer belonging to that iteration.
_state: Dict[str, object] = {"owned": frozenset(), "seen": set(), "logged": False}


def _rotate_ownership(grad_ownership, *, discard: bool) -> FrozenSet[Slice]:
    """Start a producer epoch, optionally discarding the completed epoch.

    Pipeline warmup runs with ``args.curr_iteration == -1``. Its beta=0 writes
    exercise kernels but must never authorize skipped clearing in iteration 0.
    Rotating and discarding here clears the producer log without carrying those
    synthetic writes into real training.
    """
    owned = grad_ownership.begin_step()
    return frozenset() if discard else owned


def _is_enabled(ctx: PatchContext) -> bool:
    if _DISABLED:
        return False
    args = get_args(ctx)

    # The claim this patch trusts is only ever recorded by the fused wgrad path.
    if not bool(getattr(args, "gradient_accumulation_fusion", False)):
        return False
    return is_primus_turbo_can_patch(ctx)


def _owned_slices(buffer) -> List[Tuple[int, int, Slice]]:
    """Buffer offsets whose parameter was fully overwritten last iteration.

    Returns ``(start, end, key)`` triples, where ``key`` is the producer's
    ``(data_ptr, numel, dtype)`` log entry, so the caller can report back exactly what
    it skipped.
    """
    owned: FrozenSet[Slice] = _state["owned"]
    grad = getattr(buffer, "grad_data", None)
    index_map = getattr(buffer, "param_index_map", None)
    if not owned or grad is None or index_map is None or grad.numel() == 0:
        return []

    base = grad.data_ptr()
    itemsize = grad.element_size()
    found = []
    for param, entry in index_map.items():
        start, end, _bucket_id = entry
        main_grad = getattr(param, "main_grad", None)
        if main_grad is None or main_grad.dtype != grad.dtype:
            continue
        if not main_grad.is_contiguous() or main_grad.numel() != (end - start):
            continue
        if main_grad.data_ptr() != base + start * itemsize:
            continue
        key = (main_grad.data_ptr(), main_grad.numel(), main_grad.dtype)
        if key in owned:
            found.append((start, end, key))

    found.sort()
    for (_s0, end0, _k0), (start1, _e1, _k1) in zip(found, found[1:]):
        if start1 < end0:
            raise RuntimeError(f"overlapping owned gradient slices ending {end0} and starting {start1}")
    return found


def _reset_complement(buffer) -> bool:
    """Zero everything in ``buffer`` except the owned slices. False if it declined."""
    slices = _owned_slices(buffer)
    if not slices:
        return False

    grad = buffer.grad_data
    numel = grad.numel()
    cursor = 0
    for start, end, _key in slices:
        if start > cursor:
            grad[cursor:start].zero_()
        cursor = end
    if cursor < numel:
        grad[cursor:numel].zero_()

    if _POISON:
        for start, end, _key in slices:
            grad[start:end].fill_(float("nan"))

    from primus_turbo.pytorch.core import grad_ownership

    grad_ownership.note_skipped(key for _start, _end, key in slices)

    if not _state["logged"]:
        skipped = sum(end - start for start, end, _key in slices)
        log_rank_0(
            f"[Patch:megatron.turbo.grad_buffer_ownership] buffer numel={numel} owned slices="
            f"{len(slices)} skipped={skipped} ({100.0 * skipped / numel:.2f}%)"
            + ("  POISON=NaN" if _POISON else "")
        )
        _state["logged"] = True
    return True


@register_patch(
    "megatron.core.distributed.grad_buffer_ownership",
    backend="megatron",
    phase="before_train",
    description="Zero only the complement of the gradient-buffer slices that Primus-Turbo's beta=0 wgrad epilogue fully overwrites, instead of the whole buffer.",
    condition=_is_enabled,
)
def patch_grad_buffer_ownership(ctx: PatchContext) -> None:
    from megatron.core.distributed.distributed_data_parallel import (
        DistributedDataParallel,
    )
    from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBuffer
    from primus_turbo.pytorch.core import grad_ownership

    if getattr(_ParamAndGradBuffer.reset, "_primus_grad_ownership", False):
        return

    original_zero_grad_buffer = DistributedDataParallel.zero_grad_buffer
    original_reset = _ParamAndGradBuffer.reset
    args = get_args(ctx)

    def zero_grad_buffer(self):
        seen = _state["seen"]
        if id(self) in seen or not seen:
            _state["owned"] = _rotate_ownership(
                grad_ownership,
                discard=getattr(args, "curr_iteration", None) == -1,
            )
            seen.clear()
        seen.add(id(self))
        return original_zero_grad_buffer(self)

    def reset(self):
        if not _reset_complement(self):
            original_reset(self)

    setattr(reset, "_primus_grad_ownership", True)
    DistributedDataParallel.zero_grad_buffer = zero_grad_buffer
    _ParamAndGradBuffer.reset = reset

    log_rank_0(
        "[Patch:megatron.turbo.grad_buffer_ownership] Wrapped zero_grad_buffer() and "
        "_ParamAndGradBuffer.reset(); owned slices are reported on the first iteration "
        "that has any." + ("  POISON MODE: skipped slices filled with NaN." if _POISON else "")
    )
