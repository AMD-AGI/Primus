###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Add the gradients of several micro-batches into ``main_grad`` in one pass.

Opt-in: ``grad_accum_group: K`` (0 or 1 = off, the default; capped at 8).

Megatron DDP's backward hook runs ``param.main_grad.add_(param.grad)`` for every parameter in
every micro-batch. With this patch the hook keeps ``param.grad`` of up to K-1 micro-batches
and, at the K-th, adds them all with one kernel (``core/fusions/grouped_grad_accum.py``): the
same additions, each rounded to the buffer's dtype, in the same order, so ``main_grad`` is
bit-identical. Whatever is still held is added before anything reads the buffer:
``start_grad_sync``, ``finish_grad_sync`` and ``scale_gradients`` flush first, and under
``overlap_grad_reduce`` the hook adds the held gradients together with the current one in the
last micro-batch (``is_last_microbatch``), before Megatron's hook registers the parameter for
its bucket's reduction.

What stays as it is: the gradient buffer, its zeroing, every addition and its rounding, the
reduction across ranks. The step's micro-batch count and the rank count do not matter: a
group is whatever arrived when the K-th gradient, the last micro-batch or a flush comes.

Cost, and the reason this is opt-in: up to K-1 extra gradient tensors per parameter (one BF16
copy of the rank's parameters per held micro-batch) stay alive across the following
micro-batches.

Only the addition is taken over, and only in the hook's plain case: a gradient in
``param.grad`` that is not already in ``main_grad`` (``grad_added_to_main_grad``, as under
``gradient_accumulation_fusion``: those parameters never reach this path), no CUDA graph
capture, a layout the kernel covers. Megatron's hook still runs for everything else it does
(its asserts, ``param.grad = None``, ``register_grad_ready``): it is called with
``grad_added_to_main_grad`` set for the duration of the call, which is Megatron's own way of
saying "do not add this one". Every other case flushes the parameter's held gradients and
runs Megatron's hook unchanged.

A held gradient is autograd's tensor, not a copy. A Function that returns a reused buffer as a
weight gradient would overwrite a held one, so the hook raises when a new gradient shares its
storage address with a held one.
"""

import torch

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0, warning_rank_0

_PATCH_KEY = "megatron.ddp.grouped_grad_accum"
_PENDING = "_primus_pending_grads"
_warned_drop = []


def group_size(args) -> int:
    try:
        k = int(getattr(args, "grad_accum_group", 0) or 0)
    except (TypeError, ValueError):
        return 0
    return k if k >= 2 else 0


def flush_pending(ddp) -> None:
    """Add every held gradient into its ``main_grad``, in micro-batch order."""
    pending = getattr(ddp, _PENDING, None)
    if not pending or not any(pending.values()):
        return
    from primus.backends.megatron.core.fusions.grouped_grad_accum import flush_held

    for param in pending:
        flush_held(param)


def drop_pending(ddp) -> None:
    """Forget held gradients (the buffer is about to be zeroed)."""
    pending = getattr(ddp, _PENDING, None)
    if not pending:
        return
    if any(pending.values()) and not _warned_drop:
        _warned_drop.append(True)
        warning_rank_0(
            f"[Patch:{_PATCH_KEY}] the gradient buffer is zeroed while gradients are still held "
            "(a step that never reduced them); dropping them, as the zeroing drops their sums"
        )
    for held in pending.values():
        held.clear()


def make_hook_factory(original_factory, group: int, is_graph_capturing):
    """``DistributedDataParallel._make_backward_post_hook`` whose hooks hold up to ``group - 1``
    gradients per parameter."""

    def _make_backward_post_hook(self, param: torch.nn.Parameter):
        original_hook = original_factory(self, param)

        def hook(*unused):
            pending = self.__dict__.setdefault(_PENDING, {})
            held = pending.get(param)
            grad = param.grad
            # Eligibility before any work: the case in which Megatron's hook adds param.grad.
            plain = (
                grad is not None
                and not is_graph_capturing()
                and param in self.param_to_bucket_group
                and not param.grad_added_to_main_grad
                and not getattr(param, "zero_out_wgrad", False)
            )
            if not plain and not held:
                return original_hook(*unused)
            # Imported here: only a run with the switch on needs Triton.
            from primus.backends.megatron.core.fusions.grouped_grad_accum import (
                HELD,
                accumulate_,
                can_fuse,
                flush_held,
            )

            if not plain or not can_fuse(param.main_grad, grad.data):
                # this addition goes Megatron's way: the held ones come first
                flush_held(param)
                return original_hook(*unused)
            if held is None:
                held = pending[param] = []
                setattr(param, HELD, held)
            new = grad.data
            if any(g.data_ptr() == new.data_ptr() for g in held):
                raise RuntimeError(
                    f"[Patch:{_PATCH_KEY}] a gradient of shape {tuple(new.shape)} arrived in the storage "
                    "of one still held from an earlier micro-batch: some autograd Function reuses its "
                    "weight-gradient buffer, which grad_accum_group cannot hold. Set grad_accum_group: 0."
                )
            held.append(new)
            # Under overlap_grad_reduce the bucket's reduction may start inside Megatron's hook
            # in the last micro-batch (register_grad_ready), so nothing may stay held past it.
            # Without it the reduction starts in start_grad_sync / finish_grad_sync, which flush.
            last = self.ddp_config.overlap_grad_reduce and self.param_to_bucket_group[param].is_last_microbatch
            if len(held) >= group or last:
                accumulate_(param.main_grad, held)
                held.clear()
            # The rest of Megatron's hook (asserts, param.grad = None, register_grad_ready), with
            # its addition switched off the way Megatron switches it off.
            param.grad_added_to_main_grad = True
            try:
                return original_hook(*unused)
            finally:
                param.grad_added_to_main_grad = False

        return hook

    return _make_backward_post_hook


def _flush_first(original):
    def method(self, *args, **kwargs):
        flush_pending(self)
        return original(self, *args, **kwargs)

    method.__name__ = getattr(original, "__name__", "method")
    method.__doc__ = getattr(original, "__doc__", None)
    return method


@register_patch(
    _PATCH_KEY,
    backend="megatron",
    phase="before_train",
    description="Accumulate K micro-batch gradients into main_grad in one pass (grad_accum_group)",
    condition=lambda ctx: group_size(get_args(ctx)) >= 2,
)
def patch_grouped_grad_accum(ctx: PatchContext) -> None:
    from megatron.core.distributed.distributed_data_parallel import DistributedDataParallel as DDP
    from megatron.core.transformer.cuda_graphs import is_graph_capturing

    from primus.backends.megatron.core.fusions.grouped_grad_accum import MAX_GRADS

    if getattr(DDP, "_primus_grouped_grad_accum", False):
        return
    group = min(group_size(get_args(ctx)), MAX_GRADS)
    original_zero = DDP.zero_grad_buffer

    def zero_grad_buffer(self, *args, **kwargs):
        drop_pending(self)
        return original_zero(self, *args, **kwargs)

    DDP._make_backward_post_hook = make_hook_factory(DDP._make_backward_post_hook, group, is_graph_capturing)
    DDP.start_grad_sync = _flush_first(DDP.start_grad_sync)
    DDP.finish_grad_sync = _flush_first(DDP.finish_grad_sync)
    DDP.scale_gradients = _flush_first(DDP.scale_gradients)
    DDP.zero_grad_buffer = zero_grad_buffer
    DDP._primus_grouped_grad_accum = True
    log_rank_0(
        f"[Patch:{_PATCH_KEY}] main_grad accumulation grouped over {group} micro-batches "
        f"(holds up to {group - 1} extra gradients per parameter)"
    )
