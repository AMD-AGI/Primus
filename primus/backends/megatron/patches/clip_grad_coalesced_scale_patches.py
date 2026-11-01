###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Global gradient clipping scales contiguous runs of the gradients, not one tensor at a time.

With Transformer Engine, Megatron keeps the clip coefficient on the device and scales the gradients
with ``multi_tensor_scale_tensor``, which walks them in 64K-element chunks, at most 320 a launch.
Under the precision-aware optimizer those gradients are ``decoupled_grad``: bf16 views of the DDP
gradient buffers, one per parameter shard. A rank's gradients then take many launches, each too small
to reach the memory bandwidth.

The views are adjacent slices of a few buffers. The replacement merges them into contiguous runs
(planned once per set of views, cached by their addresses and sizes; the cache holds no tensor) and
scales each run with one Triton pass that reads the coefficient from the device: the same fp32
product, rounded once, as the multi-tensor kernel, so the clipped gradients are bit for bit the same.
Gradients that overlap, are not contiguous or have nothing to merge (no two adjacent, e.g. separate
fp32 copies), a host (float) coefficient and a build without TE's tensor kernel keep the upstream
function.

When every parameter belongs to an optimizer that folds the coefficient into its Adam step
(``deferred_clip``, the Triton param-remainder Adam), the gradients are not scaled here at all: the
coefficient is left on the parameters and the step applies it as it reads each gradient, to the
same value.
"""

import collections
from functools import wraps

import torch

from primus.backends.megatron.core.optimizer import deferred_clip
from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched
from primus.backends.megatron.patches._rebind import rebind_everywhere
from primus.core.patches import PatchContext, register_patch
from primus.core.utils.gpu_arch import is_gfx1250
from primus.core.utils.module_utils import log_rank_0

_PATCH_KEY = "megatron.optimizer.clip_grad_coalesced_scale"
# run plans by the gradients' addresses and sizes: ints only, so the cache never keeps a gradient alive
_PLANS: "collections.OrderedDict" = collections.OrderedDict()
_PLANS_SIZE = 8


def _plan(grads):
    """``(index of its first gradient, length)`` of the fewest contiguous runs that cover exactly the
    elements of ``grads``; empty when two of them overlap, one is not contiguous, or no two are
    adjacent (nothing to merge)."""
    by_storage = collections.defaultdict(list)
    nonempty = 0
    for i, g in enumerate(grads):
        if g.numel() == 0:
            continue
        if not g.is_contiguous():
            return ()
        nonempty += 1
        by_storage[(g.untyped_storage().data_ptr(), g.dtype, g.device)].append(i)
    plan = []
    for indices in by_storage.values():
        indices.sort(key=lambda i: grads[i].storage_offset())
        first = indices[0]
        start = grads[first].storage_offset()
        end = start + grads[first].numel()
        for i in indices[1:]:
            offset = grads[i].storage_offset()
            if offset < end:
                return ()
            if offset > end:
                plan.append((first, end - start))
                first, start = i, offset
            end = offset + grads[i].numel()
        plan.append((first, end - start))
    return tuple(plan) if len(plan) < nonempty else ()


def _runs(grads, plan):
    """The runs of ``plan`` over ``grads``, or None when one does not fit in its first gradient's
    storage any more (the same addresses now hold other tensors)."""
    runs = []
    for i, n in plan:
        g = grads[i]
        if g.storage_offset() + n > g.untyped_storage().nbytes() // g.element_size():
            return None
        runs.append(g.as_strided((n,), (1,), g.storage_offset()))
    return runs


def contiguous_runs(grads):
    """``grads`` as the fewest contiguous 1-D tensors that cover exactly their elements, or None when
    two of them overlap, one is not contiguous, or no two are adjacent (nothing to merge)."""
    key = tuple((g.data_ptr(), g.numel(), g.dtype, g.device) for g in grads)
    plan = _PLANS.get(key)
    if plan is not None:
        _PLANS.move_to_end(key)
        if not plan:
            return None
        runs = _runs(grads, plan)
        if runs is not None:
            return runs
    plan = _PLANS[key] = _plan(grads)
    if len(_PLANS) > _PLANS_SIZE:
        _PLANS.popitem(last=False)
    if not plan:
        return None
    log_rank_0(f"[Patch:{_PATCH_KEY}] {len(grads)} gradients in {len(plan)} contiguous runs")
    return _runs(grads, plan)


def _make_clip(original):
    from megatron.core.optimizer import clip_grads
    from megatron.core.utils import to_local_if_dtensor

    # loaded only once the patch applies: every Megatron job imports this module, on any GPU
    from primus.backends.megatron.core.extensions._triton.grad_scale import (
        scale_by_device_scalar,
    )

    @wraps(original)
    def clip_grad_by_total_norm_fp32(parameters, max_norm, total_norm, use_decoupled_grad=False):
        if not isinstance(total_norm, torch.Tensor) or clip_grads.multi_tensor_scale_tensor_impl is None:
            return original(parameters, max_norm, total_norm, use_decoupled_grad)
        # upstream's gradient selection
        grads, with_grad = [], []
        for param in parameters:
            if use_decoupled_grad:
                if hasattr(param, "decoupled_grad") and param.decoupled_grad is not None:
                    assert param.decoupled_grad.dtype in [torch.float32, torch.bfloat16]
                    grads.append(to_local_if_dtensor(param.decoupled_grad).detach())
                    with_grad.append(param)
            else:
                if param.grad is not None:
                    assert param.grad.type() == "torch.cuda.FloatTensor"
                    grads.append(to_local_if_dtensor(param.grad).detach())
                    with_grad.append(param)
        clip_coeff = max_norm / (total_norm + 1.0e-6)
        clip_coeff.clamp_max_(1.0)
        clip_coeff = clip_coeff.to(torch.float32).reshape(-1)[:1]
        if deferred_clip.defer(with_grad, clip_coeff):
            return
        runs = contiguous_runs(grads)
        if runs is None or any(r.device != total_norm.device for r in runs):
            return original(parameters, max_norm, total_norm, use_decoupled_grad)
        for run in runs:
            scale_by_device_scalar(run, clip_coeff)

    return clip_grad_by_total_norm_fp32


@register_patch(
    _PATCH_KEY,
    backend="megatron",
    phase="before_train",
    description="Clip gradients by scaling contiguous runs of them, one Triton pass each (gfx1250)",
    # measured and validated on gfx1250 only; other architectures keep the upstream clip
    condition=lambda ctx: is_gfx1250(),
)
def patch_clip_grad_coalesced_scale(ctx: PatchContext) -> None:
    from megatron.core.optimizer import clip_grads

    if is_patched(clip_grads, _PATCH_KEY):
        return
    # the optimizers import the function by name: rebind it in every module that holds it
    patched = rebind_everywhere(
        clip_grads, "clip_grad_by_total_norm_fp32", _make_clip(clip_grads.clip_grad_by_total_norm_fp32)
    )
    mark_patched(clip_grads, _PATCH_KEY)
    log_rank_0(
        f"[Patch:{_PATCH_KEY}] clip_grad_by_total_norm_fp32 scales contiguous runs of the gradients "
        f"({', '.join(patched)})"
    )
