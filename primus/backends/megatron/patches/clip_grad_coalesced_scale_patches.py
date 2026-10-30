###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Global gradient clipping scales contiguous runs of the gradients, not one tensor at a time.

With Transformer Engine, Megatron keeps the clip coefficient on the device and scales the gradients
with ``multi_tensor_scale_tensor``, which walks them in 64K-element chunks, at most 320 a launch.
Under the precision-aware optimizer those gradients are ``decoupled_grad``: bf16 views of the DDP
gradient buffers, one per parameter shard. On the 4-layer DeepSeek-V3 proxy (EP4) a rank's ~11 GB of
them take ~300 launches and 7.9 ms of a ~940 ms step, about 3 TB/s.

The views are adjacent slices of a few buffers. The replacement merges them into contiguous runs
(once per set of views, cached by their addresses and sizes) and scales each run with one Triton
pass that reads the coefficient from the device: the same fp32 product, rounded once, as the
multi-tensor kernel, so the clipped gradients are bit for bit the same. Gradients that overlap or
are not contiguous, a host (float) coefficient and a build without TE's tensor kernel keep the
upstream function.
"""

import collections

import torch
import triton
import triton.language as tl

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0

_BLOCK = 16384
_RUNS_CACHE: "collections.OrderedDict" = collections.OrderedDict()
_RUNS_CACHE_SIZE = 8


@triton.jit
def _scale_by_device_scalar_kernel(X, S, n, BLOCK: tl.constexpr):
    """X[block] *= S[0] (fp32 product, one rounding to X's dtype)."""
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    s = tl.load(S)
    x = tl.load(X + offs, mask=mask, other=0.0)
    tl.store(X + offs, (x.to(tl.float32) * s).to(X.dtype.element_ty), mask=mask)


def scale_by_device_scalar(x: torch.Tensor, scale: torch.Tensor) -> None:
    """``x *= scale`` in place for a contiguous ``x`` and a one-element fp32 ``scale`` on its device."""
    n = x.numel()
    if n:
        _scale_by_device_scalar_kernel[(triton.cdiv(n, _BLOCK),)](x, scale, n, BLOCK=_BLOCK, num_warps=8)


def contiguous_runs(grads):
    """``grads`` as the fewest contiguous 1-D tensors that cover exactly their elements, or None
    when two of them overlap or one is not contiguous."""
    key = tuple((g.data_ptr(), g.numel(), g.dtype, g.device) for g in grads)
    runs = _RUNS_CACHE.get(key)
    if runs is not None:
        _RUNS_CACHE.move_to_end(key)
        return runs
    by_storage = collections.defaultdict(list)
    for g in grads:
        if g.numel() == 0:
            continue
        if not g.is_contiguous():
            return None
        by_storage[(g.untyped_storage().data_ptr(), g.dtype, g.device)].append(g)
    runs = []
    for views in by_storage.values():
        views.sort(key=lambda t: t.storage_offset())
        base = views[0]
        start = views[0].storage_offset()
        end = start + views[0].numel()
        for t in views[1:]:
            offset = t.storage_offset()
            if offset < end:
                return None
            if offset > end:
                runs.append(base.as_strided((end - start,), (1,), start))
                start = offset
            end = offset + t.numel()
        runs.append(base.as_strided((end - start,), (1,), start))
    _RUNS_CACHE[key] = runs
    if len(_RUNS_CACHE) > _RUNS_CACHE_SIZE:
        _RUNS_CACHE.popitem(last=False)
    log_rank_0(
        f"[Patch:megatron.optimizer.clip_grad_coalesced_scale] {len(grads)} gradients in "
        f"{len(runs)} contiguous runs"
    )
    return runs


def _make_clip(original):
    from megatron.core.optimizer import clip_grads
    from megatron.core.utils import to_local_if_dtensor

    def clip_grad_by_total_norm_fp32(parameters, max_norm, total_norm, use_decoupled_grad=False):
        if not isinstance(total_norm, torch.Tensor) or clip_grads.multi_tensor_scale_tensor_impl is None:
            return original(parameters, max_norm, total_norm, use_decoupled_grad)
        # upstream's gradient selection
        grads = []
        for param in parameters:
            if use_decoupled_grad:
                if hasattr(param, "decoupled_grad") and param.decoupled_grad is not None:
                    assert param.decoupled_grad.dtype in [torch.float32, torch.bfloat16]
                    grads.append(to_local_if_dtensor(param.decoupled_grad).detach())
            else:
                if param.grad is not None:
                    assert param.grad.type() == "torch.cuda.FloatTensor"
                    grads.append(to_local_if_dtensor(param.grad).detach())
        runs = contiguous_runs(grads)
        if runs is None or any(r.device != total_norm.device for r in runs):
            return original(parameters, max_norm, total_norm, use_decoupled_grad)
        clip_coeff = max_norm / (total_norm + 1.0e-6)
        clip_coeff.clamp_max_(1.0)
        clip_coeff = clip_coeff.to(torch.float32).reshape(-1)[:1]
        for run in runs:
            scale_by_device_scalar(run, clip_coeff)

    clip_grad_by_total_norm_fp32._primus_coalesced = True
    return clip_grad_by_total_norm_fp32


@register_patch(
    "megatron.optimizer.clip_grad_coalesced_scale",
    backend="megatron",
    phase="before_train",
    description="Clip gradients by scaling contiguous runs of them, one Triton pass each",
)
def patch_clip_grad_coalesced_scale(ctx: PatchContext) -> None:
    import importlib

    from megatron.core.optimizer import clip_grads

    original = clip_grads.clip_grad_by_total_norm_fp32
    if getattr(original, "_primus_coalesced", False):
        return
    replacement = _make_clip(original)
    patched = []
    # the function is imported by name into these modules
    for name in (
        "megatron.core.optimizer.clip_grads",
        "megatron.core.optimizer.optimizer",
        "megatron.core.models.mimo.optimizer",
    ):
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        if getattr(module, "clip_grad_by_total_norm_fp32", None) is original:
            module.clip_grad_by_total_norm_fp32 = replacement
            patched.append(name)
    log_rank_0(
        "[Patch:megatron.optimizer.clip_grad_coalesced_scale] clip_grad_by_total_norm_fp32 scales "
        f"contiguous runs of the gradients ({', '.join(patched)})"
    )
