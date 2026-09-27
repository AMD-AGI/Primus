###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Scale gradients for clipping with one flat ``mul_`` per contiguous run.

Megatron's ``clip_grad_by_total_norm_fp32`` scales gradients through
``multi_tensor_applier(multi_tensor_scale, ...)`` over a per-parameter tensor list.
That applier caps each launch at 320 blocks x 65536 elements, so the Flux-12B shard at
DP=8 costs 72 launches per step and runs at roughly half the HBM roof:

    multi_tensor_applier (shipped)   2231 us   2.45 TB/s
    torch._foreach_mul_              1846 us   2.97 TB/s
    flat buffer .mul_()               922 us   5.94 TB/s

(1369M bf16 elements, 250 tensors, 5.48 GB read+write, measured 2026-09-26. The 2231 us
reproduces the 2414 us seen in a production trace, so the model is sound.)

Traffic is identical in all three -- the difference is launch structure, which is why
``torch._foreach_mul_`` only buys 17% while the flat form buys 142%.

The flat form is valid here because the gradients are already contiguous slices of the
DDP bucket buffers. ``DistributedOptimizer`` builds them that way:

    megatron/core/optimizer/distrib_optimizer.py:2422
        shard_model_grad = model_grad.view(-1)[param_range.start : param_range.end]
        shard_main_param.decoupled_grad = shard_model_grad

so a run of consecutive shard grads is one contiguous range of one storage, and scaling
that range once is exactly equivalent to scaling each grad.

Correctness is structural rather than assumed: the patch groups the grads by storage,
sorts by ``storage_offset``, and only coalesces a run when each grad begins exactly where
the previous one ended and the dtypes match. Anything that does not form such a run --
a gap (a parameter in the buffer but not in the clip list), a non-contiguous grad, a
mixed dtype -- is left to the original code path. It therefore never scales a byte the
shipped path would not have scaled, and never scales one twice.

Off by default, like every other optimization gate in this tree. Enable with
``PRIMUS_FLAT_GRAD_CLIP=1``.
"""

import os

import torch

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0


def _coalesce(grads):
    """Group ``grads`` into (storage_tensor, offset, numel) runs, or None if not coalescible.

    Returns a list of runs covering every grad exactly once, or None when the set cannot
    be expressed as contiguous runs -- in which case the caller must fall back.
    """
    by_storage = {}
    for g in grads:
        if not g.is_contiguous() or g.numel() == 0:
            return None
        by_storage.setdefault(g.untyped_storage().data_ptr(), []).append(g)

    runs = []
    for tensors in by_storage.values():
        tensors.sort(key=lambda t: t.storage_offset())
        dtype = tensors[0].dtype
        start = tensors[0].storage_offset()
        end = start
        base = tensors[0]
        for t in tensors:
            if t.dtype is not dtype:
                return None
            off = t.storage_offset()
            if off == end:
                end += t.numel()
            elif off > end:
                # A gap: something in this storage is not being clipped. Close the run
                # and start a new one rather than scaling memory we were not asked to.
                runs.append((base, start, end - start))
                start, end = off, off + t.numel()
            else:
                # Overlap. Should not happen for shard views, but scaling twice would be
                # silently wrong, so refuse.
                return None
        runs.append((base, start, end - start))
    return runs


def _flat_scale(grads, coeff) -> bool:
    """Scale every grad by ``coeff`` using one ``mul_`` per contiguous run.

    Returns False without touching anything if the grads cannot be coalesced.
    """
    runs = _coalesce(grads)
    if runs is None:
        return False
    for base, offset, numel in runs:
        torch.as_strided(base, (numel,), (1,), offset).mul_(coeff)
    return True


@register_patch(
    "megatron.optimizer.flat_grad_clip",
    backend="megatron",
    phase="before_train",
    description=(
        "Clip gradients with one flat mul_ per contiguous run instead of the "
        "chunked multi_tensor applier."
    ),
    condition=lambda ctx: os.environ.get("PRIMUS_FLAT_GRAD_CLIP", "0") == "1",
)
def patch_flat_grad_clip(ctx: PatchContext) -> None:
    from megatron.core.optimizer import clip_grads as _clip_grads
    from megatron.core.optimizer import optimizer as _optimizer
    from megatron.core.utils import to_local_if_dtensor

    orig = _clip_grads.clip_grad_by_total_norm_fp32
    if getattr(orig, "_primus_flat_grad_clip", False):
        return

    def clip_grad_by_total_norm_fp32(
        parameters, max_norm, total_norm, use_decoupled_grad=False
    ):
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]

        grads = []
        for param in parameters:
            if use_decoupled_grad:
                g = getattr(param, "decoupled_grad", None)
                if g is not None:
                    assert g.dtype in (torch.float32, torch.bfloat16)
                    grads.append(to_local_if_dtensor(g).detach())
            elif param.grad is not None:
                assert param.grad.type() == "torch.cuda.FloatTensor"
                grads.append(to_local_if_dtensor(param.grad).detach())

        clip_coeff = max_norm / (total_norm + 1.0e-6)
        if clip_coeff >= 1.0 or not grads:
            return
        if not _flat_scale(grads, clip_coeff):
            # Not coalescible on this stack -- defer to the shipped implementation
            # rather than guessing.
            orig(parameters, max_norm, total_norm, use_decoupled_grad=use_decoupled_grad)

    clip_grad_by_total_norm_fp32._primus_flat_grad_clip = True
    # optimizer.py binds the name at import time (`from .clip_grads import ...`), so the
    # module attribute alone is not enough -- both references have to move.
    _clip_grads.clip_grad_by_total_norm_fp32 = clip_grad_by_total_norm_fp32
    _optimizer.clip_grad_by_total_norm_fp32 = clip_grad_by_total_norm_fp32
    log_rank_0("[Patch:megatron.optimizer.flat_grad_clip] flat per-run gradient clipping enabled")
