###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Deferred, sync-free DDP gradient NaN/Inf check.

With ``check_for_nan_in_loss_and_grad`` (Megatron default) every bucket group's
``start_grad_sync`` calls ``check_grads``: a full ``grad_data.norm()`` per bucket followed
by two ``rerun_state_machine.validate_result`` calls whose ``bool(torch.isnan(norm))`` is a
host sync. Gradient-ready hooks fire inside the backward, so on the DeepSeek-V4 1-GPU
proxy that is 33 norm reductions (3.0 ms of kernels, 31% of the HBM floor) and 66 host
syncs per step in the middle of the backward, each draining the GPU queue.

This patch keeps the check but makes it sync-free while the backward runs: each group
folds its buckets' norms into one device accumulator (NaN/Inf propagate through the sum)
(a bandwidth-bound Triton sum of squares per bucket), and ``DistributedDataParallel.finish_grad_sync``
validates the accumulator once per step (one sync, same RuntimeError on NaN/Inf).
Only the per-bucket index in the error message is lost. ``check_for_large_grads`` keeps
the upstream per-bucket path.

Gate: ``PRIMUS_DEFERRED_GRAD_NAN_CHECK`` (default ``1``; ``0`` restores upstream).
"""

import os

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0


# g90mem: the per-step sum of squares of every grad bucket, kept so the optimizer can take the global grad
# norm from it instead of re-reading all gradients (PRIMUS_GRAD_NORM_FROM_NAN_CHECK=1, see
# te_patches/fused_adam_clip_patches.py). None when any bucket used the norm (not sum-of-squares) fallback.
_LAST_SUMSQ = {"value": None}


def pop_last_grad_sumsq():
    """Sum of squares of this step's grads (None if not available). A value produced while a CUDA graph was
    being captured (full-iteration graph) is a static graph-pool tensor that every replay rewrites in place,
    and the Python code that produced it never runs again on replay, so it is kept instead of cleared."""
    v = _LAST_SUMSQ["value"]
    if not _LAST_SUMSQ.get("static", False):
        _LAST_SUMSQ["value"] = None
    return v


def _norm_reuse_requested() -> bool:
    return os.environ.get("PRIMUS_GRAD_NORM_FROM_NAN_CHECK", "0") == "1"


def _enabled(ctx: PatchContext) -> bool:
    if os.environ.get("PRIMUS_DEFERRED_GRAD_NAN_CHECK", "1") == "0":
        return False
    try:
        from megatron.core.distributed.distributed_data_parallel import (  # noqa: F401
            DistributedDataParallel,
        )
        from megatron.core.distributed.param_and_grad_buffer import (
            _ParamAndGradBucketGroup,
        )
    except ImportError:
        return False
    return hasattr(_ParamAndGradBucketGroup, "check_grads")


@register_patch(
    "megatron.ddp.deferred_grad_nan_check",
    backend="megatron",
    phase="before_train",
    description="Fold per-bucket grad NaN/Inf checks into one device accumulator, validated once per step",
    condition=_enabled,
)
def patch_deferred_grad_nan_check(ctx: PatchContext):
    import torch
    from megatron.core.distributed.distributed_data_parallel import (
        DistributedDataParallel,
    )
    from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBucketGroup
    from megatron.core.rerun_state_machine import get_rerun_state_machine

    if getattr(_ParamAndGradBucketGroup.check_grads, "_primus_deferred", False):
        return

    use_triton = os.environ.get("PRIMUS_DEFERRED_GRAD_NAN_TRITON", "1") != "0"
    if use_triton:
        try:
            from primus.backends.megatron.core.distributed.grad_sumsq import grad_sumsq
        except Exception:  # pragma: no cover - triton unavailable
            use_triton = False

    original_check = _ParamAndGradBucketGroup.check_grads
    original_finish = DistributedDataParallel.finish_grad_sync
    state = {"acc": None, "exact": True}

    def check_grads(self, check_for_nan_or_inf, check_for_large):
        if check_for_large or not check_for_nan_or_inf:
            return original_check(self, check_for_nan_or_inf, check_for_large)
        grads = [b.grad_data for b in self.buckets if b.grad_data is not None and b.grad_data.numel() > 0]
        if not grads:
            return
        # Sum of squares (NaN/Inf iff the norm is NaN/Inf). The Triton two-stage reduction
        # streams a 1 GB bf16 bucket ~2x faster than ATen's NormTwo reduce_kernel on gfx1250;
        # torch._foreach_norm is ~3x slower. Either way no host sync happens here.
        part = None
        for g in grads:
            if use_triton and g.is_cuda and g.is_contiguous():
                n = grad_sumsq(g)
            else:
                n = g.norm(p=2).float()
                state["exact"] = False
            part = n if part is None else part + n
        state["acc"] = part if state["acc"] is None else state["acc"] + part

    def _bucket_ranges(b, n):
        # wgsmall: params whose wgrad epilogue already produced the sum of squares of their final main_grad
        # (Turbo PRIMUS_TURBO_MOE_WGRAD_SUMSQ=1: ``_primus_main_grad_sumsq`` partial buffer, flagged fresh by this
        # step's backward) are cut out of the bucket; returns the element ranges still to reduce and those buffers.
        skip, folded = [], []
        for prm in getattr(b, "params_list", ()):
            buf = getattr(prm, "_primus_main_grad_sumsq", None)
            if buf is None or not getattr(prm, "_primus_main_grad_sumsq_fresh", False):
                continue
            idx = getattr(b, "param_to_index", {}).get(prm)
            if idx is None:
                continue
            prm._primus_main_grad_sumsq_fresh = False  # consumed; the next step's backward sets it again
            skip.append(idx)
            folded.append(buf)
        if not skip:
            return [(0, n)], folded
        ranges, cur = [], 0
        for lo, hi in sorted(skip):
            if lo > cur:
                ranges.append((cur, lo))
            cur = max(cur, hi)
        if cur < n:
            ranges.append((cur, n))
        return ranges, folded

    def _sumsq_all_buckets(ddp):
        # opt3-mem: with the NaN check off (e.g. --check_for_nan_in_loss_and_grad False under the
        # full-iteration graph) check_grads is never called; the clip norm still wants the sum of squares,
        # so take it here, once, over every bucket (the same reduction the l2norm would do).
        if not use_triton:
            return None
        acc = None
        folded = []
        for bg in list(getattr(ddp, "bucket_groups", [])) + list(getattr(ddp, "expert_parallel_bucket_groups", [])):
            for b in bg.buckets:
                g = b.grad_data
                if g is None or g.numel() == 0:
                    continue
                if not (g.is_cuda and g.is_contiguous()):
                    return None
                ranges, fb = _bucket_ranges(b, g.numel())
                folded += fb
                for lo, hi in ranges:
                    n = grad_sumsq(g[lo:hi])
                    acc = n if acc is None else acc + n
        if folded:
            n = torch.cat(folded).sum()
            acc = n if acc is None else acc + n
        return acc

    def finish_grad_sync(self, *args, **kwargs):
        # Run upstream first: without overlap_grad_reduce it is what calls start_grad_sync
        # (and so check_grads) for every group.
        out = original_finish(self, *args, **kwargs)
        acc, state["acc"] = state["acc"], None
        checked = acc is not None
        if acc is None and _norm_reuse_requested():
            acc = _sumsq_all_buckets(self)
            state["exact"] = acc is not None
        _LAST_SUMSQ["value"] = acc if state["exact"] else None
        _LAST_SUMSQ["static"] = bool(acc is not None and torch.cuda.is_current_stream_capturing())
        state["exact"] = True
        if checked:
            rsm = get_rerun_state_machine()
            msg = "in backward pass (deferred check over all grad buckets of this step)"
            rsm.validate_result(
                result=acc,
                rejection_func=torch.isnan,
                message=f"found NaN in local grad norm {msg}",
                tolerance=0.001,
                fatal=True,
            )
            rsm.validate_result(
                result=acc,
                rejection_func=torch.isinf,
                message=f"found Inf in local grad norm {msg}",
                tolerance=0.001,
                fatal=True,
            )
        return out

    check_grads._primus_deferred = True
    _ParamAndGradBucketGroup.check_grads = check_grads
    DistributedDataParallel.finish_grad_sync = finish_grad_sync
    log_rank_0("[Patch:megatron.ddp.deferred_grad_nan_check] per-bucket grad NaN/Inf checks deferred to finish_grad_sync")
