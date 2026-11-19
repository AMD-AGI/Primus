###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Deferred DDP NaN/Inf gradient check: one device reduction per bucket group, one host read per step.

Megatron's ``_ParamAndGradBucketGroup.check_grads`` (``check_for_nan_in_grad``) runs
``bucket.grad_data.norm(p=2)`` per bucket and then ``rerun_state_machine.validate_result``
twice per bucket (isnan, isinf). Each validation is a host ``bool(tensor)`` read, i.e. a
stream synchronization inside the backward hooks, and torch's bf16 ``NormTwoOps`` reduction
runs well below HBM bandwidth. On the 1-GPU DSv4 proxy this is 33 reductions (~3 ms) and
~66 synchronizations per step.

With ``PRIMUS_DEFERRED_GRAD_CHECK=1`` (opt-in) the check computes the norm of each bucket
group's local (pre-communication) gradients with TE ``multi_tensor_l2norm`` (one launch per
group, FP32 accumulation) and keeps it on the device. ``DistributedDataParallel.finish_grad_sync``
then reads all group norms with a single host synchronization. A non-finite group norm runs
Megatron's original per-bucket check (same messages, ``fatal=True``), so a NaN/Inf gradient
still aborts the job before the optimizer step. Rejections are therefore identical, except
that the check is evaluated at the end of the backward pass instead of at bucket dispatch.

Only the rerun-disabled mode without ``check_for_large_grads`` is deferred; every other mode
keeps the original code path.
"""

import os

import torch

from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched
from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0

_PATCH_KEY = "megatron.ddp.deferred_grad_check"
_MAX_PIECE = 1 << 30
_CHUNK = 2048 * 32


def _enabled() -> bool:
    return os.environ.get("PRIMUS_DEFERRED_GRAD_CHECK", "0") == "1"


@register_patch(
    _PATCH_KEY,
    backend="megatron",
    phase="before_train",
    description="Defer the DDP NaN/Inf grad check to one host read per step (TE l2norm per bucket group)",
    priority=60,
    condition=lambda ctx: _enabled(),
)
def patch_deferred_grad_check(ctx: PatchContext) -> None:
    del ctx
    from megatron.core.distributed.distributed_data_parallel import (
        DistributedDataParallel,
    )
    from megatron.core.distributed.param_and_grad_buffer import (
        _ParamAndGradBucketGroup,
    )
    from megatron.core.optimizer import clip_grads
    from megatron.core.rerun_state_machine import RerunMode, get_rerun_state_machine

    if is_patched(_ParamAndGradBucketGroup, _PATCH_KEY):
        return

    try:
        import transformer_engine.pytorch  # noqa: F401  (loads transformer_engine_torch)
        from transformer_engine_torch import multi_tensor_l2norm as raw_l2norm
    except ImportError:
        raw_l2norm = None

    original_check_grads = _ParamAndGradBucketGroup.check_grads
    original_finish_grad_sync = DistributedDataParallel.finish_grad_sync
    noop_flags = {}
    logged = []

    def deferred_check_grads(self, check_for_nan_or_inf, check_for_large):
        if (
            not check_for_nan_or_inf
            or check_for_large
            or get_rerun_state_machine().mode != RerunMode.DISABLED
        ):
            return original_check_grads(self, check_for_nan_or_inf, check_for_large)
        grads = [bucket.grad_data for bucket in self.buckets if bucket.grad_data.numel() > 0]
        if not grads:
            return None
        device = grads[0].device
        flag = noop_flags.get(device)
        if flag is None:
            flag = noop_flags[device] = torch.zeros(1, dtype=torch.int, device=device)
        if getattr(self, "_primus_deferred_grad_norm", None) is not None:
            # The previous norm was never consumed (finish_grad_sync not reached): check it now.
            if not bool(torch.isfinite(self._primus_deferred_grad_norm).all()):
                original_check_grads(self, True, False)
        # One TE l2norm launch for the whole group, FP32 accumulation, result stays on device.
        # The raw TE kernel is used because TE's ROCm Python wrapper sends a single-tensor list
        # to torch.linalg.vector_norm (1.7x slower here); views of <= 2^30 elements avoid the
        # ROCm TE out-of-bounds bug for >= 2^31-element tensors in a multi-tensor list.
        if raw_l2norm is not None:
            pieces = [p for g in grads for p in g.reshape(-1).split(_MAX_PIECE)]
            norm, _ = raw_l2norm(_CHUNK, flag, [pieces], False)
        else:
            norm, _ = clip_grads.multi_tensor_applier(clip_grads.l2_norm_impl, flag, [grads], False)
        self._primus_deferred_grad_norm = norm
        return None

    def finish_grad_sync_with_deferred_check(self, *args, **kwargs):
        result = original_finish_grad_sync(self, *args, **kwargs)
        groups = [
            group
            for group in self.bucket_groups + self.expert_parallel_bucket_groups
            if getattr(group, "_primus_deferred_grad_norm", None) is not None
        ]
        if groups:
            norms = torch.cat([group._primus_deferred_grad_norm.reshape(1) for group in groups])
            all_finite = bool(torch.isfinite(norms).all())  # the single host read per step
            if not all_finite:
                bad = (~torch.isfinite(norms)).nonzero().flatten().tolist()
                for index in bad:
                    # Megatron's per-bucket validation: identical messages, fatal on NaN/Inf.
                    # (A group norm can also overflow with every bucket finite; then this passes.)
                    original_check_grads(groups[index], True, False)
            for group in groups:
                group._primus_deferred_grad_norm = None
            if not logged:
                logged.append(True)
                log_rank_0(f"[Patch:{_PATCH_KEY}] first deferred check: {len(groups)} bucket groups")
        return result

    _ParamAndGradBucketGroup.check_grads = deferred_check_grads
    DistributedDataParallel.finish_grad_sync = finish_grad_sync_with_deferred_check
    mark_patched(_ParamAndGradBucketGroup, _PATCH_KEY)
    log_rank_0(
        f"[Patch:{_PATCH_KEY}] DDP NaN/Inf grad check: TE l2norm per bucket group, "
        "one host read per step in finish_grad_sync"
    )
