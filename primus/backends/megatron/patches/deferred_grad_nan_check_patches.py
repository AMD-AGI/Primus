###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Check the gradient buckets for NaN / Inf once per step instead of once per bucket.

With ``check_for_nan_in_loss_and_grad`` (on by default), Megatron DDP's
``_ParamAndGradBucketGroup.check_grads`` runs when a bucket's gradients are ready, inside
the backward pass (from the AccumulateGrad hook). For every bucket it validates the
bucket's grad norm through the rerun state machine, which reads the norm on the host:
one ``hipStreamSynchronize`` per bucket. Each one stalls the CPU until the GPU queue
drains, leaving the GPU idle while the next kernels are launched. On the Muse Spark
1-GPU proxy that is 23 syncs and ~58 ms of host wait per step.

With ``deferred_grad_nan_check: true`` (the Primus default), ``check_grads`` keeps each bucket's non-finite
flag on the device, and ``DistributedDataParallel.finish_grad_sync`` (after the
backward pass, which waits for the gradients anyway) reads all of them with one host
sync and raises when any is set. ``deferred_grad_nan_check: false`` restores Megatron's
per-bucket check.

Differences from Megatron's path:

- A NaN / Inf raises immediately; the rerun state machine (re-running the step to tell a
  transient fault from a persistent one) is not used for these checks. So the patch only
  applies with ``rerun_mode: disabled`` (the default); with ``validate_results`` or
  ``report_stats`` it is skipped and Megatron's per-bucket check runs.
- ``check_for_large_grads`` keeps Megatron's path, since it needs the host-side history.
"""

import torch

from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

PATCH_NAME = "megatron.ddp.deferred_grad_nan_check"

# (bucket group, bucket index, 0-d bool device tensor: the bucket's grad norm is not finite)
_PENDING = []


def _check_pending():
    """Raise when any bucket recorded since the last call had a non-finite grad norm."""
    if not _PENDING:
        return
    pending = list(_PENDING)
    _PENDING.clear()
    flags = torch.stack([flag for _, _, flag in pending])
    if bool(flags.any()):  # the one host sync per step
        bad = [i for (_, i, _), f in zip(pending, flags.tolist()) if f]
        raise RuntimeError(
            f"found NaN or Inf in the local grad norm of bucket(s) {bad} in the backward pass, "
            "before the data-parallel collective (deferred_grad_nan_check)"
        )


@register_patch(
    PATCH_NAME,
    backend="megatron",
    phase="before_train",
    condition=lambda ctx: bool(getattr(get_args(ctx), "deferred_grad_nan_check", False)),
    description="Check DDP gradient buckets for NaN / Inf once per step (no per-bucket host sync)",
)
def patch_deferred_grad_nan_check(ctx: PatchContext) -> None:
    rerun_mode = getattr(get_args(ctx), "rerun_mode", "disabled")
    if rerun_mode != "disabled":
        log_rank_0(
            f"[Patch:{PATCH_NAME}] skipped: rerun_mode is '{rerun_mode}', which needs Megatron's "
            "per-bucket grad NaN / Inf check"
        )
        return

    from megatron.core.distributed import distributed_data_parallel as ddp_module
    from megatron.core.distributed import param_and_grad_buffer as pgb

    group_cls = pgb._ParamAndGradBucketGroup
    ddp_cls = ddp_module.DistributedDataParallel
    if is_patched(group_cls, PATCH_NAME):
        return

    orig_check_grads = group_cls.check_grads
    orig_finish_grad_sync = ddp_cls.finish_grad_sync

    def check_grads(self, check_for_nan_or_inf, check_for_large):
        if check_for_large:
            return orig_check_grads(self, check_for_nan_or_inf, check_for_large)
        if check_for_nan_or_inf:
            for i, bucket in enumerate(self.buckets):
                _PENDING.append((self, i, torch.isfinite(bucket.grad_data.norm(p=2)).logical_not()))

    def finish_grad_sync(self, *args, **kwargs):
        out = orig_finish_grad_sync(self, *args, **kwargs)
        _check_pending()
        return out

    group_cls.check_grads = check_grads
    ddp_cls.finish_grad_sync = finish_grad_sync
    mark_patched(group_cls, PATCH_NAME)
    log_rank_0(f"[Patch:{PATCH_NAME}] DDP grad NaN / Inf check deferred to one host sync per step")
