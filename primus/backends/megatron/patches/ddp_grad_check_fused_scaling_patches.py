###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""One pass over each gradient bucket for the NaN / Inf check and the data-parallel scaling.

With ``check_for_nan_in_grad`` (on by default, from ``check_for_nan_in_loss_and_grad``),
``_ParamAndGradBucketGroup.start_grad_sync`` makes two passes over every bucket before its
reduce-scatter: ``check_grads`` takes ``grad_data.norm(p=2)`` and validates it twice, ``isnan`` and
``isinf``, each a host sync; then ``grad_data *= gradient_scaling_factor``. Both passes read
every gradient, and each sync leaves the GPU idle until the host queues work again.

The replacement ``check_grads`` scales each bucket in place in one Triton pass that also returns
its sum of squares, then validates all of the group's norms with one sync, and sets the bucket's
``gradient_scaling_factor`` to 1 so that ``start_grad_sync``'s scaling loop, which runs right
after it, leaves the bucket alone. The true factor is kept on the bucket for the next step.

- The scaled gradients are bit for bit the two-pass ones: the same fp32 product, rounded once.
- A norm differs from ``Tensor.norm`` only in the order of its fp32 sum. It is validated by the
  same calls with the same messages, bucket by bucket, whenever any of the group's is not finite.
- ``check_for_large_grads`` (whose history compares norm values) and rerun modes other than
  ``disabled`` keep the upstream ``check_grads``, as does a bucket not contiguous on the GPU.

Megatron's ``start_grad_sync`` and Primus' RCCL-SDMA copy of it both call ``check_grads`` and then
run that scaling loop, and nothing else reads a bucket's ``gradient_scaling_factor``.
"""

import inspect
from functools import wraps

import torch

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.gpu_arch import is_gfx1250
from primus.core.utils.module_utils import log_rank_0

_PATCH_KEY = "megatron.ddp.fused_grad_check_scaling"
_TRUE_FACTOR = "_primus_true_gradient_scaling_factor"


def _fusable(group, check_for_nan_or_inf: bool, check_for_large: bool) -> bool:
    from megatron.core.rerun_state_machine import RerunMode, get_rerun_state_machine

    return (
        check_for_nan_or_inf
        and not check_for_large
        and get_rerun_state_machine().mode == RerunMode.DISABLED
        and all(
            b.grad_data.is_cuda
            and b.grad_data.is_contiguous()
            and b.grad_data.dtype in (torch.bfloat16, torch.float16, torch.float32)
            for b in group.buckets
        )
    )


def _upstream_layout() -> bool:
    """``check_grads`` takes the per-bucket norm and ``start_grad_sync`` scales right after it."""
    try:
        from megatron.core.distributed.param_and_grad_buffer import (
            _ParamAndGradBucketGroup,
        )

        check = inspect.getsource(_ParamAndGradBucketGroup.check_grads)
        start = inspect.getsource(_ParamAndGradBucketGroup.start_grad_sync)
    except (ImportError, AttributeError, OSError, TypeError):
        return False
    return (
        "grad_data.norm(p=2)" in check
        and "rejection_func=torch.isnan" in check
        and "rejection_func=torch.isinf" in check
        and "self.check_grads(" in start
        and "bucket.grad_data *= bucket.gradient_scaling_factor" in start
        and start.index("self.check_grads(")
        < start.index("bucket.grad_data *= bucket.gradient_scaling_factor")
    )


@register_patch(
    _PATCH_KEY,
    backend="megatron",
    phase="before_train",
    description="One pass per gradient bucket for the NaN/Inf check and the DP scaling, one sync per group",
    # measured and validated on gfx1250 only; other architectures keep the upstream check_grads
    condition=lambda ctx: is_gfx1250() and _upstream_layout(),
)
def patch_fused_grad_check_scaling(ctx: PatchContext) -> None:
    from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBucketGroup
    from megatron.core.rerun_state_machine import get_rerun_state_machine

    # loaded only once the patch applies: every Megatron job imports this module, on any GPU
    from primus.backends.megatron.core.extensions._triton.grad_scale import (
        scale_and_norm,
    )
    from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched

    if is_patched(_ParamAndGradBucketGroup, _PATCH_KEY):
        return
    original = _ParamAndGradBucketGroup.check_grads

    @wraps(original)
    def check_grads(self, check_for_nan_or_inf, check_for_large):
        if not _fusable(self, check_for_nan_or_inf, check_for_large):
            # hand the scaling back to start_grad_sync
            for bucket in self.buckets:
                if hasattr(bucket, _TRUE_FACTOR):
                    bucket.gradient_scaling_factor = getattr(bucket, _TRUE_FACTOR)
            return original(self, check_for_nan_or_inf, check_for_large)
        norms = []
        with torch.no_grad():
            for bucket in self.buckets:
                factor = getattr(bucket, _TRUE_FACTOR, bucket.gradient_scaling_factor)
                setattr(bucket, _TRUE_FACTOR, factor)
                norms.append(scale_and_norm(bucket.grad_data, factor))
                bucket.gradient_scaling_factor = 1.0
        rerun_state_machine = get_rerun_state_machine()
        # With megatron.rerun.deferred_nan_check the calls below only queue their verdicts, checked
        # once per step; otherwise one sync here covers the group and the calls run on a failure.
        if not getattr(type(rerun_state_machine).validate_result, "_primus_deferred", False):
            if bool(torch.stack([norm.float() for norm in norms]).isfinite().all()):
                return
        # upstream's checks and messages, in its order
        for i, grad_norm in enumerate(norms):
            rerun_state_machine.validate_result(
                result=grad_norm,
                rejection_func=torch.isnan,
                message=f"found NaN in local grad norm for bucket #{i} "
                f"in backward pass before data-parallel communication collective",
                tolerance=0.001,  # 0.1% tolerance to account for non-deterministic FA backward
                fatal=True,
            )
            rerun_state_machine.validate_result(
                result=grad_norm,
                rejection_func=torch.isinf,
                message=f"found Inf in local grad norm for bucket #{i} "
                f"in backward pass before data-parallel communication collective",
                tolerance=0.001,  # 0.1% tolerance to account for non-deterministic FA backward
                fatal=True,
            )

    _ParamAndGradBucketGroup.check_grads = check_grads
    mark_patched(_ParamAndGradBucketGroup, _PATCH_KEY)
    log_rank_0(
        f"[Patch:{_PATCH_KEY}] check_grads scales each bucket in the pass "
        "that takes its norm, at most one host sync per bucket group"
    )
