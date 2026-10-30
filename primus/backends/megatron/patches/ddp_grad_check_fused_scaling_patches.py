###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""One pass over each gradient bucket for the NaN / Inf check and the data-parallel scaling.

With ``check_for_nan_in_grad`` (on by default, from ``check_for_nan_in_loss_and_grad``),
``_ParamAndGradBucketGroup.start_grad_sync`` makes two passes over every bucket before its
reduce-scatter: ``check_grads`` takes ``grad_data.norm(p=2)`` and validates it twice, ``isnan`` and
``isinf``, each a host sync; then ``grad_data *= gradient_scaling_factor``. On the 4-layer
DeepSeek-V3 proxy (EP4) a rank holds ~16 GB of bf16 gradients in 22 buckets: the norms take
2.6 ms and the scaling 3.9 ms of a ~950 ms step, and the 44 syncs leave the GPU idle between them.

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

import torch
import triton
import triton.language as tl

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0

_BLOCK = 16384
_TRUE_FACTOR = "_primus_true_gradient_scaling_factor"


@triton.jit
def _scale_sumsq_kernel(X, PARTIAL, n, factor, BLOCK: tl.constexpr, SCALE: tl.constexpr):
    """PARTIAL[pid] = sum of X[block]^2 in fp32, and X[block] *= factor (fp32 product, one rounding)."""
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(PARTIAL + pid, tl.sum(x * x, axis=0))
    if SCALE:
        tl.store(X + offs, (x * factor).to(X.dtype.element_ty), mask=mask)


def scale_and_norm(x: torch.Tensor, factor: float) -> torch.Tensor:
    """``x *= factor`` in place; returns the L2 norm of ``x`` before the scaling, in ``x.dtype``."""
    n = x.numel()
    if n == 0:
        return torch.zeros((), dtype=x.dtype, device=x.device)
    grid = triton.cdiv(n, _BLOCK)
    partial = torch.empty(grid, dtype=torch.float32, device=x.device)
    _scale_sumsq_kernel[(grid,)](x, partial, n, float(factor), BLOCK=_BLOCK, SCALE=factor != 1.0, num_warps=8)
    return partial.sum().sqrt().to(x.dtype)


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
    "megatron.ddp.fused_grad_check_scaling",
    backend="megatron",
    phase="before_train",
    description="One pass per gradient bucket for the NaN/Inf check and the DP scaling, one sync per group",
    condition=lambda ctx: _upstream_layout(),
)
def patch_fused_grad_check_scaling(ctx: PatchContext) -> None:
    from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBucketGroup
    from megatron.core.rerun_state_machine import get_rerun_state_machine

    original = _ParamAndGradBucketGroup.check_grads
    if getattr(original, "_primus_fused_scaling", False):
        return

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
            if bool(torch.stack([norm.float() for norm in norms]).isfinite().all()):
                return
        # upstream's checks and messages, in its order
        rerun_state_machine = get_rerun_state_machine()
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

    check_grads._primus_fused_scaling = True
    _ParamAndGradBucketGroup.check_grads = check_grads
    log_rank_0(
        "[Patch:megatron.ddp.fused_grad_check_scaling] check_grads scales each bucket in the pass "
        "that takes its norm, one host sync per bucket group"
    )
