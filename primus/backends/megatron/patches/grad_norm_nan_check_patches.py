###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Per-step NaN / Inf check on the grad norm, without a sync.

Megatron's ``check_for_nan_in_loss_and_grad`` checks every microbatch's loss and every gradient
bucket with ``.item()``: about 22 host syncs per step on the DeepSeek-V3 proxy (8 microbatches),
each of which drains the GPU queue (about 22 ms of GPU idle per step, EP4 on 4x MI455X).

With ``check_for_nan_in_grad_norm: true`` this patch raises after any step whose grad norm is
NaN or Inf. A NaN / Inf loss on any rank gives a NaN / Inf gradient and so a NaN / Inf norm,
so with ``check_for_nan_in_loss_and_grad: false`` the run still stops on the step that went bad;
it only stops after that step's optimizer update instead of before it, and reports the norm
rather than the rank. The norm is already a host float (the optimizer synchronizes for gradient
clipping), so the check costs nothing.
"""

import math

from primus.core.patches import PatchContext, get_args, register_patch


def _enabled(ctx: PatchContext) -> bool:
    args = get_args(ctx)
    return bool(args is not None and getattr(args, "check_for_nan_in_grad_norm", False))


@register_patch(
    "megatron.training.grad_norm_nan_check",
    backend="megatron",
    phase="before_train",
    description="Raise when a step's grad norm is NaN / Inf (no extra host sync)",
    condition=_enabled,
)
def patch_grad_norm_nan_check(ctx: PatchContext) -> None:
    import megatron.training.training as training

    original_train_step = training.train_step

    def _train_step_checked(*args, **kwargs):
        out = original_train_step(*args, **kwargs)
        grad_norm = out[5]
        if grad_norm is not None and not math.isfinite(grad_norm):
            raise RuntimeError(f"grad norm is {grad_norm} after this step (NaN / Inf loss or gradient)")
        return out

    training.train_step = _train_step_checked
