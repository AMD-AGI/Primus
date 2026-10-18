###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Fused clip + AdamW + bf16 write optimizer step, and a global-norm grad NaN check
================================================================================

Two opt-in patches on Megatron's DistributedOptimizer path, both implemented in
``primus.backends.megatron.core.optimizer.fused_adamw_clip`` (one shared set of
hooks on ``clip_grad_by_total_norm_fp32``,
``MixedPrecisionOptimizer.step_with_ready_grads``,
``FP32Optimizer.step_with_ready_grads`` and
``_ParamAndGradBucketGroup.check_grads``; the hooks are inert unless a switch is on):

``use_primus_fused_adamw_clip`` (patch ``megatron.optimizer.fused_adamw_clip``)
    One Triton pass per main param (pinned BLOCK 2048, num_warps 8, no autotune)
    does the grad clip (coefficient in registers), the AdamW update of the fp32
    main param / exp_avg / exp_avg_sq and the bf16 model-param write. Bitwise equal
    to TE's FusedAdam + the original clip + the main -> model copy; replaces three
    passes over HBM (TE multi_tensor_scale, TE multi_tensor_adam, per-param copy_).
    Needs the distributed optimizer with bf16/fp16 params, fp32 main params and
    fp32 moments (no precision-aware optimizer, no FSDP): otherwise the patch is
    not enabled (warning). Any other ineligible step runs the original step
    (logged once).

``use_primus_global_grad_nan_check`` (patch ``megatron.ddp.global_grad_nan_check``)
    Replaces DDP's per-bucket grad NaN/Inf check (a norm kernel and two host syncs
    per bucket inside the last micro-batch's backward) by a check of the global
    grad norm before any parameter update (fatal, like the bucket check). Needs
    ``check_for_nan_in_loss_and_grad`` (otherwise there is nothing to replace),
    ``clip_grad > 0`` (the check sits in the clip call), ``rerun_mode: disabled``
    (the bucket check feeds Megatron's rerun state machine otherwise) and no
    ``sft_sanitize_nan_grads`` (which zeroes bad grads inside the bucket check);
    otherwise it is not enabled.

Both are mutually exclusive with ``PRIMUS_FUSED_ADAM_CLIP=1`` (patch
``megatron.optimizer.te_fused_adam_clip``), which moves the clip into TE's Adam
kernel and bypasses ``clip_grad_by_total_norm_fp32``: the clip would be dropped
and the global NaN check would never run. Setting both aborts the run
(``SystemExit``) at ``before_train``.
"""

from __future__ import annotations

import importlib
import os

from primus.backends.megatron.patches._gfx1250_notice import warn_if_not_gfx1250
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0, warning_rank_0

_FUSED_KEY = "megatron.optimizer.fused_adamw_clip"
_FUSED_ARG = "use_primus_fused_adamw_clip"
_NAN_KEY = "megatron.ddp.global_grad_nan_check"
_NAN_ARG = "use_primus_global_grad_nan_check"
# Switch of the patch "megatron.optimizer.te_fused_adam_clip" (te_patches/fused_adam_clip_patches.py).
_TE_FUSED_ADAM_CLIP_ENV = "PRIMUS_FUSED_ADAM_CLIP"


def _check_exclusive(key: str, arg: str) -> None:
    # A conflicting explicit configuration is a hard error. run_patches logs and swallows Exception,
    # so abort with SystemExit (a BaseException), which it does not catch.
    if os.environ.get(_TE_FUSED_ADAM_CLIP_ENV, "0") == "1":
        msg = (
            f"{arg} cannot be combined with {_TE_FUSED_ADAM_CLIP_ENV}=1: that patch bypasses "
            "clip_grad_by_total_norm_fp32, which this one relies on. Use one of the two."
        )
        log_rank_0(f"[Patch:{key}] FATAL: {msg}")
        raise SystemExit(f"[{key}] {msg}")


def _hooks(key: str) -> None:
    """Install the shared hooks (idempotent) and log what was hooked."""
    from primus.backends.megatron.core.optimizer import fused_adamw_clip as F

    res = F.install_hooks()
    log_rank_0(f"[Patch:{key}] hooks: {res}")
    if res.get("_ParamAndGradBucketGroup.check_grads") == "absent" and key == _NAN_KEY:
        warning_rank_0(f"[Patch:{key}] _ParamAndGradBucketGroup.check_grads not found: bucket checks stay on")


@register_patch(
    _FUSED_KEY,
    backend="megatron",
    phase="before_train",
    description="Fused grad clip + AdamW + bf16 param write (pinned Triton) for the DistributedOptimizer step",
    condition=lambda ctx: bool(getattr(get_args(ctx), _FUSED_ARG, False)),
    tags=["optimizer", "triton"],
)
def patch_fused_adamw_clip(ctx: PatchContext) -> None:
    warn_if_not_gfx1250(_FUSED_KEY, _FUSED_ARG)
    from primus.backends.megatron.core.optimizer import fused_adamw_clip as F

    _check_exclusive(_FUSED_KEY, _FUSED_ARG)
    args = get_args(ctx)
    uncovered = [
        what
        for what, hit in (
            ("use_distributed_optimizer is off", not getattr(args, "use_distributed_optimizer", False)),
            ("use_precision_aware_optimizer", getattr(args, "use_precision_aware_optimizer", False)),
            ("FSDP", getattr(args, "use_torch_fsdp2", False) or getattr(args, "use_megatron_fsdp", False)),
        )
        if hit
    ]
    if uncovered:
        warning_rank_0(
            f"[Patch:{_FUSED_KEY}] {', '.join(uncovered)}: the fused step does not cover this setup; not enabled"
        )
        return
    # Import the Triton kernel now: a missing or broken Triton fails at startup instead of at the first step.
    importlib.import_module("primus.backends.megatron.core.optimizer.triton_fused_adamw")
    _hooks(_FUSED_KEY)
    F.set_fused(True)
    log_rank_0(f"[Patch:{_FUSED_KEY}] fused clip + AdamW + bf16 write enabled")


@register_patch(
    _NAN_KEY,
    backend="megatron",
    phase="before_train",
    description="Replace DDP's per-bucket grad NaN/Inf check by a global grad-norm check before the update",
    condition=lambda ctx: bool(getattr(get_args(ctx), _NAN_ARG, False)),
    tags=["optimizer", "ddp", "sync"],
)
def patch_global_grad_nan_check(ctx: PatchContext) -> None:
    warn_if_not_gfx1250(_NAN_KEY, _NAN_ARG)
    from primus.backends.megatron.core.optimizer import fused_adamw_clip as F

    _check_exclusive(_NAN_KEY, _NAN_ARG)
    args = get_args(ctx)
    if not getattr(args, "check_for_nan_in_loss_and_grad", False):
        warning_rank_0(
            f"[Patch:{_NAN_KEY}] check_for_nan_in_loss_and_grad is off: nothing to replace, not enabled"
        )
        return
    rerun_mode = getattr(args, "rerun_mode", "disabled") or "disabled"
    if rerun_mode != "disabled":
        warning_rank_0(
            f"[Patch:{_NAN_KEY}] rerun_mode={rerun_mode}: the per-bucket check feeds the rerun state machine "
            "(fault attribution / determinism stats); per-bucket checks stay on"
        )
        return
    if getattr(args, "sft_sanitize_nan_grads", False):
        warning_rank_0(
            f"[Patch:{_NAN_KEY}] sft_sanitize_nan_grads is set: it zeroes NaN/Inf grads inside the bucket "
            "check, which must keep running; per-bucket checks stay on"
        )
        return
    clip = getattr(args, "clip_grad", 0.0) or 0.0
    if clip <= 0.0:
        warning_rank_0(
            f"[Patch:{_NAN_KEY}] clip_grad={clip}: the global-norm check runs inside the clip call, which is "
            "skipped without clipping; per-bucket checks stay on"
        )
        return
    _hooks(_NAN_KEY)
    F.set_nancheck(True)
    log_rank_0(f"[Patch:{_NAN_KEY}] per-bucket grad NaN/Inf checks replaced by the global grad-norm check")
