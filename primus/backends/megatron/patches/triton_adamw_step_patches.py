###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Opt-in patches for Megatron's optimizer step.

``use_primus_triton_adamw_step``      grad clip + AdamW + bf16 model-param write in one Triton pass per
                                     param (DistributedOptimizer with TE FusedAdam AdamW).
``use_primus_global_grad_nan_check``  check the global grad norm before the update instead of every DDP
                                     bucket.

Both install the shared hooks of ``primus.backends.megatron.core.optimizer.triton_adamw_step`` (see there
for eligibility, fallbacks and what the global check does not cover). Both rely on
``clip_grad_by_total_norm_fp32`` and therefore abort the run when ``PRIMUS_FUSED_ADAM_CLIP=1``, whose patch
bypasses it.
"""

import importlib
import os

from primus.backends.megatron.core.fusions._notices import warn_if_not_gfx1250
from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0, warning_rank_0

_STEP_KEY = "megatron.optimizer.triton_adamw_step"
_STEP_ARG = "use_primus_triton_adamw_step"
_NAN_KEY = "megatron.ddp.global_grad_nan_check"
_NAN_ARG = "use_primus_global_grad_nan_check"
_HOOKS_KEY = "megatron.optimizer.step_hooks"  # one set of hooks, shared by both patches
# Switch of the patch "megatron.optimizer.te_fused_adam_clip" (te_patches/fused_adam_clip_patches.py).
_TE_FUSED_ADAM_CLIP_ENV = "PRIMUS_FUSED_ADAM_CLIP"
_STEP_MODULE = "primus.backends.megatron.core.optimizer.triton_adamw_step"


def _check_exclusive(key: str, arg: str) -> None:
    # A conflicting explicit configuration is a hard error. run_patches logs and swallows Exception,
    # so abort with SystemExit (a BaseException), which it does not catch.
    if os.environ.get(_TE_FUSED_ADAM_CLIP_ENV, "0") == "1":
        msg = (
            f"{arg} cannot be combined with {_TE_FUSED_ADAM_CLIP_ENV}=1 (patch "
            "megatron.optimizer.te_fused_adam_clip): that patch bypasses clip_grad_by_total_norm_fp32, which "
            "this one relies on. Use one of the two."
        )
        log_rank_0(f"[Patch:{key}] FATAL: {msg}")
        raise SystemExit(f"[{key}] {msg}")


def _install_hooks() -> dict:
    """Hook the clip, both step_with_ready_grads and the DDP bucket check (idempotent; inert while both
    switches are off). Returns {hook: "patched" | "already" | "absent"}."""
    F = importlib.import_module(_STEP_MODULE)
    om = importlib.import_module("megatron.core.optimizer.optimizer")
    dm = importlib.import_module("megatron.core.optimizer.distrib_optimizer")
    pb = importlib.import_module("megatron.core.distributed.param_and_grad_buffer")
    targets = (
        ("clip_grad_by_total_norm_fp32", om, F.make_clip_grad_by_total_norm_fp32),
        (
            "MixedPrecisionOptimizer.step_with_ready_grads",
            om.MixedPrecisionOptimizer,
            lambda orig: F.make_step_with_ready_grads(orig, dm, _STEP_KEY),
        ),
        ("FP32Optimizer.step_with_ready_grads", om.FP32Optimizer, F.make_flush_then_step),
        (
            "_ParamAndGradBucketGroup.check_grads",
            getattr(pb, "_ParamAndGradBucketGroup", None),
            F.make_check_grads,
        ),
    )
    res = {}
    for name, owner, make in targets:
        attr, key = name.rsplit(".", 1)[-1], f"{_HOOKS_KEY}:{name}"
        if getattr(owner, attr, None) is None:
            res[name] = "absent"
        elif is_patched(owner, key):
            res[name] = "already"
        else:
            setattr(owner, attr, make(getattr(owner, attr)))
            mark_patched(owner, key)
            res[name] = "patched"
    return res


@register_patch(
    _STEP_KEY,
    backend="megatron",
    phase="before_train",
    description="Grad clip + AdamW + bf16 param write in one Triton pass per param (DistributedOptimizer)",
    condition=lambda ctx: bool(getattr(get_args(ctx), _STEP_ARG, False)),
    tags=["optimizer", "triton"],
)
def patch_triton_adamw_step(ctx: PatchContext) -> None:
    warn_if_not_gfx1250(_STEP_KEY, _STEP_ARG)
    _check_exclusive(_STEP_KEY, _STEP_ARG)
    args = get_args(ctx)
    uncovered = []
    if not getattr(args, "use_distributed_optimizer", False):
        uncovered.append("use_distributed_optimizer is off")
    if getattr(args, "use_precision_aware_optimizer", False):
        uncovered.append("use_precision_aware_optimizer")
    if getattr(args, "use_torch_fsdp2", False) or getattr(args, "use_megatron_fsdp", False):
        uncovered.append("FSDP")
    if uncovered:
        warning_rank_0(
            f"[Patch:{_STEP_KEY}] {', '.join(uncovered)}: the fused step does not cover this setup; "
            "not enabled"
        )
        return
    # Import the Triton kernel now: a missing or broken Triton fails at startup instead of at the first step.
    importlib.import_module("primus.backends.megatron.core.optimizer.triton_adamw_kernel")
    log_rank_0(f"[Patch:{_STEP_KEY}] hooks: {_install_hooks()}")
    importlib.import_module(_STEP_MODULE).set_fused(True)
    log_rank_0(f"[Patch:{_STEP_KEY}] fused clip + AdamW + bf16 write enabled")


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
    hooks = _install_hooks()
    log_rank_0(f"[Patch:{_NAN_KEY}] hooks: {hooks}")
    if hooks["_ParamAndGradBucketGroup.check_grads"] == "absent":
        warning_rank_0(
            f"[Patch:{_NAN_KEY}] _ParamAndGradBucketGroup.check_grads not found: bucket checks stay on"
        )
    importlib.import_module(_STEP_MODULE).set_nancheck(True)
    log_rank_0(f"[Patch:{_NAN_KEY}] per-bucket grad NaN/Inf checks replaced by the global grad-norm check")
