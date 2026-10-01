###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
NeMo-aligned LR warmup patch.

NeMo's WarmupHoldPolicy._get_warmup_lr (nemo/core/optim/lr_scheduler.py)
computes warmup as:

    lr = base_lr * (step + 1) / (warmup_steps + 1)

Megatron's OptimizerParamScheduler.get_lr uses:

    lr = init_lr + (max_lr - init_lr) * num_steps / lr_warmup_steps

In Megatron's sample-space (num_steps increments by GBS per iteration,
lr_warmup_steps = warmup_iters * GBS), the NeMo-equivalent formula is:

    lr = init_lr + (max_lr - init_lr) * (num_steps + GBS) / (lr_warmup_steps + GBS)

This patch replaces the warmup branch of get_lr with the NeMo formula.
Enabled by setting nemo_aligned_lr_warmup: true in the YAML config.
"""

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0


def _nemo_lr_enabled(ctx: PatchContext) -> bool:
    args = get_args(ctx)
    return args is not None and getattr(args, "nemo_aligned_lr_warmup", False)


@register_patch(
    "megatron.lr_schedule.nemo_aligned",
    backend="megatron",
    phase="before_train",
    description="Align LR warmup with NeMo's (step+1)/(warmup_steps+1) formula",
    condition=_nemo_lr_enabled,
    priority=50,
)
def patch_nemo_aligned_lr_warmup(ctx: PatchContext):
    from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

    _original_get_lr = OptimizerParamScheduler.get_lr
    _gbs_cache = [None]

    def _nemo_get_lr(self, param_group):
        if self.lr_warmup_steps > 0 and self.num_steps <= self.lr_warmup_steps:
            if _gbs_cache[0] is None:
                from megatron.training import get_args as megatron_get_args

                _gbs_cache[0] = megatron_get_args().global_batch_size
            gbs = _gbs_cache[0]
            max_lr = param_group.get("max_lr", self.max_lr)
            return self.init_lr + (
                (max_lr - self.init_lr) * float(self.num_steps + gbs) / float(self.lr_warmup_steps + gbs)
            )
        return _original_get_lr(self, param_group)

    OptimizerParamScheduler.get_lr = _nemo_get_lr
    log_rank_0(
        "[Patch:nemo_aligned_lr] Patched get_lr warmup: "
        "(num_steps+GBS)/(lr_warmup_steps+GBS) = NeMo's (step+1)/(warmup+1)"
    )


# ---------------------------------------------------------------------------
# Reference (MLPerf Flux / torchtitan) LR warmup.
#
# torchtitan's linear_warmup_stable_decay, which the MLPerf Flux reference uses through
# LambdaLR, scales the peak LR by (step + 1) / warmup_steps for a 0-indexed step below
# warmup_steps, and by 1 after: the first optimizer step already runs at peak/W and step W-1
# reaches the peak. Megatron's stock warmup (num_steps / lr_warmup_steps) runs the first step at
# LR 0 and lags one step throughout; the NeMo form above is (step+1)/(W+1). Neither is the
# reference formula, so MLPerf closed-division runs set reference_lr_warmup: true.
#
# Megatron counts samples: num_steps = step * GBS before a step and lr_warmup_steps = W * GBS,
# so the reference factor is (num_steps + GBS) / lr_warmup_steps.
# ---------------------------------------------------------------------------


def _reference_lr_enabled(ctx: PatchContext) -> bool:
    args = get_args(ctx)
    return args is not None and getattr(args, "reference_lr_warmup", False)


@register_patch(
    "megatron.lr_schedule.reference_warmup",
    backend="megatron",
    phase="before_train",
    description="LR warmup = peak * (step+1) / warmup_steps, as the MLPerf Flux reference",
    condition=_reference_lr_enabled,
    priority=50,
)
def patch_reference_lr_warmup(ctx: PatchContext):
    from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

    if getattr(get_args(ctx), "nemo_aligned_lr_warmup", False):
        raise ValueError(
            "reference_lr_warmup and nemo_aligned_lr_warmup are mutually exclusive: they are two "
            "different warmup formulas. Set only one."
        )
    _original_get_lr = OptimizerParamScheduler.get_lr
    _gbs_cache = [None]

    def _reference_get_lr(self, param_group):
        if self.lr_warmup_steps > 0 and self.num_steps < self.lr_warmup_steps:
            if _gbs_cache[0] is None:
                from megatron.training import get_args as megatron_get_args

                _gbs_cache[0] = megatron_get_args().global_batch_size
            factor = min(1.0, float(self.num_steps + _gbs_cache[0]) / float(self.lr_warmup_steps))
            max_lr = param_group.get("max_lr", self.max_lr)
            return self.init_lr + (max_lr - self.init_lr) * factor
        return _original_get_lr(self, param_group)

    OptimizerParamScheduler.get_lr = _reference_get_lr
    log_rank_0(
        "[Patch:reference_lr_warmup] Patched get_lr warmup: "
        "(num_steps+GBS)/lr_warmup_steps = reference (step+1)/warmup_steps"
    )
