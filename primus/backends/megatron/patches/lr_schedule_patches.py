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
from primus.modules.module_utils import log_rank_0


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
