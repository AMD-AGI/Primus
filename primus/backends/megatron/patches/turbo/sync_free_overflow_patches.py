###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fail the step when the sync-free MoE permute budget overflowed.

With ``turbo_sync_free_moe_stage: 3`` and ``turbo_sync_free_permute_factor > 0`` the permuted-token
buffer holds ``tokens * topk * factor`` rows. A dispatch that routes more rows to a rank drops the
extra rows and counts them on the GPU. This patch reads that count once per training step (one
small D2H after ``train_step``, which already synchronizes for the loss) and raises if any permute
overflowed, so dropped tokens never go unnoticed.
"""

from primus.core.patches import PatchContext, get_args, register_patch


def _budget_enabled(ctx: PatchContext) -> bool:
    args = get_args(ctx)
    return bool(
        args is not None
        and getattr(args, "use_turbo_deepep", False)
        and getattr(args, "turbo_sync_free_moe_stage", 0) > 2
        and (getattr(args, "turbo_sync_free_permute_factor", 0.0) or 0.0) > 0
    )


@register_patch(
    "megatron.training.sync_free_permute_overflow_check",
    backend="megatron",
    phase="before_train",
    description="Raise when the sync-free MoE permute budget dropped tokens in a step",
    condition=_budget_enabled,
)
def patch_sync_free_overflow_check(ctx: PatchContext) -> None:
    import megatron.training.training as training

    from primus_turbo.pytorch.modules.moe.token_dispatcher import permute_overflow_count

    factor = get_args(ctx).turbo_sync_free_permute_factor
    original_train_step = training.train_step

    def _train_step_checked(*args, **kwargs):
        out = original_train_step(*args, **kwargs)
        dropped = permute_overflow_count(reset=True)
        if dropped:
            raise RuntimeError(
                f"sync-free MoE: {dropped} permute(s) in this step routed more tokens to a rank than "
                f"the budget tokens * topk * {factor}; tokens were dropped. Raise "
                "turbo_sync_free_permute_factor (or use turbo_sync_free_moe_stage <= 2)."
            )
        return out

    training.train_step = _train_step_checked
