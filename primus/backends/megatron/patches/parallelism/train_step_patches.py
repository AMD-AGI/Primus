###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron train_step patches for ZeroBubble / Primus Pipeline compatibility.

When num_seq_splits > 1, the vanilla Megatron train_step passes
num_microbatches and seq_length without accounting for sequence splitting.
This patch wraps train_step to multiply num_microbatches by num_seq_splits
and divide seq_length accordingly, matching the legacy trainer behavior.
"""

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0


def _needs_seq_split_adjustment(ctx: PatchContext) -> bool:
    args = get_args(ctx)
    if args is None:
        return False
    return getattr(args, "num_seq_splits", 1) > 1 and (
        getattr(args, "patch_primus_pipeline", False) or getattr(args, "patch_zero_bubble", False)
    )


@register_patch(
    "megatron.training.train_step_seq_split",
    backend="megatron",
    phase="before_train",
    description="Patch Megatron train_step to adjust num_microbatches/seq_length for num_seq_splits > 1",
    condition=_needs_seq_split_adjustment,
    priority=40,
)
def patch_train_step_seq_split(ctx: PatchContext):
    import megatron.training.training as megatron_training

    _original_train_step = megatron_training.train_step

    def _patched_train_step(
        forward_step_func,
        data_iterator,
        model,
        optimizer,
        opt_param_scheduler,
        config,
        forward_backward_func,
    ):
        args = get_args(ctx)
        num_seq_splits = getattr(args, "num_seq_splits", 1)

        if num_seq_splits > 1:
            _orig_fwd_bwd = forward_backward_func

            def _adjusted_fwd_bwd(**kwargs):
                kwargs["num_microbatches"] = kwargs.get("num_microbatches", 1) * num_seq_splits
                if "seq_length" in kwargs and kwargs["seq_length"] is not None:
                    kwargs["seq_length"] = kwargs["seq_length"] // num_seq_splits
                return _orig_fwd_bwd(**kwargs)

            forward_backward_func = _adjusted_fwd_bwd

        return _original_train_step(
            forward_step_func,
            data_iterator,
            model,
            optimizer,
            opt_param_scheduler,
            config,
            forward_backward_func,
        )

    megatron_training.train_step = _patched_train_step
    log_rank_0(
        f"[Patch:megatron.training.train_step_seq_split]   "
        f"Patched train_step for num_seq_splits={getattr(get_args(ctx), 'num_seq_splits', 1)}"
    )
