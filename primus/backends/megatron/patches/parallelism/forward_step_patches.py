###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron forward_step patches for ZeroBubble compatibility.

When patch_zero_bubble is enabled, the ZB runtime passes DataLoaderStore
(a class, not an iterator) as the data_iterator argument to forward_step.
Vanilla pretrain_gpt.forward_step doesn't know about DataLoaderStore and
would try to iterate on it, causing a failure.

This patch replaces pretrain_gpt.forward_step with a version that detects
DataLoaderStore and uses its push/pop mechanism for batch retrieval.
"""

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0


def _needs_forward_step_patch(ctx: PatchContext) -> bool:
    args = get_args(ctx)
    return getattr(args, "patch_zero_bubble", False)


@register_patch(
    "megatron.pretrain_gpt.forward_step.zero_bubble",
    backend="megatron",
    phase="before_train",
    description="Patch pretrain_gpt.forward_step to support DataLoaderStore for ZeroBubble PP",
    condition=_needs_forward_step_patch,
    priority=40,
)
def patch_forward_step_zero_bubble(ctx: PatchContext):
    import pretrain_gpt

    _original_forward_step = pretrain_gpt.forward_step

    def _patched_forward_step(data_iterator, model, return_schedule_plan=False):
        from collections.abc import Iterable

        from megatron.core.utils import get_attr_wrapped_model
        from megatron.training import get_args as _get_args

        args = _get_args()

        if not args.patch_zero_bubble:
            return _original_forward_step(data_iterator, model, return_schedule_plan)

        from primus.modules.trainer.megatron.pre_trainer import DataLoaderStore

        timers = None
        try:
            from megatron.training import get_timers

            timers = get_timers()
        except Exception:
            pass

        if timers:
            timers("batch-generator", log_level=2).start()

        vp_stage = get_attr_wrapped_model(model, "vp_stage")

        if data_iterator is DataLoaderStore or (
            not isinstance(data_iterator, Iterable) and data_iterator is not None
        ):
            tokens, labels, loss_mask, attention_mask, position_ids = data_iterator.pop()
        else:
            DataLoaderStore.push(data_iterator, h2d_stream=False, vp_stage=vp_stage)
            tokens, labels, loss_mask, attention_mask, position_ids = DataLoaderStore.pop()

        if timers:
            timers("batch-generator").stop()

        from functools import partial

        if args.use_legacy_models:
            output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
        else:
            if return_schedule_plan:
                schedule_plan = model.build_schedule_plan(
                    tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
                )
                return schedule_plan, partial(pretrain_gpt.loss_func, loss_mask, model=model)
            else:
                output_tensor = model(
                    tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
                )

        return output_tensor, partial(pretrain_gpt.loss_func, loss_mask, model=model)

    pretrain_gpt.forward_step = _patched_forward_step
    log_rank_0(
        "[Patch:megatron.pretrain_gpt.forward_step.zero_bubble]   "
        "Patched pretrain_gpt.forward_step for DataLoaderStore support"
    )
