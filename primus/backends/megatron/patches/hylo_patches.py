###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
HyLo Distillation Patch
=======================

``model_type: hylo`` trains a hybrid student against a frozen dense teacher. Its registry entry
builds both and gives the pretrain trainer HyLo's forward step and dataset provider
(``training/hylo_distillation.py``). This patch checks the run's HyLo settings before training and
gives the SFT trainer HyLo's SFT forward step in place of its own.

With ``hylo_order_data_dir`` set, the dataset provider serves ``HyLoOrderDataset``: the batches a
reference HyLo (TRL) run recorded, in its order. Megatron's random sampler would reorder them, so
the patch also sets ``dataloader_type`` to ``single``.
"""

from primus.core.patches import PatchContext, get_args, get_param, register_patch
from primus.core.utils.module_utils import log_rank_0

_PATCH_KEY = "megatron.hylo"


@register_patch(
    _PATCH_KEY,
    backend="megatron",
    phase="before_train",
    description="Check HyLo distillation settings and give the SFT trainer HyLo's forward step.",
    condition=lambda ctx: get_param(ctx, "model_type") == "hylo",
)
def patch_hylo(ctx: PatchContext) -> None:
    try:
        from primus.backends.megatron.training import hylo_distillation

        args = get_args(ctx)
        hylo_distillation.validate_hylo_args(args)
        if not hylo_distillation.unset(getattr(args, "hylo_order_data_dir", None)):
            if args.dataloader_type != "single":
                log_rank_0(
                    f"[Patch:{_PATCH_KEY}] dataloader_type {args.dataloader_type} -> single, "
                    "to keep the recorded order"
                )
                args.dataloader_type = "single"
            log_rank_0(f"[Patch:{_PATCH_KEY}] training reads {args.hylo_order_data_dir}")
        if getattr(args, "stage", None) == "sft":
            from primus.backends.megatron.sft import forward_step as sft_forward_step

            sft_forward_step.create_sft_forward_step = hylo_distillation.create_sft_forward_step
        log_rank_0(
            f"[Patch:{_PATCH_KEY}] HyLo {args.hylo_distill_stage} with teacher {args.hylo_teacher_checkpoint}"
        )
    except Exception as exc:
        log_rank_0(f"[Patch:{_PATCH_KEY}] FATAL: {exc}")
        raise SystemExit(f"[{_PATCH_KEY}] HyLo setup failed: {exc}") from exc
