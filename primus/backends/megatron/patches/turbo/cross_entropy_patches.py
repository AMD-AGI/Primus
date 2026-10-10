# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""Opt-in Turbo TP=1 vocabulary CE without replacing installed TE files."""

from functools import wraps

import torch

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0


def _enabled(ctx: PatchContext) -> bool:
    return bool(getattr(get_args(ctx), "use_turbo_cross_entropy", False))


@register_patch(
    "megatron.turbo.cross_entropy",
    backend="megatron",
    phase="before_train",
    description="Use Turbo fused CE with deferred backward for TP=1 BF16/FP32 logits",
    condition=_enabled,
)
def patch_cross_entropy(ctx: PatchContext) -> None:
    from megatron.core.models.common.language_module.language_module import (
        LanguageModule,
    )
    from primus_turbo.pytorch.ops.cross_entropy import cross_entropy

    original = LanguageModule.compute_language_model_loss
    if getattr(original, "_primus_turbo_ce", False):
        return
    overwrite_input = bool(getattr(get_args(ctx), "turbo_ce_overwrite_input", False))

    @wraps(original)
    def compute_language_model_loss(self, labels, logits):
        if (
            self.config.tensor_model_parallel_size != 1
            or logits.dtype not in (torch.bfloat16, torch.float32)
            or not self.config.cross_entropy_loss_fusion
        ):
            return original(self, labels, logits)
        # Core supplies sequence-first logits and batch-first targets. Keep
        # the original per-token loss contract; training owns masks/scaling.
        target = labels.transpose(0, 1).contiguous()
        loss = cross_entropy(
            logits,
            target,
            overwrite_input=overwrite_input and logits.is_contiguous(),
        )
        return loss.transpose(0, 1).contiguous()

    compute_language_model_loss._primus_turbo_ce = True
    LanguageModule.compute_language_model_loss = compute_language_model_loss
    log_rank_0(
        "[Patch:megatron.turbo.cross_entropy] Enabled Turbo CE for TP=1 BF16/FP32; "
        f"overwrite_input={overwrite_input}; other cases retain the configured loss"
    )
