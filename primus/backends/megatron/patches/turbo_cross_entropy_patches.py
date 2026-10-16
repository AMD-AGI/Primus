###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Primus-Turbo cross entropy for the LM loss (opt-in, ``PRIMUS_TURBO_CROSS_ENTROPY=1``).

``primus_turbo.pytorch.ops.cross_entropy`` reads the logits once in forward (keeping two fp32
statistics per token) and once in backward, where it writes the logits gradient in place and
applies grad_output in fp32. TE's Triton cross entropy makes more passes over the [s, b, vocab]
logits and syncs the host at the start of backward. Only the single-rank vocabulary (TP=1) case is
handled; anything else keeps Megatron's path. The gradient follows current TE arithmetic (fp32
probabilities), so results match TE's to rounding.
"""

import os

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0


def _enabled(ctx: PatchContext) -> bool:
    return os.environ.get("PRIMUS_TURBO_CROSS_ENTROPY", "0").strip().lower() in ("1", "true", "yes", "on")


@register_patch(
    "megatron.turbo_cross_entropy",
    backend="megatron",
    phase="before_train",
    description="LM loss through primus_turbo.pytorch.ops.cross_entropy (TP=1)",
    condition=_enabled,
)
def patch_turbo_cross_entropy(ctx: PatchContext):
    import primus_turbo.pytorch as pt
    from megatron.core.models.common.language_module import language_module as lm

    cls = lm.LanguageModule
    orig = cls.compute_language_model_loss
    if getattr(orig, "_primus_turbo_ce", False):
        return

    def compute_language_model_loss(self, labels, logits):
        tp = getattr(self, "pg_collection", None)
        tp_size = tp.tp.size() if tp is not None and getattr(tp, "tp", None) is not None else 1
        if tp_size != 1 or not self.config.cross_entropy_loss_fusion:
            return orig(self, labels, logits)
        # [b s] => [s b], logits [s b vocab]
        labels = labels.transpose(0, 1).contiguous()
        loss = pt.ops.cross_entropy(logits, labels, overwrite_input=logits.is_contiguous())
        # [s b] => [b s]
        return loss.transpose(0, 1).contiguous()

    compute_language_model_loss._primus_turbo_ce = True
    cls.compute_language_model_loss = compute_language_model_loss
    log_rank_0(
        "[Patch:megatron.turbo_cross_entropy] LM loss uses primus_turbo.pytorch.ops.cross_entropy (TP=1)"
    )
