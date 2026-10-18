###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Opt-in fused LM cross entropy (``use_primus_fused_lm_cross_entropy``).

Routes ``LanguageModule.compute_language_model_loss`` to
``primus.backends.megatron.core.fusions.triton_lm_cross_entropy`` (no fp32 copy of the logits, gradient
written in place). Tensor-parallel size 1 and contiguous ``[s, b, V]`` logits only; other calls keep
Megatron's loss. Takes precedence over ``cross_entropy_loss_fusion`` for the calls it covers.
"""

from primus.backends.megatron.core.fusions._notices import warn_if_not_gfx1250
from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0, warning_rank_0

_PATCH_KEY = "megatron.loss.fused_lm_cross_entropy"
_ARG = "use_primus_fused_lm_cross_entropy"


@register_patch(
    _PATCH_KEY,
    backend="megatron",
    phase="before_train",
    description="Fused LM cross entropy (pinned Triton, TP=1) for LanguageModule.compute_language_model_loss",
    condition=lambda ctx: bool(getattr(get_args(ctx), _ARG, False)),
    tags=["loss", "triton"],
)
def patch_fused_lm_cross_entropy(ctx: PatchContext) -> None:
    warn_if_not_gfx1250(_PATCH_KEY, _ARG)
    from megatron.core.models.common.language_module.language_module import (
        LanguageModule,
    )

    from primus.backends.megatron.core.fusions.triton_lm_cross_entropy import (
        make_compute_language_model_loss,
    )

    if is_patched(LanguageModule, _PATCH_KEY):
        return
    if getattr(get_args(ctx), "cross_entropy_loss_fusion", False):
        warning_rank_0(
            f"[Patch:{_PATCH_KEY}] cross_entropy_loss_fusion is also set; {_ARG} takes precedence "
            "for eligible calls."
        )
    LanguageModule.compute_language_model_loss = make_compute_language_model_loss(
        LanguageModule.compute_language_model_loss, _PATCH_KEY
    )
    mark_patched(LanguageModule, _PATCH_KEY)
    log_rank_0(f"[Patch:{_PATCH_KEY}] patched LanguageModule.compute_language_model_loss (pinned Triton CE)")
