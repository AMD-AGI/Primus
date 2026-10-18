###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Fused LM cross entropy (pinned Triton, TP = 1)
==============================================

With ``use_primus_fused_lm_cross_entropy: true``,
``LanguageModule.compute_language_model_loss`` runs
``primus.backends.megatron.core.fusions.triton_lm_cross_entropy``: two Triton
kernels with one pinned config (BLOCK 8192, num_warps 8, no autotune). The
forward reads the logits once and keeps the per-row logsumexp, the backward
writes the gradient in place into the saved logits buffer. No fp32 copy of the
logits.

Megatron's own fusions do not fit ROCm / gfx1250 well:
``cross_entropy_fusion_impl: te`` launches Transformer Engine's Triton CE with
``num_warps=16`` on HIP (a gfx1250 hang class), and ``native`` is a
``jit_fuser`` (torch.compile) function, i.e. eager when compilation is disabled.

Scope: tensor-parallel size 1, contiguous ``[s, b, V]`` CUDA logits, ``[b, s]``
integer labels. Any other call (TP > 1, other layouts) runs the original method;
the reason is logged once. When ``cross_entropy_loss_fusion`` is also set, this
patch takes precedence for the calls it covers.
"""

from __future__ import annotations

from primus.backends.megatron.patches._gfx1250_notice import warn_if_not_gfx1250
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0, warning_rank_0

_PATCH_KEY = "megatron.loss.fused_lm_cross_entropy"
_ARG = "use_primus_fused_lm_cross_entropy"

_LOGGED = set()


def _tp_size(module) -> int | None:
    pg = getattr(module, "pg_collection", None)
    tp = getattr(pg, "tp", None) if pg is not None else None
    if tp is not None:
        try:
            return tp.size()
        except Exception:  # noqa: BLE001 - fall back to the global parallel state
            pass
    try:
        from megatron.core import parallel_state

        return parallel_state.get_tensor_model_parallel_world_size()
    except Exception:  # noqa: BLE001
        return None


def make_compute_language_model_loss(orig):
    """Wrap ``LanguageModule.compute_language_model_loss``: fused CE for eligible calls, ``orig`` otherwise."""
    from primus.backends.megatron.core.fusions.triton_lm_cross_entropy import (
        fused_lm_cross_entropy,
        ineligible_reason,
    )

    def compute_language_model_loss(self, labels, logits):
        why = ineligible_reason(logits, labels)
        if why is None:
            tp = _tp_size(self)
            if tp != 1:
                why = f"tensor-parallel size {tp}"
        if why is None:
            return fused_lm_cross_entropy(logits, labels)
        if why not in _LOGGED:
            _LOGGED.add(why)
            warning_rank_0(f"[Patch:{_PATCH_KEY}] original Megatron CE for this call: {why}")
        return orig(self, labels, logits)

    compute_language_model_loss._primus_fused_lm_ce = True
    compute_language_model_loss.__wrapped__ = orig
    return compute_language_model_loss


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
    from megatron.core.models.common.language_module import language_module

    cls = language_module.LanguageModule
    cur = cls.compute_language_model_loss
    if getattr(cur, "_primus_fused_lm_ce", False):
        return
    if getattr(get_args(ctx), "cross_entropy_loss_fusion", False):
        warning_rank_0(
            f"[Patch:{_PATCH_KEY}] cross_entropy_loss_fusion is also set; {_ARG} takes precedence "
            "for eligible calls."
        )
    cls.compute_language_model_loss = make_compute_language_model_loss(cur)
    log_rank_0(f"[Patch:{_PATCH_KEY}] patched LanguageModule.compute_language_model_loss (pinned Triton CE)")
