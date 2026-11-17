###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Run the dense MLP's clamped quick-GEGLU (``swigluoai``) as one Triton kernel each way.

With ``fused_clamped_geglu`` Megatron ``MLP``'s eager clamped quick-GEGLU branch becomes
``primus/backends/megatron/core/fusions/fused_clamped_geglu.py``, bitwise equal to it.
``PrimusGroupedMLP`` (routed experts) reads the same option in its activation.

Only MLPs that reach Megatron's eager GLU branch with ``quick_gelu`` and a clamp bound
are affected (no TE activation module, no bias-activation fusion, no FP8 / FP4, no
per-token scale); everything else runs Megatron's forward.
"""

from __future__ import annotations

from typing import Optional

import torch

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

_PATCH_KEY = "megatron.mlp.fused_clamped_geglu"
_DECISION_LOGGED = False


def _mlp_eligible(mlp) -> bool:
    cached = getattr(mlp, "_primus_fused_clamped_geglu", None)
    if cached is not None:
        return cached
    from megatron.core.fusions.fused_bias_geglu import quick_gelu

    cfg = mlp.config
    mlp._primus_fused_clamped_geglu = bool(
        cfg.gated_linear_unit
        and cfg.activation_func is quick_gelu
        and mlp.activation_func is quick_gelu
        and getattr(cfg, "activation_func_clamp_value", None) is not None
        and not getattr(cfg, "use_te_activation_func", False)
        and not cfg.bias_activation_fusion
        and not getattr(cfg, "fp8", None)
        and not getattr(cfg, "fp4", None)
    )
    return mlp._primus_fused_clamped_geglu


def _fc1_width(mlp) -> Optional[int]:
    """Output width of ``linear_fc1`` on this rank, from its ``[out, in]`` weight."""
    weight = getattr(mlp.linear_fc1, "weight", None)
    return weight.shape[0] if isinstance(weight, torch.Tensor) and weight.dim() == 2 else None


def _log_decision_once(reason: Optional[str]) -> None:
    global _DECISION_LOGGED
    if _DECISION_LOGGED:
        return
    _DECISION_LOGGED = True
    if reason is None:
        log_rank_0(f"[Patch:{_PATCH_KEY}] MLP swigluoai: one bitwise Triton kernel each way")
    else:
        log_rank_0(f"[Patch:{_PATCH_KEY}] MLP swigluoai: Megatron's eager branch ({reason})")


def _install() -> None:
    from megatron.core.transformer.mlp import MLP

    from primus.backends.megatron.core.fusions.fused_clamped_geglu import (
        fused_clamped_quick_geglu,
        ineligible_reason,
    )

    if getattr(MLP.forward, "_primus_fused_clamped_geglu", False):
        return
    original_forward = MLP.forward

    def forward(self, hidden_states, per_token_scale=None, **kwargs):
        # Megatron's MLP.forward ignores **kwargs (the transformer layer passes padding_mask).
        width = _fc1_width(self) if per_token_scale is None and _mlp_eligible(self) else None
        if width is None:
            return original_forward(self, hidden_states, per_token_scale, **kwargs)
        limit, offset = self.config.activation_func_clamp_value, self.config.glu_linear_offset
        shape = (*hidden_states.shape[:-1], width)
        reason = ineligible_reason(shape, hidden_states.dtype, hidden_states.device, limit, offset)
        _log_decision_once(reason)
        if reason is not None:
            return original_forward(self, hidden_states, per_token_scale, **kwargs)
        intermediate, bias = self.linear_fc1(hidden_states)
        if bias is not None:
            intermediate = intermediate + bias
        if intermediate.dtype != hidden_states.dtype or tuple(intermediate.shape) != shape:
            raise RuntimeError(
                f"[Patch:{_PATCH_KEY}] linear_fc1 returned {intermediate.dtype} {tuple(intermediate.shape)}, "
                f"not the {hidden_states.dtype} {shape} the fused path was chosen for"
            )
        return self.linear_fc2(fused_clamped_quick_geglu(intermediate, limit, offset))

    forward._primus_fused_clamped_geglu = True
    MLP.forward = forward


@register_patch(
    _PATCH_KEY,
    backend="megatron",
    phase="before_train",
    description="MLP clamped quick-GEGLU (swigluoai) as one bitwise Triton kernel each way",
    condition=lambda ctx: bool(getattr(get_args(ctx), "fused_clamped_geglu", False)),
)
def patch_fused_clamped_geglu(ctx: PatchContext) -> None:
    del ctx
    _install()
    log_rank_0(f"[Patch:{_PATCH_KEY}] installed; the first eligible MLP logs which path it takes")
