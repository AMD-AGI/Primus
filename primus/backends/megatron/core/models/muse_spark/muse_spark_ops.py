###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Muse Spark math that has no Megatron dependency.

Every function mirrors one piece of the reference ``modeling_muse_spark.py``
in transformers. ``muse_spark_modules`` wraps them in Megatron modules and
``jit_fuser``; the unit tests import this file directly.
"""

from typing import List

import torch

__all__ = [
    "global_attention_layer_mask",
    "input_scaled_rms_norm",
    "residual_gate",
    "softcap_logits",
]


def global_attention_layer_mask(num_layers: int, interval: int) -> List[bool]:
    """Return ``True`` for the full-attention NoPE layers (0-based).

    Muse counts the pattern backward from the last layer, so with 62 layers
    and ``interval=4`` the global layers are 1, 5, ..., 61. All other layers
    use sliding-window attention with RoPE.
    """
    if interval < 1:
        raise ValueError(f"interval must be >= 1, got {interval}")
    return [(num_layers - 1 - i) % interval == 0 for i in range(num_layers)]


def residual_gate(
    residual: torch.Tensor,
    hidden_states: torch.Tensor,
    gate: torch.Tensor,
    temperature: float,
    eps: float,
) -> torch.Tensor:
    """Variance-preserving residual merge ``alpha * residual + beta * hidden``.

    ``beta = sigmoid(g / T)`` and ``alpha = sqrt(sigmoid(-g / T) * (1 + beta))``,
    so ``alpha**2 + beta**2 == 1`` before the clamp.
    """
    w = gate.float() / temperature
    beta = torch.sigmoid(w)
    alpha = torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1.0 + beta), min=eps))
    return (alpha * residual.float() + beta * hidden_states.float()).to(residual.dtype)


def input_scaled_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Scale by ``1 + weight`` first, round to the input dtype, then RMS-normalize."""
    scaled = (x.float() * (1.0 + weight.float())).to(x.dtype).float()
    return (scaled * torch.rsqrt(scaled.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


def softcap_logits(logits: torch.Tensor, multiplier: float, cap: float) -> torch.Tensor:
    """``cap * tanh(logits * multiplier / cap)``, computed in fp32."""
    return (cap * torch.tanh(logits.float() * (multiplier / cap))).to(logits.dtype)
