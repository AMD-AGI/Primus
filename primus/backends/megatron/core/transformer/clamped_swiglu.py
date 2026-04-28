###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Clamped SwiGLU activation for DeepSeek-V4.

Reference: techblog §3 ("Activation: clamped SwiGLU") and the inference
reference at ``DeepSeek-V4-Flash/inference/model.py:clamp_swiglu``.

V4 replaces the standard ``SwiGLU(x) = SiLU(x_gate) * x_up`` with a
clamped variant whose post-multiplication output is bounded to
``[-α, α]`` (V4 default ``α = 7.0``). The clamp keeps the expert
output magnitudes well-behaved so the MoE summation does not explode in
bf16.

This module exposes:

* :func:`clamped_swiglu` — the pointwise activation. Input layout is the
  Megatron / V4 convention ``[..., 2 * intermediate]`` where the gate /
  up halves are concatenated along the **last** dim (matching
  Megatron's :func:`~megatron.core.fusions.fused_bias_swiglu.bias_swiglu`).
* :class:`ClampedSwiGLUMLP` — a tiny MLP wrapper used by V4's routed and
  shared experts. Two linears down the hidden axis (gate / up are merged
  into a single ``[..., 2I]`` projection so we can apply the activation
  in one call), then a ``[..., I] → [..., D]`` down projection.

Phase-5 use:
* ``DeepseekV4MoE`` builds a list of :class:`ClampedSwiGLUMLP` for the
  routed experts and one for the shared expert.
* Phase 4's ``DeepseekV4HybridLayer._SwiGLUMLP`` was a placeholder; the
  block wiring is updated in P5 to call the V4 MoE instead.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def clamped_swiglu(x: torch.Tensor, *, alpha: float = 7.0) -> torch.Tensor:
    """Clamped SwiGLU activation.

    Args:
        x: ``[..., 2 * intermediate_size]`` — gate / up halves concatenated
            along the last dim.
        alpha: clamp bound; V4 default is 7.0. Set to ``inf`` (or a very
            large number) to fall back to vanilla SwiGLU.

    Returns:
        ``[..., intermediate_size]`` clamped to ``[-alpha, alpha]``.
    """
    if x.shape[-1] % 2 != 0:
        raise ValueError(
            "clamped_swiglu expects a concatenated [gate | up] last dim; "
            f"got shape {tuple(x.shape)} (last dim must be even)."
        )
    gate, up = x.chunk(2, dim=-1)
    out = F.silu(gate) * up
    if alpha is not None and alpha > 0.0:
        out = out.clamp(min=-alpha, max=alpha)
    return out


class ClampedSwiGLUMLP(nn.Module):
    """Standard SwiGLU MLP block with V4's clamp.

    Computes::

        gate_up = W_gu @ x          # [..., 2 * intermediate]
        h = clamp(silu(gate) * up, min=-alpha, max=alpha)
        y = W_d @ h                 # [..., hidden]

    Notes:
        * ``W_gu`` is a single fused projection so we only do one GEMM
          before the activation. This matches Megatron's fused-SwiGLU
          input layout. (P8 will swap this for a fused activation
          kernel; P5 keeps it eager so we can validate on CPU.)
        * Bias is omitted by default to match V4's reference checkpoints.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        alpha: float = 7.0,
        bias: bool = False,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        if intermediate_size <= 0:
            raise ValueError(f"intermediate_size must be > 0, got {intermediate_size}")
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.alpha = float(alpha)

        kw = {} if dtype is None else {"dtype": dtype}
        self.gate_up = nn.Linear(hidden_size, 2 * intermediate_size, bias=bias, **kw)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=bias, **kw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up(x)
        h = clamped_swiglu(gate_up, alpha=self.alpha)
        return self.down(h)


__all__ = ["clamped_swiglu", "ClampedSwiGLUMLP"]
