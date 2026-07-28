###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Clamped SwiGLU activation for DeepSeek-V4 (pre-mul layout).

Reference: techblog §3 ("Activation: clamped SwiGLU") and the inference
reference at ``DeepSeek-V4-Flash/inference/model.py:Expert.forward``.

DeepSeek-V4 replaces the standard ``SwiGLU(gate, up) = SiLU(gate) * up``
with a **pre-multiplication** clamp:

    gate_c = clamp(gate, max=alpha)             # one-sided (top only)
    up_c   = clamp(up,   min=-alpha, max=alpha) # two-sided
    out    = SiLU(gate_c) * up_c

This matches the released checkpoint and Megatron's
``mlp.MLP``-side clamp path (``activation_func_clamp_value``); both
clamp the gate / up *before* the activation and multiply, so the
post-multiply value is bounded by ``alpha * max(SiLU)`` and stays
well-behaved in bf16 expert summations.

Plan-2 P14 contract:

* :func:`clamped_swiglu_pre_mul` — split-input pointwise activation. Used
  by the eager :class:`ClampedSwiGLUMLP` and as the canonical reference
  for unit tests.
* :func:`clamped_swiglu_pre_mul_fused` — Megatron-fused-input
  ``[..., 2I]`` gate-concat-up form. Lets a grouped-gemm expert backend
  call a single function on the post-GEMM output.
* :class:`ClampedSwiGLUMLP` — eager MLP using **separate** ``w1`` (gate)
  and ``w3`` (up) Linears so the parameter layout matches the
  ``DeepSeek-V4-Flash`` checkpoint (``w1.weight`` / ``w3.weight``).
  An optional ``fused_gate_up`` flag lets callers fuse the two GEMMs at
  forward time without changing the state-dict layout — the saved /
  loaded keys are always ``w1.weight`` / ``w3.weight``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _resolve_alpha(alpha: Optional[float]) -> Optional[float]:
    """Return a clamp bound or ``None`` when clamping is disabled."""
    if alpha is None:
        return None
    if alpha <= 0.0:
        return None
    return float(alpha)


def clamped_swiglu_pre_mul(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    alpha: float = 7.0,
) -> torch.Tensor:
    """Pre-multiplication clamped SwiGLU on split inputs.

    Args:
        gate: ``[..., I]`` gate stream (output of ``w1``).
        up:   ``[..., I]`` up stream (output of ``w3``).
        alpha: clamp bound; V4-Flash default is ``7.0``. Pass ``0`` /
            ``None`` to fall back to vanilla ``SiLU(gate) * up``.

    Returns:
        ``[..., I]`` activation output, ``SiLU(clamp(gate, max=alpha))
        * clamp(up, +/- alpha)``.
    """
    if gate.shape != up.shape:
        raise ValueError(
            "clamped_swiglu_pre_mul expects matching gate / up shapes; "
            f"got {tuple(gate.shape)} vs {tuple(up.shape)}."
        )
    bound = _resolve_alpha(alpha)
    if bound is not None:
        gate_c = gate.clamp(max=bound)
        up_c = up.clamp(min=-bound, max=bound)
    else:
        gate_c = gate
        up_c = up
    return F.silu(gate_c) * up_c


def clamped_swiglu_pre_mul_fused(
    x: torch.Tensor,
    *,
    alpha: float = 7.0,
) -> torch.Tensor:
    """Pre-multiplication clamped SwiGLU on a fused ``[..., 2I]`` input.

    The input is the Megatron-convention concatenated ``[gate | up]``
    along the last dimension (matching
    :func:`megatron.core.fusions.fused_bias_swiglu.bias_swiglu` and the
    eager glu path in :class:`megatron.core.transformer.mlp.MLP`).

    Args:
        x: ``[..., 2 * I]`` — ``[gate | up]`` halves concatenated along
            the last dim.
        alpha: clamp bound; same semantics as :func:`clamped_swiglu_pre_mul`.

    Returns:
        ``[..., I]`` activation output.
    """
    if x.shape[-1] % 2 != 0:
        raise ValueError(
            "clamped_swiglu_pre_mul_fused expects a [gate | up] last dim; "
            f"got shape {tuple(x.shape)} (last dim must be even)."
        )
    gate, up = x.chunk(2, dim=-1)
    return clamped_swiglu_pre_mul(gate, up, alpha=alpha)


class ClampedSwiGLUMLP(nn.Module):
    """Eager SwiGLU MLP with V4's pre-multiplication clamp.

    Computes::

        gate = w1(x)                            # [..., I]
        up   = w3(x)                            # [..., I]
        h    = SiLU(clamp(gate, max=alpha))
                * clamp(up, +/- alpha)
        y    = w2(h)                            # [..., D]

    The parameter layout (``w1`` / ``w2`` / ``w3``) mirrors the released
    DeepSeek-V4-Flash ``Expert`` checkpoint exactly. The ``fused_gate_up``
    knob fuses the gate / up GEMMs at *forward time only* by stacking
    ``w1.weight`` and ``w3.weight`` on the fly; the saved / loaded
    ``state_dict`` keys remain ``w1.weight`` / ``w3.weight`` so released
    checkpoints can be loaded without remapping.

    Notes:
        * Bias is omitted by default to match V4 reference checkpoints.
        * ``alpha=0`` (or ``None``) disables clamping → vanilla SwiGLU.
        * This module is the canonical *eager* reference. Production
          training uses Megatron's grouped-MLP path with
          ``activation_func_clamp_value`` set; the math is the same.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        alpha: float = 7.0,
        bias: bool = False,
        dtype: Optional[torch.dtype] = None,
        fused_gate_up: bool = False,
    ) -> None:
        super().__init__()
        if intermediate_size <= 0:
            raise ValueError(f"intermediate_size must be > 0, got {intermediate_size}")
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.alpha = float(alpha)
        self.fused_gate_up = bool(fused_gate_up)

        kw = {} if dtype is None else {"dtype": dtype}
        # w1 = gate, w3 = up (V4 convention; matches DeepSeek-V4-Flash checkpoint).
        self.w1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias, **kw)
        self.w3 = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias, **kw)
        self.w2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias, **kw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused_gate_up:
            # Build the fused weight on the fly so the state_dict layout
            # is unchanged (w1.weight / w3.weight). Bias is None by default.
            w_gu = torch.cat([self.w1.weight, self.w3.weight], dim=0)
            b_gu = None
            if self.w1.bias is not None and self.w3.bias is not None:
                b_gu = torch.cat([self.w1.bias, self.w3.bias], dim=0)
            gate_up = F.linear(x, w_gu, b_gu)
            h = clamped_swiglu_pre_mul_fused(gate_up, alpha=self.alpha)
        else:
            gate = self.w1(x)
            up = self.w3(x)
            h = clamped_swiglu_pre_mul(gate, up, alpha=self.alpha)
        return self.w2(h)


__all__ = [
    "clamped_swiglu_pre_mul",
    "clamped_swiglu_pre_mul_fused",
    "ClampedSwiGLUMLP",
]
