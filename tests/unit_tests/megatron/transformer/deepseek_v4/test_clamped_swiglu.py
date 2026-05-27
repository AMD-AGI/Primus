###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for V4 pre-multiplication clamped SwiGLU (G3, plan-2 §04).

These tests pin the math against the HF reference at
``DeepSeek-V4-Flash/inference/model.py:Expert.forward`` and assert the
parameter layout (``w1`` / ``w2`` / ``w3``) so the released checkpoint
loads through the V4 state-dict adapter without remapping.

Tolerance: 1e-6 absolute, fp32, on randomized inputs (G3 gate).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from primus.backends.megatron.core.transformer.clamped_swiglu import (
    ClampedSwiGLUMLP,
    clamped_swiglu_pre_mul,
    clamped_swiglu_pre_mul_fused,
)


def _hf_reference_clamped_swiglu(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    alpha: float,
) -> torch.Tensor:
    """Inline HF reference for ``Expert.forward``'s activation.

    Mirrors the HF code exactly, including the float() cast and the
    pre-multiplication one-sided / two-sided clamp.
    """
    gate = gate.float()
    up = up.float()
    if alpha > 0.0:
        up = torch.clamp(up, min=-alpha, max=alpha)
        gate = torch.clamp(gate, max=alpha)
    return F.silu(gate) * up


@pytest.mark.parametrize("alpha", [7.0, 3.5, 1.0, 0.5])
def test_clamped_swiglu_pre_mul_matches_hf_reference(alpha: float) -> None:
    """G3: split-input form matches HF reference within 1e-6 fp32."""
    torch.manual_seed(1234)
    gate = torch.randn(4, 17, 32, dtype=torch.float32) * 5.0
    up = torch.randn(4, 17, 32, dtype=torch.float32) * 5.0

    out = clamped_swiglu_pre_mul(gate, up, alpha=alpha)
    ref = _hf_reference_clamped_swiglu(gate, up, alpha=alpha)

    assert out.shape == gate.shape
    assert torch.isfinite(out).all()
    max_abs = (out - ref).abs().max().item()
    assert max_abs <= 1.0e-6, f"max-abs error vs HF reference = {max_abs}"


def test_clamped_swiglu_pre_mul_alpha_zero_disables_clamp() -> None:
    """``alpha == 0`` falls back to vanilla SwiGLU (no clamp)."""
    torch.manual_seed(7)
    gate = torch.randn(2, 16, dtype=torch.float32) * 100.0
    up = torch.randn(2, 16, dtype=torch.float32) * 100.0
    out = clamped_swiglu_pre_mul(gate, up, alpha=0.0)
    expected = F.silu(gate) * up
    assert (out - expected).abs().max().item() <= 1.0e-6


def test_clamped_swiglu_pre_mul_fused_matches_split_form() -> None:
    """The fused [gate | up] form matches the split-input form."""
    torch.manual_seed(99)
    I = 24
    gate = torch.randn(3, 11, I, dtype=torch.float32) * 4.0
    up = torch.randn(3, 11, I, dtype=torch.float32) * 4.0
    fused = torch.cat([gate, up], dim=-1)

    out_split = clamped_swiglu_pre_mul(gate, up, alpha=7.0)
    out_fused = clamped_swiglu_pre_mul_fused(fused, alpha=7.0)
    assert (out_split - out_fused).abs().max().item() <= 1.0e-6


def test_clamped_swiglu_pre_mul_one_sided_gate_clamp() -> None:
    """Verify the gate clamp is one-sided (max=alpha) and up is two-sided.

    Sets gate values both well below -alpha and well above +alpha; the
    output activation should pass the negative-gate values through SiLU
    (since gate clamp only bounds the top side) and bound the positive
    gate side at alpha.
    """
    alpha = 7.0
    gate_lo = torch.tensor([[-100.0]], dtype=torch.float32)
    gate_hi = torch.tensor([[+100.0]], dtype=torch.float32)
    up = torch.tensor([[1.0]], dtype=torch.float32)

    out_lo = clamped_swiglu_pre_mul(gate_lo, up, alpha=alpha)
    out_hi = clamped_swiglu_pre_mul(gate_hi, up, alpha=alpha)

    # Gate negative side is NOT clamped; SiLU(-100) ~ -100 * sigmoid(-100) ~ 0.
    # We just assert the output is finite and small (close to 0).
    assert torch.isfinite(out_lo).item()
    assert out_lo.abs().item() < 1.0e-6
    # Gate positive side IS clamped at alpha; SiLU(alpha) is the upper bound.
    expected_hi = F.silu(torch.tensor(alpha)) * up
    assert (out_hi - expected_hi).abs().max().item() <= 1.0e-6


def test_clamped_swiglu_mlp_state_dict_uses_w1_w2_w3_layout() -> None:
    """Released-checkpoint compatibility: w1 / w2 / w3 keys exist.

    The HF released ``Expert`` checkpoint uses ``w1.weight`` /
    ``w2.weight`` / ``w3.weight`` (no ``gate_up.weight``). This test
    fails if a future refactor breaks that promise.
    """
    mlp = ClampedSwiGLUMLP(hidden_size=8, intermediate_size=16, alpha=7.0, bias=False)
    keys = set(mlp.state_dict().keys())
    assert "w1.weight" in keys
    assert "w2.weight" in keys
    assert "w3.weight" in keys
    # No fused projection key may leak into the state-dict.
    assert "gate_up.weight" not in keys


def test_clamped_swiglu_mlp_fused_gate_up_matches_split_path() -> None:
    """The fused-forward variant produces identical outputs to the eager path."""
    torch.manual_seed(42)
    hidden = torch.randn(3, 7, 8, dtype=torch.float32)

    mlp_split = ClampedSwiGLUMLP(hidden_size=8, intermediate_size=16, alpha=7.0)
    mlp_fused = ClampedSwiGLUMLP(hidden_size=8, intermediate_size=16, alpha=7.0, fused_gate_up=True)
    # Share weights so we are only testing the forward layout, not init.
    mlp_fused.load_state_dict(mlp_split.state_dict())

    out_split = mlp_split(hidden)
    out_fused = mlp_fused(hidden)
    assert (out_split - out_fused).abs().max().item() <= 1.0e-6


def test_clamped_swiglu_mlp_forward_matches_hf_expert_forward() -> None:
    """End-to-end: ``ClampedSwiGLUMLP.forward`` matches HF ``Expert.forward``."""
    torch.manual_seed(0)
    H, I = 12, 28
    alpha = 7.0
    mlp = ClampedSwiGLUMLP(hidden_size=H, intermediate_size=I, alpha=alpha)

    x = torch.randn(2, 5, H, dtype=torch.float32)

    # Inline HF reference using the same w1/w2/w3 weights.
    w1 = mlp.w1.weight.detach()
    w3 = mlp.w3.weight.detach()
    w2 = mlp.w2.weight.detach()
    gate = F.linear(x, w1)
    up = F.linear(x, w3)
    h = _hf_reference_clamped_swiglu(gate, up, alpha=alpha)
    ref = F.linear(h, w2)

    out = mlp(x)
    assert (out - ref).abs().max().item() <= 1.0e-6
