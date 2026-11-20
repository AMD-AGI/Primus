###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused LeRoPE kernel (``lerope_kernel``) against the eager path, forward and backward
(``x`` and the inverse frequencies, so ``alpha``). The kernel cases run on a GPU only."""

import pytest
import torch

from primus.backends.megatron.core.transformer.mistral_large4.lerope_kernel import (
    fused_lerope_available,
    lerope_rotary,
    lerope_rotary_reference,
)
from primus.backends.megatron.core.transformer.mistral_large4.rotary import (
    yarn_inv_freq,
)

GPU = torch.cuda.is_available()


def _inputs(s, b, h, dim, dtype, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(s, b, h, dim, generator=g).to(device=device, dtype=dtype)
    pos = torch.cat([torch.tensor([0, 511, 4095]), torch.randint(0, 1 << 16, (s,), generator=g)])[:s]
    alpha = (torch.randn(dim // 2, generator=g) * 0.3).to(device)
    return x, pos.to(device), alpha


def _inv(alpha):
    return yarn_inv_freq(alpha.numel() * 2, 10000.0, 2048.0, 512, 32.0, 1.0).to(alpha.device) * alpha.exp()


def test_reference_is_the_interleaved_rotation():
    """The eager path rotates pairs (x[2i], x[2i+1]) by pos * inv_freq[i], output [evens | odds]."""
    x, pos, alpha = _inputs(5, 2, 3, 8, torch.float64, "cpu")
    inv = _inv(alpha).double()
    y = lerope_rotary_reference(x, pos, inv)
    # the model computes the angles in fp32 (positions.float() * inv_freq.float())
    ang = (pos.float()[:, None] * inv.float()[None, :]).double()
    c, s = ang.cos()[:, None, None, :], ang.sin()[:, None, None, :]
    x0, x1 = x[..., 0::2], x[..., 1::2]
    torch.testing.assert_close(y, torch.cat([x0 * c - x1 * s, x1 * c + x0 * s], -1))


def test_cpu_falls_back_to_the_reference_and_alpha_gets_a_gradient():
    x, pos, alpha = _inputs(6, 1, 2, 8, torch.float32, "cpu")
    alpha.requires_grad_(True)
    assert not fused_lerope_available(x)
    y = lerope_rotary(x, pos, _inv(alpha))
    y.square().sum().backward()
    assert alpha.grad is not None and torch.isfinite(alpha.grad).all() and alpha.grad.abs().sum() > 0


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if GPU else []))
def test_empty_sequence_returns_empty(device):
    x = torch.empty(0, 2, 3, 8, device=device)
    alpha = torch.zeros(4, device=device)
    y = lerope_rotary(x, torch.empty(0, dtype=torch.long, device=device), _inv(alpha))
    assert y.shape == x.shape


def test_positions_must_match_the_rows():
    x, pos, alpha = _inputs(4, 1, 1, 8, torch.float32, "cpu")
    with pytest.raises(ValueError, match="positions"):
        lerope_rotary(x, pos[:3], _inv(alpha))


@pytest.mark.skipif(not GPU, reason="the fused kernel runs on a GPU")
@pytest.mark.parametrize("dtype,tol", [(torch.float32, 1e-5), (torch.bfloat16, 2e-2)])
@pytest.mark.parametrize("shape", [(64, 2, 16, 64), (37, 1, 33, 64), (1, 1, 1, 64), (19, 3, 1, 64)])
# alpha frozen (the model's buffer: the kernel's NEED_DINV=False path, dx only) or trainable
@pytest.mark.parametrize("alpha_grad", [False, True])
def test_kernel_matches_reference_forward_and_backward(shape, dtype, tol, alpha_grad):
    s, b, h, dim = shape
    x, pos, alpha = _inputs(s, b, h, dim, dtype, "cuda", seed=s + h)
    # the comparison is only meaningful if lerope_rotary takes the fused kernel (Triton present)
    assert fused_lerope_available(x), "Triton is missing: lerope_rotary would run the eager path"
    dy = torch.randn(x.shape, generator=torch.Generator(device="cpu").manual_seed(1)).to("cuda", dtype)
    outs = []
    for fn in (lerope_rotary, lerope_rotary_reference):
        xx = x.clone().requires_grad_(True)
        aa = alpha.clone().requires_grad_(alpha_grad)
        xr = xx if fn is lerope_rotary else xx.float()  # reference in fp32, as an L2 spec runs it
        y = fn(xr, pos, _inv(aa))
        (y.float() * dy.float()).sum().backward()
        out = (y.float(), xx.grad.float())
        if alpha_grad:
            out += (aa.grad.float(),)
        else:
            assert aa.grad is None
        outs.append(out)
    for got, ref in zip(outs[0], outs[1]):
        scale = ref.abs().max().clamp_min(1e-12)
        assert float((got - ref).abs().max() / scale) <= tol


@pytest.mark.skipif(not GPU, reason="the fused kernel runs on a GPU")
def test_kernel_backward_is_deterministic():
    x, pos, alpha = _inputs(128, 2, 16, 64, torch.bfloat16, "cuda")
    assert fused_lerope_available(x), "Triton is missing: lerope_rotary would run the eager path"
    grads = []
    for _ in range(3):
        aa = alpha.clone().requires_grad_(True)
        lerope_rotary(x, pos, _inv(aa)).float().square().sum().backward()
        grads.append(aa.grad.clone())
    assert all(torch.equal(grads[0], g) for g in grads[1:])
