###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Q-path RoPE / q-norm kernels.

* ``PRIMUS_GLUE_ROPE_NOCAST``: RoPE, inverse RoPE and the fused q-norm + RoPE give bit-identical outputs and
  gradients with the position / inverse-frequency casts removed, for fp32 and bf16 ``inv_freq``.
* fp16 activations go through the pair-split forward kernel (``PRIMUS_V4_QNR_SPLIT=1``) and agree with an fp32
  reference of RMSNorm + interleaved partial RoPE; the padding columns are zero.
"""

import pytest
import torch

from .helpers import bits_equal, requires_gfx1250

pytestmark = requires_gfx1250

DEV = "cuda"
_PKG = "primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common"


@pytest.mark.parametrize("invdt", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("kind", ["rope", "rope_inv", "qnr"])
def test_rope_nocast_is_bit_identical(monkeypatch, kind, invdt):
    import importlib

    RP = importlib.import_module(f"{_PKG}.rope_interleaved_partial")
    Q = importlib.import_module(f"{_PKG}.q_norm_rope")
    B, S, H, D, rd = 1, 256, 4, 512, 64
    pos = torch.arange(S, device=DEV).view(1, S)
    inv = (1.0 / (10000 ** (torch.arange(0, rd, 2, device=DEV).float() / rd))).to(invdt)
    outs = {}
    for nc in ("0", "1"):
        monkeypatch.setenv("PRIMUS_GLUE_ROPE_NOCAST", nc)
        torch.manual_seed(1)
        x = torch.randn(B, S, H, D, device=DEV, dtype=torch.bfloat16, requires_grad=True)
        if kind == "qnr":
            y = Q.q_norm_rope_pad(x, pos.reshape(-1), inv, rotary_dim=rd, eps=1e-6, pad=64)
        else:
            y = RP.apply_rope_from_positions(x, pos, inv, rotary_dim=rd, inverse=(kind == "rope_inv"))
        y.backward(torch.randn(y.shape, device=DEV, dtype=y.dtype))
        outs[nc] = (y.detach().clone(), x.grad.clone())
    assert bits_equal(outs["0"][0].contiguous(), outs["1"][0].contiguous()), "output differs"
    assert bits_equal(outs["0"][1], outs["1"][1]), "gradient differs"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_q_norm_rope_forward_fp16_vs_fp32_reference(monkeypatch, dtype):
    import importlib

    Q = importlib.import_module(f"{_PKG}.q_norm_rope")
    monkeypatch.setenv("PRIMUS_V4_QNR_SPLIT", "1")
    B, S, H, D, rd, pad = 1, 128, 8, 512, 64, 64
    torch.manual_seed(5)
    x = torch.randn(B, S, H, D, device=DEV).to(dtype)
    assert Q.q_norm_rope_supported(x, rd)
    pos = torch.arange(S, device=DEV)
    inv = 1.0 / (10000 ** (torch.arange(0, rd, 2, device=DEV).float() / rd))
    out = Q.q_norm_rope_pad(x, pos, inv, rotary_dim=rd, eps=1e-6, pad=pad)
    torch.cuda.synchronize()
    assert out.dtype == dtype and out.shape == x.shape
    buf_pad = out.as_strided((B, S, H, pad), out.stride(), storage_offset=out.storage_offset() + D)
    assert not buf_pad.float().any(), "padding columns are not zero"

    xf = x.float()
    xn = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
    nope = D - rd
    ang = pos.float().view(1, S, 1, 1) * inv.view(1, 1, 1, -1)
    c, s = torch.cos(ang), torch.sin(ang)
    a, b = xn[..., nope::2], xn[..., nope + 1 :: 2]
    ref = xn.clone()
    ref[..., nope::2] = a * c - b * s
    ref[..., nope + 1 :: 2] = a * s + b * c
    err = (out.float() - ref).abs().max().item() / ref.abs().max().item()
    assert err < (2e-3 if dtype == torch.float16 else 1.5e-2), f"max relative error {err:.2e}"
