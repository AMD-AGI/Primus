###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""In-place inverse RoPE of the attention output (``PRIMUS_V4_OROPE_INPLACE=1``, with
``PRIMUS_V4_ROPE_BWD_INPLACE=1``) against the out-of-place inverse RoPE kernel.

The attention core is replaced by a stand-in whose backward reads its saved output, so the test also checks that
the in-place rotation is undone before the saved output is used: rotated tail bytes, NoPE values (up to the sign
of zero), the o_a stand-in output, the input gradient and the restored saved buffer must all match.
"""

import pytest
import torch

from .helpers import bits_equal, requires_gfx1250

pytestmark = requires_gfx1250

DEV = "cuda"


class _Core(torch.autograd.Function):
    """Stand-in for the attention core: output is the buffer it saves, backward depends on its bytes."""

    @staticmethod
    def forward(ctx, q):
        o = q * 1.0
        ctx.save_for_backward(o)
        return o

    @staticmethod
    def backward(ctx, do):
        (o,) = ctx.saved_tensors
        return do * 2.0 + o


@pytest.mark.parametrize(
    "B,S,H,D,RD,inv_dtype,seed",
    [
        (1, 8, 4, 128, 64, torch.float32, 0),
        (1, 512, 8, 512, 64, torch.float32, 1),
        (1, 512, 8, 512, 64, torch.bfloat16, 2),
        (2, 128, 8, 512, 64, torch.float32, 3),
    ],
)
def test_inplace_inverse_rope_matches_out_of_place(monkeypatch, B, S, H, D, RD, inv_dtype, seed):
    monkeypatch.setenv("PRIMUS_V4_OROPE_INPLACE", "1")
    monkeypatch.setenv("PRIMUS_V4_ROPE_BWD_INPLACE", "1")
    from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common import (
        rope_interleaved_partial as R,
    )

    torch.manual_seed(seed)
    q = (torch.randn(B, S, H, D, device=DEV) * 3).to(torch.bfloat16)
    q[0, 0, 0, :8] = 0.0
    q[0, 1, 0, D - RD : D - RD + 4] = -0.0
    pos = torch.arange(S, device=DEV).unsqueeze(0).expand(B, S)
    inv = (1.0 / (10000.0 ** (torch.arange(0, RD, 2, device=DEV).float() / RD))).to(inv_dtype)
    w = (torch.randn(H * D, 64, device=DEV) * 0.05).to(torch.bfloat16)
    g = torch.randn(B, S, 64, device=DEV).to(torch.bfloat16)

    res, restored = {}, None
    for name in ("ref", "inp"):
        qq = q.clone().requires_grad_(True)
        o = _Core.apply(qq)
        if name == "ref":
            r = R.apply_rope_from_positions(o, pos, inv, rotary_dim=RD, inverse=True)
        else:
            r = R.out_inv_rope_inplace(o, pos, inv, rotary_dim=RD)
            assert r.data_ptr() == o.data_ptr()
        y = r.reshape(B, S, H * D) @ w
        rc = r.detach().clone()
        y.backward(g)
        torch.cuda.synchronize()
        if name == "inp":
            restored = bits_equal(o.detach(), q * 1.0)
        res[name] = (rc, y.detach().clone(), qq.grad.clone())

    (rr, yr, gr), (ri, yi, gi) = res["ref"], res["inp"]
    assert bits_equal(
        rr[..., D - RD :].contiguous(), ri[..., D - RD :].contiguous()
    ), "rotated tail bytes differ"
    assert torch.equal(rr[..., : D - RD], ri[..., : D - RD]), "NoPE values differ"
    assert bits_equal(yr, yi), "o_a stand-in output differs"
    assert bits_equal(gr, gi), "input gradient differs"
    assert restored, "saved attention output was not restored after backward"
