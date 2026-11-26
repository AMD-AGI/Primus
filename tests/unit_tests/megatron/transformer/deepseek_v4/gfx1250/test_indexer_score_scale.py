###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Gluon indexer score with the score scale applied in-kernel (``indexer_score_gluon(..., w_scale=)``).

The frozen indexer used to run ``w_i * score_scale`` (one bf16 elementwise launch) before the Gluon score kernel.
The kernel now reads the unscaled ``w_i`` in place (a row-strided column slice of the fused w_dq_w output) and
rounds ``w * scale`` to w's dtype itself. Checked: the scores are byte-equal to the previous launch (pre-multiplied
contiguous weights, BM 128, waves_per_eu 2) at the DeepSeek-V4 CSA shape and smaller ones; with autograd the scale
falls back to the torch multiply and the gradients are those of the explicit product.
"""

import pytest
import torch

from .helpers import bits_equal, requires_gluon

pytestmark = requires_gluon

DEV = "cuda"


def _G():
    from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common import (
        indexer_score_gluon as G,
    )

    return G


def _inputs(B, S, P, H, D, DQ, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    q = torch.randn(B, S, H, D, device=DEV, generator=g).to(torch.bfloat16)
    k = torch.randn(B, P, D, device=DEV, generator=g).to(torch.bfloat16)
    dqw = torch.randn(B, S, DQ + H, device=DEV, generator=g).to(torch.bfloat16)
    return q, k, dqw[..., DQ:]  # w: column slice of the [B, S, dq_rank + H] projection output


@pytest.mark.parametrize(
    "B,S,P,qo", [(1, 4096, 1024, 0), (2, 1024, 256, 0), (1, 1024, 512, 1024), (1, 1000, 250, 0)]
)
def test_in_kernel_scale_matches_premultiplied_weights(B, S, P, qo):
    G = _G()
    H, D, DQ, ratio = 64, 128, 1024, 4
    scale = (H**-0.5) * (D**-0.5)
    q, k, w = _inputs(B, S, P, H, D, DQ, seed=S + P)
    assert w.stride(1) == DQ + H and not w.is_contiguous()
    wm = (w * scale).contiguous()

    ref = torch.empty((B, S, P), dtype=torch.bfloat16, device=DEV)
    grid, kw = G.launch_args(B, S, P, H, D, ratio, qo, BM=128)
    G._indexer_score_fwd_gluon_kernel[grid](q, k, wm, ref, ref, **kw, num_warps=G.NUM_WARPS, waves_per_eu=2)

    got = G.indexer_score_gluon(
        q, k, w, compress_ratio=ratio, out_dtype=torch.bfloat16, q_offset=qo, w_scale=scale
    )
    pre = G.indexer_score_gluon(q, k, wm, compress_ratio=ratio, out_dtype=torch.bfloat16, q_offset=qo)
    torch.cuda.synchronize()
    assert bits_equal(ref, pre), "BM 64 launch differs from the BM 128 launch"
    assert bits_equal(ref, got), "in-kernel score scale differs from the pre-multiplied weights"


def test_scale_under_autograd_is_the_torch_multiply():
    G = _G()
    B, S, P, H, D, DQ = 1, 256, 64, 8, 128, 64
    scale = 0.0123
    q0, k0, w0 = _inputs(B, S, P, H, D, DQ, seed=5)
    gout = torch.randn(B, S, P, device=DEV)
    res = []
    for in_kernel in (False, True):
        q, k = q0.clone().requires_grad_(True), k0.clone().requires_grad_(True)
        w = w0.clone().requires_grad_(True)
        if in_kernel:
            out = G.indexer_score_gluon(q, k, w, compress_ratio=4, out_dtype=torch.bfloat16, w_scale=scale)
        else:
            out = G.indexer_score_gluon(q, k, w * scale, compress_ratio=4, out_dtype=torch.bfloat16)
        fin = torch.isfinite(out)
        (out.float() * torch.where(fin, gout, 0)).sum().backward()
        torch.cuda.synchronize()
        res.append((out.detach(), q.grad, k.grad, w.grad))
    for name, a, b in zip(("scores", "dq", "dk", "dw"), *res):
        assert bits_equal(a, b), f"{name} differs"
