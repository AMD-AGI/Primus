###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""GLM-5.3 (glm5_next) kernels vs eager references: kpool indexer, sparse MLA, mHC."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _kpool():
    from primus.backends.megatron.core.transformer.glm5_next import kpool_indexer

    return kpool_indexer


@pytest.mark.parametrize("B,S,H", [(1, 300, 32), (2, 2600, 32)])
def test_kpool_scores_match_torch(B, S, H):
    ki = _kpool()
    torch.manual_seed(0)
    D, kpool = 128, 4
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, S, D, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(B, S, D, device="cuda", dtype=torch.bfloat16)
    ape = torch.randn(kpool, D, device="cuda")
    w = torch.randn(B, S, H, device="cuda") * 0.02
    pooled = ki.build_pooled_keys(k, gate, ape, kpool)
    got = ki.kpool_scores(q, pooled, w, kpool)
    ref = ki.kpool_scores_torch(q, pooled, w, kpool)
    assert torch.equal(torch.isinf(got), torch.isinf(ref))
    fin = torch.isfinite(ref)
    torch.testing.assert_close(got[fin], ref[fin], rtol=2e-3, atol=2e-3)


def _select_reference(scores, B, S, topk, kpool):
    rows = []
    for b in range(B):
        for l in range(S):
            if l + 1 <= topk:
                sel = list(range(l + 1))
            else:
                sc = scores[b * S + l]
                n_vis = (l + 1) // kpool
                order = torch.topk(sc[:n_vis], topk // kpool).indices.tolist()
                sel = [p * kpool + j for p in order for j in range(kpool)]
                tail0 = ((l + 1) // kpool) * kpool
                sel += list(range(tail0, l + 1))
            rows.append(sorted(b * S + t for t in sel))
    return rows


def test_kpool_select_matches_reference():
    ki = _kpool()
    torch.manual_seed(1)
    B, S, topk, kpool = 2, 97, 16, 4
    P = S // kpool
    scores = torch.randn(B * S, P, device="cuda")
    pos = torch.arange(S, device="cuda").repeat(B)
    scores = torch.where(
        torch.arange(P, device="cuda")[None] < ((pos + 1) // kpool)[:, None], scores, float("-inf")
    )
    got = ki.kpool_select(scores, B, S, topk, kpool)
    ref = _select_reference(scores, B, S, topk, kpool)
    for r, want in enumerate(ref):
        have = sorted(x for x in got[r].tolist() if x >= 0)
        assert have == want, f"row {r}"


def test_causal_all_indices():
    ki = _kpool()
    idx = ki.causal_all_indices(2, 70, torch.device("cuda"))
    assert idx.shape == (140, 128)
    assert idx[75].tolist()[:6] == [70, 71, 72, 73, 74, 75] and idx[75, 6].item() == -1


@pytest.mark.parametrize("H", [64, 8])
def test_sparse_mla_fwd_bwd_match_torch(H):
    from primus.backends.megatron.core.transformer.glm5_next.sparse_mla import (
        sparse_mla,
        sparse_mla_torch,
    )

    ki = _kpool()
    torch.manual_seed(2)
    B, S, D, kpool, topk = 2, 200, 512, 4, 64
    T = B * S
    scale = 256**-0.5
    P = S // kpool
    pos = torch.arange(S, device="cuda").repeat(B)
    scores = torch.randn(T, P, device="cuda")
    scores = torch.where(
        torch.arange(P, device="cuda")[None] < ((pos + 1) // kpool)[:, None], scores, float("-inf")
    )
    indices = ki.kpool_select(scores, B, S, topk, kpool)

    q = (torch.randn(T, H, D, device="cuda") * 2).bfloat16().requires_grad_()
    kv = torch.randn(T, D, device="cuda").bfloat16().requires_grad_()
    q_r = q.detach().float().requires_grad_()
    kv_r = kv.detach().float().requires_grad_()

    o = sparse_mla(q, kv, indices, scale)
    o_r = sparse_mla_torch(q_r, kv_r, indices, scale)
    do = torch.randn_like(o)
    o.backward(do)
    o_r.backward(do.float())

    def rel(a, b):
        return ((a.float() - b).norm() / b.norm()).item()

    # bf16 storage of o / dq / dkv: ~0.3% relative error vs the fp32 reference.
    assert rel(o, o_r) < 1e-2
    assert rel(q.grad, q_r.grad) < 1e-2
    assert rel(kv.grad, kv_r.grad) < 1e-2


def _sglang_mhc_pre(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps, post_mult, repeat):
    import torch.nn.functional as F

    s, n, h = residual.shape
    x_flat = residual.view(s, n * h).float()
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + rms_eps)
    mixes = F.linear(x_flat, fn) * rsqrt
    pre = torch.sigmoid(mixes[:, :n] * hc_scale[0] + hc_base[:n]) + hc_pre_eps
    post = post_mult * torch.sigmoid(mixes[:, n : 2 * n] * hc_scale[1] + hc_base[n : 2 * n])
    comb = mixes[:, 2 * n :].view(s, n, n) * hc_scale[2] + hc_base[2 * n :].view(n, n)
    comb = comb.softmax(-1) + hc_sinkhorn_eps
    comb = comb / (comb.sum(-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(repeat - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + hc_sinkhorn_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + hc_sinkhorn_eps)
    layer_input = (pre.unsqueeze(-1) * residual.float()).sum(dim=1).to(residual.dtype)
    return post.unsqueeze(-1), comb, layer_input


def _sglang_mhc_post(x, residual, post, comb):
    out = post * x.unsqueeze(1) + (comb.unsqueeze(-1) * residual.unsqueeze(2)).sum(dim=1)
    return out.type_as(x)


def test_mhc_matches_sglang():
    from primus.backends.megatron.core.transformer.glm5_next.mhc import (
        mhc_post,
        mhc_pre,
    )

    torch.manual_seed(3)
    s, b, n, h = 6, 3, 4, 256
    streams = torch.randn(s, b, n, h, device="cuda").bfloat16()
    fn = torch.randn((2 + n) * n, n * h, device="cuda") * 0.05
    scale = torch.rand(3, device="cuda") + 0.5
    base = torch.randn((2 + n) * n, device="cuda")
    x, post, comb = mhc_pre(
        streams, fn, scale, base, rms_eps=1e-5, hc_eps=1e-6, sinkhorn_iters=20, post_mult=2.0
    )
    post_r, comb_r, x_r = _sglang_mhc_pre(
        streams.view(s * b, n, h), fn, scale, base, 1e-5, 1e-6, 1e-6, 2.0, 20
    )
    torch.testing.assert_close(x.view(s * b, h), x_r, rtol=0, atol=0)
    torch.testing.assert_close(comb.view(s * b, n, n), comb_r)

    f = torch.randn(s, b, h, device="cuda").bfloat16()
    out = mhc_post(f, streams, post, comb)
    out_r = _sglang_mhc_post(f.view(s * b, h), streams.view(s * b, n, h), post_r, comb_r)
    torch.testing.assert_close(out.view(s * b, n, h).float(), out_r.float(), rtol=1e-2, atol=1e-2)
