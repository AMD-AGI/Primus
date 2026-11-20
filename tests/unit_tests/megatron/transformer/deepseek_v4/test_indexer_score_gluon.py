###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Fused Gluon indexer score (PRIMUS_INDEXER_GLUON): CPU checks of the kernel's tiling
(emulated in torch) against the current einsum + indexer_score_post semantics, and of the
autograd wrapper's backward (recompute einsum + tail bwd + einsum bwd)."""

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common import (  # noqa: E402
    indexer_score_gluon as G,
)


def _post_ref(q, k, w, ratio, q_offset, out_dtype):
    dot = torch.einsum("bshd,bpd->bshp", q, k)
    sc = (F.relu(dot.float()) * w.float().unsqueeze(-1)).sum(dim=2)
    s = torch.arange(sc.shape[1])[:, None] + q_offset
    p = torch.arange(sc.shape[2])[None, :]
    return torch.where((p + 1) * ratio - 1 <= s, sc, torch.full_like(sc, float("-inf"))).to(out_dtype)


@pytest.mark.parametrize(
    "B,S,P,H,D,BM,BN,qo",
    [(1, 256, 64, 4, 128, 128, 128, 0), (2, 200, 50, 8, 128, 32, 32, 0), (1, 128, 160, 2, 128, 32, 64, 384)],
)
def test_tiled_forward_matches_post_path(B, S, P, H, D, BM, BN, qo):
    torch.manual_seed(0)
    q = torch.randn(B, S, H, D).bfloat16()
    k = torch.randn(B, P, D).bfloat16()
    w = torch.randn(B, S, H).bfloat16()
    ref = _post_ref(q, k, w, 4, qo, torch.bfloat16)
    out = G.indexer_score_fwd_tiled_reference(
        q, k, w, compress_ratio=4, out_dtype=torch.bfloat16, q_offset=qo, BM=BM, BN=BN
    )
    assert torch.equal(torch.isneginf(out), torch.isneginf(ref))
    fin = torch.isfinite(ref)
    torch.testing.assert_close(out.float()[fin], ref.float()[fin], rtol=8e-3, atol=1e-2)


def test_autograd_matches_reference():
    torch.manual_seed(1)
    B, S, P, H, D = 1, 128, 32, 4, 128
    q0, k0, w0 = torch.randn(B, S, H, D).bfloat16(), torch.randn(B, P, D).bfloat16(), torch.randn(B, S, H).bfloat16()
    g = torch.randn(B, S, P)
    q1, k1, w1 = (t.clone().requires_grad_() for t in (q0, k0, w0))
    o1 = G.indexer_score_gluon(q1, k1, w1, compress_ratio=4, out_dtype=torch.bfloat16)
    fin = torch.isfinite(o1)
    (o1.float() * torch.where(fin, g, 0)).sum().backward()
    q2, k2, w2 = (t.clone().requires_grad_() for t in (q0, k0, w0))
    o2 = _post_ref(q2, k2, w2, 4, 0, torch.bfloat16)
    (o2.float() * torch.where(fin, g, 0)).sum().backward()
    for a, b in ((q1.grad, q2.grad), (k1.grad, k2.grad), (w1.grad, w2.grad)):
        torch.testing.assert_close(a.float(), b.float(), rtol=2e-2, atol=2e-2)


def test_gate_default_off(monkeypatch):
    monkeypatch.delenv("PRIMUS_INDEXER_GLUON", raising=False)
    assert not G.is_gluon_path_enabled()
    assert not G.is_gluon_kernel_supported(torch.empty(1, 1, 4, 128), torch.empty(1, 1, 128), torch.empty(1, 1, 4))
