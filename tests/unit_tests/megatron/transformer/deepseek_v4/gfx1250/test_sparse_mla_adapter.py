###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""V4 sparse-MLA adapter routing on gfx1250.

The adapter (``make_attention`` / ``make_csa_from_pool``) is driven with the Triton-v2 kernel entry points and
routed to the Primus-Turbo gfx1250 kernels (``PRIMUS_DSA_GFX1250_TURBO=1``). Checked for the three layer kinds
(compress ratio 0 = SWA, 4 = CSA, 128 = HCA), through forward and backward including the autograd glue:

* band kernels (``PRIMUS_V4_BAND_KERNELS=1``) agree with the index-driven kernels (``=0``) and are run-to-run
  deterministic;
* the Turbo path agrees with the kept Triton fallback (``PRIMUS_DSA_GFX1250_TURBO=0``) within bf16 tolerance;
* ``PRIMUS_LAUNCH_DSINK=2`` hands the sink's ``main_grad`` to the Turbo backward (``dsink_main_grad``).
"""

import pytest
import torch

from .helpers import requires_turbo_flydsl

pytestmark = requires_turbo_flydsl

S, H, D, W = 2048, 64, 512, 128
TOPK = 256
DEV = "cuda"


def _rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp(min=1e-30)).item()


@pytest.fixture(scope="module")
def adapter():
    from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_v2.dsa_bwd_v4_triton import (
        sparse_mla_bwd_v4_triton,
    )
    from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_v2.dsa_fwd_v4_triton import (
        sparse_mla_fwd_v4_triton,
    )
    from primus.backends.megatron.core.transformer.v4_attention_kernels.v4_sparse_mla_adapter import (
        make_attention,
        make_csa_from_pool,
    )

    return (
        make_csa_from_pool(sparse_mla_fwd_v4_triton, sparse_mla_bwd_v4_triton),
        make_attention(sparse_mla_fwd_v4_triton, sparse_mla_bwd_v4_triton),
    )


def _inputs(cr, seed=0):
    g = torch.Generator(device=DEV)
    g.manual_seed(seed)
    q = torch.randn(1, H, S, D, device=DEV, generator=g).bfloat16()
    k = torch.randn(1, S, 1, D, device=DEV, generator=g).bfloat16()
    P = 0 if cr == 0 else S // cr
    pool = torch.randn(1, max(P, 1), D, device=DEV, generator=g).bfloat16()[:, :P]
    sink = torch.randn(H, device=DEV, generator=g)
    do = (torch.randn(1, H, S, D, device=DEV, generator=g) * 0.05).bfloat16()
    t = torch.arange(S, device=DEV).view(S, 1)
    topk_idxs = mask = None
    if cr == 4:
        vis = (torch.arange(P, device=DEV).view(1, P) + 1) * cr - 1 <= t
        sc = torch.rand(S, P, device=DEV, generator=g).masked_fill(~vis, float("-inf"))
        v, i = sc.topk(TOPK, dim=1)
        topk_idxs = torch.where(torch.isinf(v), torch.full_like(i, -1), i).view(1, S, TOPK)
    if cr == 128:
        s_end = (torch.arange(P, device=DEV).view(1, P) + 1) * cr - 1
        mask = torch.where(s_end <= t, 0.0, float("-inf")).bfloat16()
    return q, k, pool, sink, do, topk_idxs, mask


def _run(adapter, cr, data, sk=None):
    csa, att = adapter
    q, k, pool, sink, do, topk_idxs, mask = data
    qq = q.clone().requires_grad_(True)
    sk = sink.clone().requires_grad_(True) if sk is None else sk
    scale = D**-0.5
    if cr == 4:
        kk = k.clone().requires_grad_(True)
        pp = pool.clone().requires_grad_(True)
        out = csa(
            qq,
            kk,
            None,
            pp,
            topk_idxs=topk_idxs,
            sink=sk,
            swa_window=W,
            attn_dropout=0.0,
            training=True,
            scale=scale,
            k_is_latent=True,
        )
        out.backward(do)
        return [out.detach(), qq.grad, kk.grad, pp.grad, sk.grad]
    kv = torch.cat([k, pool.unsqueeze(2)], 1) if cr == 128 else k
    kk = kv.clone().requires_grad_(True)
    out = att(
        qq,
        kk,
        None,
        sink=sk,
        swa_window=W,
        additive_mask=mask,
        attn_dropout=0.0,
        training=True,
        scale=scale,
        hca_local_seqlen=S if cr == 128 else 0,
        k_is_latent=True,
    )
    out.backward(do)
    return [out.detach(), qq.grad, kk.grad, sk.grad]


def _names(cr):
    return ["out", "dq", "dk", "dpool", "dsink"] if cr == 4 else ["out", "dq", "dkv", "dsink"]


@pytest.mark.parametrize("cr", [0, 4, 128])
def test_band_kernels_match_index_driven(adapter, monkeypatch, cr):
    torch.manual_seed(0)
    monkeypatch.setenv("PRIMUS_DSA_GFX1250_TURBO", "1")
    data = _inputs(cr)
    res = {}
    for band in ("0", "1"):
        monkeypatch.setenv("PRIMUS_V4_BAND_KERNELS", band)
        r = _run(adapter, cr, data)
        r2 = _run(adapter, cr, data)
        assert all(torch.equal(a, b) for a, b in zip(r, r2)), f"band={band}: run-to-run result differs"
        res[band] = r
    for n, a, b in zip(_names(cr), res["1"], res["0"]):
        assert _rel(a, b) < 2e-2, f"{n}: band vs index-driven rel L2 {_rel(a, b):.3e}"


@pytest.mark.parametrize("cr", [0, 4, 128])
def test_turbo_matches_triton_fallback(adapter, monkeypatch, cr):
    """PRIMUS_DSA_GFX1250_TURBO=0 keeps the Triton-v2 kernels; both paths compute the same attention."""
    torch.manual_seed(0)
    data = _inputs(cr)
    res = {}
    for turbo in ("0", "1"):
        monkeypatch.setenv("PRIMUS_DSA_GFX1250_TURBO", turbo)
        res[turbo] = _run(adapter, cr, data)
    for n, a, b in zip(_names(cr), res["1"], res["0"]):
        assert torch.isfinite(a.float()).all(), f"{n}: non-finite Turbo result"
        assert _rel(a, b) < 3e-2, f"{n}: Turbo vs Triton fallback rel L2 {_rel(a, b):.3e}"


@pytest.mark.parametrize("cr", [0, 4])
def test_launch_dsink_writes_main_grad(adapter, monkeypatch, cr):
    """PRIMUS_LAUNCH_DSINK=2 with a bf16 sink Parameter that has a ``main_grad``: the backward passes that buffer to
    the Primus-Turbo kernel as ``dsink_main_grad`` (one finalize launch per backward) and the sink gradient is
    accumulated there, matching the autograd sink gradient of the plain path."""
    from primus_turbo.triton.attention import sparse_mla_dsink as sd

    torch.manual_seed(0)
    monkeypatch.setenv("PRIMUS_DSA_GFX1250_TURBO", "1")
    monkeypatch.delenv("PRIMUS_LAUNCH_SINK_CAST", raising=False)
    q, k, pool, sink, do, topk_idxs, mask = _inputs(cr)
    data = (q, k, pool, sink.bfloat16().float(), do, topk_idxs, mask)  # a sink that bf16 holds exactly
    monkeypatch.setenv("PRIMUS_LAUNCH_DSINK", "0")
    ref = _run(adapter, cr, data)[-1]

    calls = []
    orig = sd.sparse_mla_dsink_mg2
    monkeypatch.setattr(sd, "sparse_mla_dsink_mg2", lambda *a, **kw: (calls.append(1), orig(*a, **kw))[1])
    monkeypatch.setenv("PRIMUS_LAUNCH_DSINK", "2")
    sk = torch.nn.Parameter(data[3].bfloat16())
    sk.main_grad = torch.zeros_like(sk)
    sk.grad_added_to_main_grad = False
    for n in (1, 2):
        _run(adapter, cr, data, sk=sk)
        torch.cuda.synchronize()
        assert len(calls) == n, "the sink gradient did not go through dsink_main_grad"
        err = _rel(sk.main_grad, n * ref)
        assert err < 2e-2, f"main_grad after {n} backward(s): rel L2 {err:.3e}"


def _run_csa_side_stream(adapter, data, side):
    """CSA forward / backward with a CSADkvHolder (PRIMUS_OVERLAP CSA_BWD): the backward leaves the dKV gathers on
    ``side`` and the latent's consumer waits in CSADkvJoin, as in DeepseekV4Attention."""
    from primus.backends.megatron.core.transformer.v4_attention_kernels import (
        v4_sparse_mla_adapter as smla,
    )

    csa, _ = adapter
    q, k, pool, sink, do, topk_idxs, _ = data
    qq, kk, pp = (t.clone().requires_grad_(True) for t in (q, k, pool))
    sk = sink.clone().requires_grad_(True)
    holder = smla.CSADkvHolder(side)
    kj = smla.CSADkvJoin.apply(kk, holder)
    smla._CSA_DKV_SLOT[0] = holder
    try:
        out = csa(
            qq,
            kj,
            None,
            pp,
            topk_idxs=topk_idxs,
            sink=sk,
            swa_window=W,
            attn_dropout=0.0,
            training=True,
            scale=D**-0.5,
            k_is_latent=True,
        )
    finally:
        smla._CSA_DKV_SLOT[0] = None
    out.backward(do)
    torch.cuda.synchronize()
    assert holder.event is None and holder.keep is None, "the join did not release the dKV side work"
    return [out.detach(), qq.grad, kk.grad, pp.grad, sk.grad]


@pytest.mark.parametrize("contiguous_dpool", [True, False], ids=["dpool_out", "strided_copy"])
def test_csa_dkv_side_stream_dpool(adapter, monkeypatch, contiguous_dpool):
    """The CSA backward with its dKV gathers on a side stream takes the contiguous [P, D] dpool that Turbo writes
    next to dkv (``dpool_out``) instead of copying the pool rows out of dkv (the path of a Turbo without that
    kwarg). Both are byte-equal to the plain backward."""
    from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_v2 import (
        _turbo_gfx1250 as tg,
    )

    torch.manual_seed(0)
    monkeypatch.setenv("PRIMUS_DSA_GFX1250_TURBO", "1")
    monkeypatch.setenv("PRIMUS_V4_BAND_KERNELS", "1")
    if not tg._bwd_takes_dkv_stream():
        pytest.skip("Primus-Turbo without the dkv_stream kwarg")
    if not contiguous_dpool:
        monkeypatch.setattr(tg, "_bwd_takes_dpool_out", lambda: False)
    elif not tg._bwd_takes_dpool_out():
        pytest.skip("Primus-Turbo without the dpool_out kwarg")
    seen = []
    orig = tg._dkv_side_kwargs
    monkeypatch.setattr(tg, "_dkv_side_kwargs", lambda: (lambda kw: seen.append(sorted(kw)) or kw)(orig()))
    data = _inputs(4)
    ref = _run(adapter, 4, data)
    assert seen == [[]], "the plain backward must not get the side-stream kwargs"
    seen.clear()
    got = _run_csa_side_stream(adapter, data, torch.cuda.Stream())
    want = ["dkv_keep", "dkv_stream"] + (["dpool_out"] if contiguous_dpool else [])
    assert seen == [sorted(want)], f"side-stream kwargs {seen}"
    for n, a, b in zip(_names(4), ref, got):
        assert torch.equal(a, b), f"{n}: side-stream backward differs from the plain backward"
