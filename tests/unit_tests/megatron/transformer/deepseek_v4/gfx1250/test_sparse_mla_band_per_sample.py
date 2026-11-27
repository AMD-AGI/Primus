###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""PRIMUS_V4_BAND_PER_SAMPLE: a flattened B > 1 batch runs the closed-form band kernels one sample at a time.

For the three layer kinds (compress ratio 0 = SWA, 4 = CSA, 128 = HCA), forward and backward through the adapter:

* B = 2 with the per-sample path (on by default) gives, per sample, the BITS of two separate B = 1 calls (out, dq, dkv / dk, dpool);
  the sink gradient (taken once over the batch) matches the sum of the B = 1 sink gradients within fp32 rounding;
* it agrees with the flattened index-driven B = 2 path (PRIMUS_V4_BAND_PER_SAMPLE=0) within bf16 tolerance;
* the CSA backward with its dKV work on a side stream (PRIMUS_OVERLAP CSA_BWD) gives the same bits;
* the sample-local CSA map (``csa_topk(local=True)``) is the flattened map minus each sample's kv base.
"""

import pytest
import torch

from .helpers import requires_turbo_flydsl
from .test_sparse_mla_adapter import D, DEV, H, S, TOPK, W, _inputs, _names, _rel, _run
from .test_sparse_mla_adapter import adapter  # noqa: F401  (fixture)
from .test_sparse_mla_adapter import _run_csa_side_stream

pytestmark = requires_turbo_flydsl

B = 2


def _cat_inputs(cr):
    """B = 2 inputs made of two B = 1 inputs (seeds 0 and 1), plus the two B = 1 inputs."""
    d0, d1 = _inputs(cr, seed=0), _inputs(cr, seed=1)
    q = torch.cat([d0[0], d1[0]], 0)
    k = torch.cat([d0[1], d1[1]], 0)
    pool = torch.cat([d0[2], d1[2]], 0)
    sink = d0[3]
    do = torch.cat([d0[4], d1[4]], 0)
    topk = torch.cat([d0[5], d1[5]], 0) if cr == 4 else None
    mask = d0[6]
    d1 = (d1[0], d1[1], d1[2], sink, d1[4], d1[5], d1[6])
    return (q, k, pool, sink, do, topk, mask), d0, d1


def _setenv(mp, ps):
    mp.setenv("PRIMUS_DSA_GFX1250_TURBO", "1")
    mp.setenv("PRIMUS_V4_BAND_KERNELS", "1")
    if ps:  # the default
        mp.delenv("PRIMUS_V4_BAND_PER_SAMPLE", raising=False)
    else:
        mp.setenv("PRIMUS_V4_BAND_PER_SAMPLE", "0")
    mp.delenv("PRIMUS_LAUNCH_SINK_CAST", raising=False)
    mp.delenv("PRIMUS_LAUNCH_DSINK", raising=False)


def _ps_available():
    from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_v2 import _turbo_gfx1250 as tg

    return tg._takes_out_kwargs()


@pytest.mark.parametrize("cr", [0, 4, 128])
def test_per_sample_bits_equal_b1(adapter, monkeypatch, cr):  # noqa: F811
    if not _ps_available():
        pytest.skip("Primus-Turbo without the out= kwargs")
    _setenv(monkeypatch, ps=True)
    data, d0, d1 = _cat_inputs(cr)
    calls = []
    from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_v2 import _turbo_gfx1250 as tg

    for name in ("sparse_mla_fwd_turbo_gfx1250_ps", "sparse_mla_bwd_turbo_gfx1250_ps"):
        orig = getattr(tg, name)
        monkeypatch.setattr(tg, name, (lambda o, n: lambda *a, **k: (calls.append(n), o(*a, **k))[1])(orig, name))
    # the fwd / bwd entry points hold their own reference: patch the attributes the adapter reads
    from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_v2 import (
        dsa_bwd_v4_triton,
        dsa_fwd_v4_triton,
    )

    monkeypatch.setattr(dsa_fwd_v4_triton.sparse_mla_fwd_v4_triton, "per_sample", tg.sparse_mla_fwd_turbo_gfx1250_ps)
    monkeypatch.setattr(dsa_bwd_v4_triton.sparse_mla_bwd_v4_triton, "per_sample", tg.sparse_mla_bwd_turbo_gfx1250_ps)

    got = _run(adapter, cr, data)
    want_bwd = ["sparse_mla_bwd_turbo_gfx1250_ps"]
    want = want_bwd if cr == 4 else ["sparse_mla_fwd_turbo_gfx1250_ps"] + want_bwd
    assert calls == want, f"per-sample path not taken: {calls}"
    r0, r1 = _run(adapter, cr, d0), _run(adapter, cr, d1)
    names = _names(cr)
    for i, n in enumerate(names[:-1]):  # all but dsink: per-sample bits
        ref = torch.cat([r0[i], r1[i]], 0)
        assert torch.equal(got[i], ref), f"{n}: per-sample B=2 differs from two B=1 calls"
    ds = r0[-1].float() + r1[-1].float()
    assert _rel(got[-1], ds) < 1e-5, f"dsink: rel {_rel(got[-1], ds):.3e}"


@pytest.mark.parametrize("cr", [0, 4, 128])
def test_per_sample_matches_flattened_index_driven(adapter, monkeypatch, cr):  # noqa: F811
    if not _ps_available():
        pytest.skip("Primus-Turbo without the out= kwargs")
    data, _, _ = _cat_inputs(cr)
    _setenv(monkeypatch, ps=False)
    ref = _run(adapter, cr, data)
    _setenv(monkeypatch, ps=True)
    got = _run(adapter, cr, data)
    got2 = _run(adapter, cr, data)
    for n, a, b, c in zip(_names(cr), got, ref, got2):
        assert torch.isfinite(a.float()).all(), n
        assert torch.equal(a, c), f"{n}: per-sample path not run-to-run deterministic"
        assert _rel(a, b) < 2e-2, f"{n}: per-sample vs flattened index-driven rel L2 {_rel(a, b):.3e}"


def test_per_sample_csa_side_stream_bits(adapter, monkeypatch):  # noqa: F811
    from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_v2 import _turbo_gfx1250 as tg

    if not (_ps_available() and tg._bwd_takes_dkv_stream()):
        pytest.skip("Primus-Turbo without the out= / dkv_stream kwargs")
    _setenv(monkeypatch, ps=True)
    data, _, _ = _cat_inputs(4)
    ref = _run(adapter, 4, data)
    got = _run_csa_side_stream(adapter, data, torch.cuda.Stream())
    for n, a, b in zip(_names(4), ref, got):
        assert torch.equal(a, b), f"{n}: side-stream per-sample backward differs"


def test_csa_topk_local_map():
    from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common.sparse_mla_glue import csa_topk

    P = S // 4
    g = torch.Generator(device=DEV).manual_seed(0)
    idx = torch.randint(-1, P, (B, S, TOPK), device=DEV, generator=g, dtype=torch.int32)
    glob = csa_topk(idx, S, S, P, W)
    loc = csa_topk(idx, S, S, P, W, local=True)
    base = (torch.arange(B * S, device=DEV, dtype=torch.int32) // S * (S + P)).view(-1, 1)
    assert torch.equal(torch.where(glob >= 0, glob - base, glob), loc)
    assert torch.equal(loc[:S], glob[:S])
