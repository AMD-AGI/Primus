###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``turbo`` DeepSeek-V4 attention fwd+bwd correctness, in the **V4 form** (gfx950),
against the fp32 eager references.

The ``turbo`` backend (:mod:`..._turbo_flydsl`) is the Primus-Turbo native-FlyDSL
sparse-MLA v2 kernel-pair reached through the **turbo API**
(``primus_turbo.flydsl.attention.kernels.sparse_mla_v2``), bound to the V4
autograd adapters (:mod:`v4_csa_attention_turbo_flydsl`). It is the backend
selected by ``use_v4_attention_backend`` / ``use_v4_csa_attention_backend =
"turbo"``. This validates the production V4 invocation paths for all three layer
kinds:

* ``compress_ratio == 0``   (dense / SWA)  -> :func:`v4_attention_turbo`
* ``compress_ratio == 128`` (HCA)          -> :func:`v4_attention_turbo`
* ``compress_ratio == 4``   (CSA)          -> :func:`v4_csa_attention_turbo`

Full fwd (O) and torch-autograd bwd (dQ, dlatent, dpool, dsink) of the bf16 turbo
adapter vs the fp32 eager reference (same tolerances the gluon_v2 backend UT uses;
turbo shares the identical fused single-latent sparse-MLA-with-sink math). GPU-only;
skipped off gfx950 / when primus_turbo's flydsl attention or ``flydsl`` is absent.

FlyDSL fixes the head-block at 64, so ``num_heads`` must be a multiple of 64 here
(the kernel asserts ``num_heads % 32 == 0``; H=64/128 are the production sizes).
``S`` is a multiple of 128 so the cr=128 closed-form pool (``pool_cr = S/P = 128``)
is exact.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("turbo sparse-MLA kernels require CUDA / HIP", allow_module_level=True)

pytest.importorskip("flydsl", reason="flydsl pip package not installed")
# The turbo API: primus_turbo must carry the flydsl sparse-MLA attention submodule.
pytest.importorskip(
    "primus_turbo.flydsl.attention.kernels.sparse_mla_v2",
    reason="installed primus_turbo has no flydsl sparse-MLA attention (turbo backend)",
)

_ARCH = torch.cuda.get_device_properties(0).gcnArchName
if "gfx950" not in _ARCH:
    pytest.skip(f"turbo native-FlyDSL sparse-MLA targets gfx950; got {_ARCH}", allow_module_level=True)

from primus.backends.megatron.core.transformer.sliding_window_kv import (  # noqa: E402
    sliding_window_causal_mask,
)
from primus.backends.megatron.core.transformer.v4_attention_kernels._eager.reference import (  # noqa: E402
    eager_v4_attention,
    eager_v4_csa_attention,
)
from primus.backends.megatron.core.transformer.v4_attention_kernels.v4_csa_attention_turbo_flydsl import (  # noqa: E402
    v4_attention_turbo,
    v4_csa_attention_turbo,
)

D = 512  # V4 head_dim (RoPE baked in-place)
_SWA = 128


def _stats(a, b, *, sig=1e-2):
    a = a.float()
    b = b.float()
    d = (a - b).abs()
    m = b.abs() > sig
    rel = (d[m] / b.abs()[m]) if m.any() else d.new_zeros(0)
    med_rel = rel.median().item() if rel.numel() else 0.0
    av, bv = a.flatten(), b.flatten()
    cos_err = 1.0 - (av @ bv / (av.norm() * bv.norm() + 1e-30)).item()
    return d.max().item(), med_rel, cos_err


def _check(name, a, b, *, abs_tol, sig=1e-2, med=None, cos=None):
    # Guard against a broken kernel returning NaN/Inf (e.g. the cr=4 fast_path
    # overflow / bwd race the extraction found) — surface it as a clear failure.
    assert torch.isfinite(
        a.float()
    ).all(), f"{name} has NaN/Inf ({int((~torch.isfinite(a.float())).sum())} bad)"
    max_abs, med_rel, cos_err = _stats(a, b, sig=sig)
    print(f"    {name:8s} max_abs={max_abs:.3e} median_rel={med_rel:.3e} cos_err={cos_err:.3e}", flush=True)
    assert max_abs < abs_tol, f"{name} max_abs {max_abs:.3e} >= {abs_tol}"
    if med is not None:
        assert med_rel < med, f"{name} median_rel {med_rel:.3e} >= {med}"
    if cos is not None:
        assert cos_err < cos, f"{name} cos_err {cos_err:.3e} >= {cos}"


def _leaf(x, *, fp32):
    y = x.float() if fp32 else x.clone()
    return y.detach().requires_grad_(True)


# H must be a multiple of 64 (FlyDSL head-block); S a multiple of 128 (cr=128 pool_cr).
# B is fixed to 1: the flydsl dense/HCA "banded" path uses a closed-form SWA window
# [i-127..i] over the flat token axis, which crosses batch boundaries for B>1 (it assumes
# a single contiguous sequence). Production / bench_v4_attention.py run B(mbs)=1; multi-batch
# dense/HCA is a known flydsl-banded limitation (CSA/cr=4 is fine for B>1 — it uses the
# per-batch-offset topk). Vary H (64/128), S (256/512), and the sink instead.
_CASES = [
    (1, 64, 256, None),
    (1, 64, 512, 1.0),
    (1, 128, 512, -1.0),
]


@pytest.mark.parametrize("B,H,S,sink_init", _CASES, ids=lambda v: str(v))
def test_v4_turbo_dense_matches_eager(B, H, S, sink_init):
    """cr=0 dense/SWA: turbo fwd+bwd vs eager_v4_attention."""
    g = torch.Generator(device="cuda").manual_seed(0)
    scale = 1.0 / math.sqrt(D)
    q = torch.randn(B, H, S, D, generator=g, device="cuda", dtype=torch.bfloat16)
    lat = torch.randn(B, S, D, generator=g, device="cuda", dtype=torch.bfloat16)
    do = torch.randn(B, H, S, D, generator=g, device="cuda", dtype=torch.bfloat16)
    sink = torch.full((H,), sink_init, dtype=torch.float32, device="cuda") if sink_init is not None else None

    qg, latg = _leaf(q, fp32=False), _leaf(lat, fp32=False)
    sg = _leaf(sink, fp32=False) if sink is not None else None
    kg = latg.unsqueeze(1).expand(B, H, S, D)
    og = v4_attention_turbo(
        qg, kg, kg, sink=sg, swa_window=_SWA, additive_mask=None, attn_dropout=0.0, training=True, scale=scale
    )
    og.backward(do)

    qf, latf = _leaf(q, fp32=True), _leaf(lat, fp32=True)
    sf = _leaf(sink, fp32=True) if sink is not None else None
    kf = latf.unsqueeze(1).expand(B, H, S, D)
    of = eager_v4_attention(
        qf,
        kf,
        kf,
        sink=sf,
        swa_window=_SWA,
        additive_mask=None,
        attn_dropout=0.0,
        training=False,
        scale=scale,
    )
    of.backward(do.float())

    print(f"\n[turbo V4 dense] B={B} H={H} S={S} sink={sink_init}", flush=True)
    assert og.shape == (B, H, S, D) and og.dtype == torch.bfloat16
    _check("O", og, of, abs_tol=3e-2, med=2e-2, cos=1e-3)
    _check("dQ", qg.grad, qf.grad, abs_tol=5e-2, sig=1e-3, med=2e-2, cos=1e-3)
    # abs_tol is generous for the shared-latent grad (summed over H heads -> large-magnitude
    # elements -> bf16 abs outliers); median_rel + cos_err are the real correctness guards.
    _check("dlatent", latg.grad, latf.grad, abs_tol=3e-1, sig=1e-3, med=2e-2, cos=1e-3)
    if sink is not None:
        _check("dSink", sg.grad, sf.grad, abs_tol=5e-1, sig=1e-3, med=5e-2, cos=1e-2)


@pytest.mark.parametrize("B,H,S,sink_init", _CASES, ids=lambda v: str(v))
def test_v4_turbo_hca_matches_eager(B, H, S, sink_init):
    """cr=128 HCA: turbo (local++pool) fwd+bwd vs eager_v4_attention."""
    cr = 128
    P = max(S // cr, 1)
    g = torch.Generator(device="cuda").manual_seed(2)
    scale = 1.0 / math.sqrt(D)
    q = torch.randn(B, H, S, D, generator=g, device="cuda", dtype=torch.bfloat16)
    lat = torch.randn(B, S, D, generator=g, device="cuda", dtype=torch.bfloat16)
    pool = torch.randn(B, P, D, generator=g, device="cuda", dtype=torch.bfloat16)
    do = torch.randn(B, H, S, D, generator=g, device="cuda", dtype=torch.bfloat16)
    sink = torch.full((H,), sink_init, dtype=torch.float32, device="cuda") if sink_init is not None else None

    ti = torch.arange(S, device="cuda").view(S, 1)
    ps = torch.arange(P, device="cuda").view(1, P)
    pool_mask = torch.where(
        ((ps + 1) * cr - 1) <= ti, torch.zeros((), device="cuda"), torch.tensor(float("-inf"), device="cuda")
    ).to(torch.bfloat16)

    qg, latg, poolg = _leaf(q, fp32=False), _leaf(lat, fp32=False), _leaf(pool, fp32=False)
    sg = _leaf(sink, fp32=False) if sink is not None else None
    kg = torch.cat([latg.unsqueeze(1).expand(B, H, S, D), poolg.unsqueeze(1).expand(B, H, P, D)], dim=2)
    og = v4_attention_turbo(
        qg,
        kg,
        kg,
        sink=sg,
        swa_window=_SWA,
        additive_mask=pool_mask,
        attn_dropout=0.0,
        training=True,
        scale=scale,
        hca_local_seqlen=S,
    )
    og.backward(do)

    qf, latf, poolf = _leaf(q, fp32=True), _leaf(lat, fp32=True), _leaf(pool, fp32=True)
    sf = _leaf(sink, fp32=True) if sink is not None else None
    kf = torch.cat([latf.unsqueeze(1).expand(B, H, S, D), poolf.unsqueeze(1).expand(B, H, P, D)], dim=2)
    local_mask = sliding_window_causal_mask(S, _SWA, device="cuda", dtype=torch.float32)
    full_mask = torch.cat([local_mask, pool_mask.float()], dim=1)
    of = eager_v4_attention(
        qf,
        kf,
        kf,
        sink=sf,
        swa_window=0,
        additive_mask=full_mask,
        attn_dropout=0.0,
        training=False,
        scale=scale,
    )
    of.backward(do.float())

    print(f"\n[turbo V4 HCA] B={B} H={H} S={S} P={P} sink={sink_init}", flush=True)
    assert og.shape == (B, H, S, D) and og.dtype == torch.bfloat16
    _check("O", og, of, abs_tol=3e-2, med=2e-2, cos=1e-3)
    _check("dQ", qg.grad, qf.grad, abs_tol=5e-2, sig=1e-3, med=2e-2, cos=1e-3)
    # abs_tol is generous for the shared-latent grad (summed over H heads -> large-magnitude
    # elements -> bf16 abs outliers); median_rel + cos_err are the real correctness guards.
    _check("dlatent", latg.grad, latf.grad, abs_tol=3e-1, sig=1e-3, med=2e-2, cos=1e-3)
    _check("dpool", poolg.grad, poolf.grad, abs_tol=3e-1, sig=1e-3, med=2e-2, cos=1e-3)
    if sink is not None:
        _check("dSink", sg.grad, sf.grad, abs_tol=5e-1, sig=1e-3, med=5e-2, cos=1e-2)


@pytest.mark.parametrize("B,H,S,sink_init", _CASES, ids=lambda v: str(v))
def test_v4_turbo_csa_matches_eager(B, H, S, sink_init):
    """cr=4 CSA: turbo fwd+bwd vs eager_v4_csa_attention."""
    P = max(S // 4, 1)
    K = min(128, P)
    g = torch.Generator(device="cuda").manual_seed(3)
    scale = 1.0 / math.sqrt(D)
    q = torch.randn(B, H, S, D, generator=g, device="cuda", dtype=torch.bfloat16)
    lat = torch.randn(B, S, D, generator=g, device="cuda", dtype=torch.bfloat16)
    pool = torch.randn(B, P, D, generator=g, device="cuda", dtype=torch.bfloat16)
    do = torch.randn(B, H, S, D, generator=g, device="cuda", dtype=torch.bfloat16)
    sink = torch.full((H,), sink_init, dtype=torch.float32, device="cuda") if sink_init is not None else None

    topk_idxs = torch.randint(0, P, (B, S, K), generator=g, device="cuda", dtype=torch.int32)
    drop = torch.rand(B, S, K, generator=g, device="cuda") < 0.125
    topk_idxs = torch.where(drop, torch.full_like(topk_idxs, -1), topk_idxs)
    idx = topk_idxs.clamp(0, P - 1).long()
    bidx = torch.arange(B, device="cuda").view(B, 1, 1)
    sparse_mask_inf = torch.where(
        topk_idxs < 0, torch.tensor(float("-inf"), device="cuda"), torch.zeros((), device="cuda")
    )

    qg, latg, poolg = _leaf(q, fp32=False), _leaf(lat, fp32=False), _leaf(pool, fp32=False)
    sg = _leaf(sink, fp32=False) if sink is not None else None
    klg = latg.unsqueeze(1).expand(B, H, S, D)
    og = v4_csa_attention_turbo(
        qg,
        klg,
        klg,
        poolg,
        topk_idxs=topk_idxs,
        sink=sg,
        swa_window=_SWA,
        attn_dropout=0.0,
        training=True,
        scale=scale,
    )
    og.backward(do)

    qf, latf, poolf = _leaf(q, fp32=True), _leaf(lat, fp32=True), _leaf(pool, fp32=True)
    sf = _leaf(sink, fp32=True) if sink is not None else None
    klf = latf.unsqueeze(1).expand(B, H, S, D)
    gathered = poolf[bidx, idx]
    of = eager_v4_csa_attention(
        qf,
        klf,
        klf,
        gathered,
        sink=sf,
        swa_window=_SWA,
        sparse_mask=sparse_mask_inf.float(),
        attn_dropout=0.0,
        training=False,
        scale=scale,
    )
    of.backward(do.float())

    print(f"\n[turbo V4 CSA] B={B} H={H} S={S} P={P} K={K} sink={sink_init}", flush=True)
    assert og.shape == (B, H, S, D) and og.dtype == torch.bfloat16
    _check("O", og, of, abs_tol=3e-2, med=2e-2, cos=1e-3)
    _check("dQ", qg.grad, qf.grad, abs_tol=5e-2, sig=1e-3, med=2e-2, cos=1e-3)
    # abs_tol is generous for the shared-latent grad (summed over H heads -> large-magnitude
    # elements -> bf16 abs outliers); median_rel + cos_err are the real correctness guards.
    _check("dlatent", latg.grad, latf.grad, abs_tol=3e-1, sig=1e-3, med=2e-2, cos=1e-3)
    _check("dpool", poolg.grad, poolf.grad, abs_tol=3e-1, sig=1e-3, med=2e-2, cos=1e-3)
    if sink is not None:
        _check("dSink", sg.grad, sf.grad, abs_tol=5e-1, sig=1e-3, med=5e-2, cos=1e-2)
