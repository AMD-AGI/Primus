###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Gluon DeepSeek-V4 attention fwd+bwd correctness, in the **V4 form** (gfx950).

Unlike a raw sparse-MLA test (separate 64-rope, random absolute-token topk),
this validates the *production* V4 invocation paths — the autograd adapters in
``v4_csa_attention_gluon`` that DeepseekV4Attention actually dispatches to —
against the eager V4 references for all three layer kinds:

* ``compress_ratio == 0``   (dense / SWA)  -> :func:`v4_attention_gluon`
* ``compress_ratio == 128`` (HCA)          -> :func:`v4_attention_gluon`
* ``compress_ratio == 4``   (CSA)          -> :func:`v4_csa_attention_gluon_from_pool`

The V4 layout has ``head_dim = 512`` with RoPE applied *in-place* (K = V = 512,
score over 512); the adapter feeds the 512 latent as the gluon "lora" with a
zero rope pad and builds ``kv = [local ++ pool]`` / ``topk = [SWA window ++
pool]`` (padded to a multiple of 64 so the gluon bwd dKV tiling is valid — HCA
160 -> 192). We compare full fwd (O) and torch-autograd bwd (dQ, dlatent, dpool,
dsink) of the bf16 gluon adapter against the fp32 eager reference.

GPU-only; skipped off gfx950 / when Gluon is unavailable. ``B = 2`` to exercise
the per-batch token-flattening + topk-offset logic.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("Gluon DSA kernels require CUDA / HIP", allow_module_level=True)

# Gluon is an experimental Triton submodule; skip cleanly if absent.
pytest.importorskip("triton", reason="Triton not installed")
try:
    from triton.experimental import gluon  # noqa: F401
except Exception:  # pragma: no cover - environment-dependent
    pytest.skip("triton.experimental.gluon unavailable", allow_module_level=True)

_ARCH = torch.cuda.get_device_properties(0).gcnArchName
if "gfx950" not in _ARCH:
    pytest.skip(f"ported Gluon DSA kernels target gfx950; got {_ARCH}", allow_module_level=True)

from primus.backends.megatron.core.transformer.sliding_window_kv import (  # noqa: E402
    sliding_window_causal_mask,
)
from primus.backends.megatron.core.transformer.v4_attention_kernels.reference import (  # noqa: E402
    eager_v4_attention,
    eager_v4_csa_attention,
)
from primus.backends.megatron.core.transformer.v4_attention_kernels.v4_csa_attention_gluon import (  # noqa: E402
    v4_attention_gluon,
    v4_csa_attention_gluon_from_pool,
)

D = 512  # V4 head_dim (RoPE baked in-place)
_SWA = 128


# ---------------------------------------------------------------------------
# Comparison helpers (bf16 kernel vs fp32 eager autograd)
# ---------------------------------------------------------------------------


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
    max_abs, med_rel, cos_err = _stats(a, b, sig=sig)
    print(f"    {name:8s} max_abs={max_abs:.3e} median_rel={med_rel:.3e} cos_err={cos_err:.3e}", flush=True)
    assert max_abs < abs_tol, f"{name} max_abs {max_abs:.3e} >= {abs_tol}"
    if med is not None:
        assert med_rel < med, f"{name} median_rel {med_rel:.3e} >= {med}"
    if cos is not None:
        assert cos_err < cos, f"{name} cos_err {cos_err:.3e} >= {cos}"


def _leaf(x, *, fp32):
    """Detached leaf clone (fp32 for the eager ref, bf16 for the gluon adapter)."""
    y = x.float() if fp32 else x.clone()
    return y.detach().requires_grad_(True)


# ---------------------------------------------------------------------------
# (B, H, S, sink_init)
# ---------------------------------------------------------------------------
_CASES = [
    (2, 16, 256, None),
    (2, 16, 256, 1.0),
    (1, 32, 512, -1.0),
]


@pytest.mark.parametrize("B,H,S,sink_init", _CASES, ids=lambda v: str(v))
def test_v4_gluon_dense_matches_eager(B, H, S, sink_init):
    """cr=0 dense/SWA: v4_attention_gluon fwd+bwd vs eager_v4_attention."""
    g = torch.Generator(device="cuda").manual_seed(0)
    scale = 1.0 / math.sqrt(D)
    q = torch.randn(B, H, S, D, generator=g, device="cuda", dtype=torch.bfloat16)
    lat = torch.randn(B, S, D, generator=g, device="cuda", dtype=torch.bfloat16)
    do = torch.randn(B, H, S, D, generator=g, device="cuda", dtype=torch.bfloat16)
    sink = torch.full((H,), sink_init, dtype=torch.float32, device="cuda") if sink_init is not None else None

    qg, latg = _leaf(q, fp32=False), _leaf(lat, fp32=False)
    sg = _leaf(sink, fp32=False) if sink is not None else None
    kg = latg.unsqueeze(1).expand(B, H, S, D)
    og = v4_attention_gluon(
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

    print(f"\n[gluon V4 dense] B={B} H={H} S={S} sink={sink_init}", flush=True)
    assert og.shape == (B, H, S, D) and og.dtype == torch.bfloat16
    _check("O", og, of, abs_tol=3e-2, med=2e-2, cos=1e-3)
    _check("dQ", qg.grad, qf.grad, abs_tol=5e-2, sig=1e-3, med=2e-2, cos=1e-3)
    _check("dlatent", latg.grad, latf.grad, abs_tol=1e-1, sig=1e-3, med=2e-2, cos=1e-3)
    if sink is not None:
        _check("dSink", sg.grad, sf.grad, abs_tol=5e-1, sig=1e-3, med=5e-2, cos=1e-2)


@pytest.mark.parametrize("B,H,S,sink_init", _CASES, ids=lambda v: str(v))
def test_v4_gluon_hca_matches_eager(B, H, S, sink_init):
    """cr=128 HCA: v4_attention_gluon (local++pool) fwd+bwd vs eager_v4_attention."""
    cr = 128
    P = max(S // cr, 1)
    g = torch.Generator(device="cuda").manual_seed(2)
    scale = 1.0 / math.sqrt(D)
    q = torch.randn(B, H, S, D, generator=g, device="cuda", dtype=torch.bfloat16)
    lat = torch.randn(B, S, D, generator=g, device="cuda", dtype=torch.bfloat16)
    pool = torch.randn(B, P, D, generator=g, device="cuda", dtype=torch.bfloat16)
    do = torch.randn(B, H, S, D, generator=g, device="cuda", dtype=torch.bfloat16)
    sink = torch.full((H,), sink_init, dtype=torch.float32, device="cuda") if sink_init is not None else None

    # HCA causal pool mask [S, P]: pool slot p visible to query s iff (p+1)*cr-1 <= s.
    ti = torch.arange(S, device="cuda").view(S, 1)
    ps = torch.arange(P, device="cuda").view(1, P)
    pool_mask = torch.where(
        ((ps + 1) * cr - 1) <= ti, torch.zeros((), device="cuda"), torch.tensor(float("-inf"), device="cuda")
    ).to(torch.bfloat16)

    qg, latg, poolg = _leaf(q, fp32=False), _leaf(lat, fp32=False), _leaf(pool, fp32=False)
    sg = _leaf(sink, fp32=False) if sink is not None else None
    kg = torch.cat([latg.unsqueeze(1).expand(B, H, S, D), poolg.unsqueeze(1).expand(B, H, P, D)], dim=2)
    og = v4_attention_gluon(
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
    full_mask = torch.cat([local_mask, pool_mask.float()], dim=1)  # [S, S+P]
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

    print(f"\n[gluon V4 HCA] B={B} H={H} S={S} P={P} sink={sink_init}", flush=True)
    assert og.shape == (B, H, S, D) and og.dtype == torch.bfloat16
    _check("O", og, of, abs_tol=3e-2, med=2e-2, cos=1e-3)
    _check("dQ", qg.grad, qf.grad, abs_tol=5e-2, sig=1e-3, med=2e-2, cos=1e-3)
    _check("dlatent", latg.grad, latf.grad, abs_tol=1e-1, sig=1e-3, med=2e-2, cos=1e-3)
    _check("dpool", poolg.grad, poolf.grad, abs_tol=1e-1, sig=1e-3, med=2e-2, cos=1e-3)
    if sink is not None:
        _check("dSink", sg.grad, sf.grad, abs_tol=5e-1, sig=1e-3, med=5e-2, cos=1e-2)


@pytest.mark.parametrize("B,H,S,sink_init", _CASES, ids=lambda v: str(v))
def test_v4_gluon_csa_matches_eager(B, H, S, sink_init):
    """cr=4 CSA: v4_csa_attention_gluon_from_pool fwd+bwd vs eager_v4_csa_attention."""
    P = max(S // 4, 1)
    K = min(128, P)
    g = torch.Generator(device="cuda").manual_seed(3)
    scale = 1.0 / math.sqrt(D)
    q = torch.randn(B, H, S, D, generator=g, device="cuda", dtype=torch.bfloat16)
    lat = torch.randn(B, S, D, generator=g, device="cuda", dtype=torch.bfloat16)
    pool = torch.randn(B, P, D, generator=g, device="cuda", dtype=torch.bfloat16)
    do = torch.randn(B, H, S, D, generator=g, device="cuda", dtype=torch.bfloat16)
    sink = torch.full((H,), sink_init, dtype=torch.float32, device="cuda") if sink_init is not None else None

    # Per-query top-K pool indices in [0, P), with ~1/8 invalid (-1).
    topk_idxs = torch.randint(0, P, (B, S, K), generator=g, device="cuda", dtype=torch.int32)
    drop = torch.rand(B, S, K, generator=g, device="cuda") < 0.125
    topk_idxs = torch.where(drop, torch.full_like(topk_idxs, -1), topk_idxs)
    idx = topk_idxs.clamp(0, P - 1).long()
    bidx = torch.arange(B, device="cuda").view(B, 1, 1)
    sparse_mask_inf = torch.where(
        topk_idxs < 0, torch.tensor(float("-inf"), device="cuda"), torch.zeros((), device="cuda")
    )  # [B, S, K]

    qg, latg, poolg = _leaf(q, fp32=False), _leaf(lat, fp32=False), _leaf(pool, fp32=False)
    sg = _leaf(sink, fp32=False) if sink is not None else None
    klg = latg.unsqueeze(1).expand(B, H, S, D)
    og = v4_csa_attention_gluon_from_pool(
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
    gathered = poolf[bidx, idx]  # [B, S, K, D] (differentiable gather into poolf)
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

    print(f"\n[gluon V4 CSA] B={B} H={H} S={S} P={P} K={K} sink={sink_init}", flush=True)
    assert og.shape == (B, H, S, D) and og.dtype == torch.bfloat16
    _check("O", og, of, abs_tol=3e-2, med=2e-2, cos=1e-3)
    _check("dQ", qg.grad, qf.grad, abs_tol=5e-2, sig=1e-3, med=2e-2, cos=1e-3)
    _check("dlatent", latg.grad, latf.grad, abs_tol=1e-1, sig=1e-3, med=2e-2, cos=1e-3)
    _check("dpool", poolg.grad, poolf.grad, abs_tol=2e-1, sig=1e-3, med=2e-2, cos=1e-3)
    if sink is not None:
        _check("dSink", sg.grad, sf.grad, abs_tol=5e-1, sig=1e-3, med=5e-2, cos=1e-2)
