###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""gluon_v3 DeepSeek-V4 attention fwd+bwd correctness against fp32 eager refs."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("gluon_v3 kernels require CUDA / HIP", allow_module_level=True)

pytest.importorskip("triton", reason="Triton not installed")
try:
    from triton.experimental import gluon  # noqa: F401
except Exception:  # pragma: no cover - environment-dependent
    pytest.skip("triton.experimental.gluon unavailable", allow_module_level=True)

_ARCH = torch.cuda.get_device_properties(0).gcnArchName
if "gfx950" not in _ARCH:
    pytest.skip(f"gluon_v3 targets gfx950; got {_ARCH}", allow_module_level=True)

from primus.backends.megatron.core.transformer.sliding_window_kv import (  # noqa: E402
    sliding_window_causal_mask,
)
from primus.backends.megatron.core.transformer.v4_attention_kernels._eager.reference import (  # noqa: E402
    eager_v4_attention,
    eager_v4_csa_attention,
)

try:
    from primus.backends.megatron.core.transformer.v4_attention_kernels._gluon_v3 import (  # noqa: E402
        sparse_mla_bwd_v4_gluon_v3,
        sparse_mla_fwd_v4_gluon_v3,
    )
    from primus.backends.megatron.core.transformer.v4_attention_kernels.v4_csa_attention_gluon_v3 import (  # noqa: E402
        v4_attention_gluon_v3,
        v4_csa_attention_gluon_v3,
    )
except ImportError as exc:  # pragma: no cover - environment-dependent
    pytest.skip(f"gluon_v3 unavailable: {exc}", allow_module_level=True)

D = 512
_ROPE = 64
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


def _eager_sparse_mla(q, kv, topk, sink, *, scale):
    q_lora = q[:, :, :D]
    kv_lora = kv[:, 0, :D]
    safe = topk.clamp(min=0).long()
    gathered = kv_lora[safe]
    valid = topk >= 0
    scores = torch.einsum("thd,tkd->thk", q_lora, gathered) * scale
    scores = torch.where(valid[:, None, :], scores, torch.full_like(scores, float("-inf")))
    if sink is not None:
        sink_scores = sink.view(1, -1, 1).expand(q.shape[0], q.shape[1], 1)
        all_scores = torch.cat([scores, sink_scores], dim=-1)
    else:
        all_scores = scores
    probs = torch.softmax(all_scores, dim=-1)
    out = torch.einsum("thk,tkd->thd", probs[:, :, : topk.shape[1]], gathered)
    lse = torch.logsumexp(all_scores, dim=-1)
    return out, lse


_CASES = [
    (1, 16, 128, None),
    (1, 32, 256, 1.0),
]


@pytest.mark.parametrize("B,H,S,sink_init", _CASES, ids=lambda v: str(v))
def test_v4_gluon_v3_dense_matches_eager(B, H, S, sink_init):
    g = torch.Generator(device="cuda").manual_seed(10)
    scale = 1.0 / math.sqrt(D)
    q = torch.randn(B, H, S, D, generator=g, device="cuda", dtype=torch.bfloat16)
    lat = torch.randn(B, S, D, generator=g, device="cuda", dtype=torch.bfloat16)
    do = torch.randn(B, H, S, D, generator=g, device="cuda", dtype=torch.bfloat16)
    sink = torch.full((H,), sink_init, dtype=torch.float32, device="cuda") if sink_init is not None else None

    qg, latg = _leaf(q, fp32=False), _leaf(lat, fp32=False)
    sg = _leaf(sink, fp32=False) if sink is not None else None
    kg = latg.unsqueeze(1).expand(B, H, S, D)
    og = v4_attention_gluon_v3(
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

    print(f"\n[gluon_v3 V4 dense] B={B} H={H} S={S} sink={sink_init}", flush=True)
    _check("O", og, of, abs_tol=3e-2, med=2e-2, cos=1e-3)
    _check("dQ", qg.grad, qf.grad, abs_tol=5e-2, sig=1e-3, med=2e-2, cos=1e-3)
    _check("dlatent", latg.grad, latf.grad, abs_tol=1e-1, sig=1e-3, med=2e-2, cos=1e-3)
    if sink is not None:
        _check("dSink", sg.grad, sf.grad, abs_tol=5e-1, sig=1e-3, med=5e-2, cos=1e-2)


def test_v4_gluon_v3_hca_matches_eager():
    B, H, S, cr = 1, 32, 256, 128
    P = max(S // cr, 1)
    g = torch.Generator(device="cuda").manual_seed(11)
    scale = 1.0 / math.sqrt(D)
    q = torch.randn(B, H, S, D, generator=g, device="cuda", dtype=torch.bfloat16)
    lat = torch.randn(B, S, D, generator=g, device="cuda", dtype=torch.bfloat16)
    pool = torch.randn(B, P, D, generator=g, device="cuda", dtype=torch.bfloat16)
    do = torch.randn(B, H, S, D, generator=g, device="cuda", dtype=torch.bfloat16)
    sink = torch.full((H,), 1.0, dtype=torch.float32, device="cuda")

    ti = torch.arange(S, device="cuda").view(S, 1)
    ps = torch.arange(P, device="cuda").view(1, P)
    pool_mask = torch.where(
        ((ps + 1) * cr - 1) <= ti, torch.zeros((), device="cuda"), torch.tensor(float("-inf"), device="cuda")
    ).to(torch.bfloat16)

    qg, latg, poolg, sg = (
        _leaf(q, fp32=False),
        _leaf(lat, fp32=False),
        _leaf(pool, fp32=False),
        _leaf(sink, fp32=False),
    )
    kg = torch.cat([latg.unsqueeze(1).expand(B, H, S, D), poolg.unsqueeze(1).expand(B, H, P, D)], dim=2)
    og = v4_attention_gluon_v3(
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

    qf, latf, poolf, sf = (
        _leaf(q, fp32=True),
        _leaf(lat, fp32=True),
        _leaf(pool, fp32=True),
        _leaf(sink, fp32=True),
    )
    kf = torch.cat([latf.unsqueeze(1).expand(B, H, S, D), poolf.unsqueeze(1).expand(B, H, P, D)], dim=2)
    full_mask = torch.cat(
        [sliding_window_causal_mask(S, _SWA, device="cuda", dtype=torch.float32), pool_mask.float()], dim=1
    )
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

    print(f"\n[gluon_v3 V4 HCA] B={B} H={H} S={S} P={P}", flush=True)
    _check("O", og, of, abs_tol=3e-2, med=2e-2, cos=1e-3)
    _check("dQ", qg.grad, qf.grad, abs_tol=5e-2, sig=1e-3, med=2e-2, cos=1e-3)
    _check("dlatent", latg.grad, latf.grad, abs_tol=1e-1, sig=1e-3, med=2e-2, cos=1e-3)
    _check("dpool", poolg.grad, poolf.grad, abs_tol=1e-1, sig=1e-3, med=2e-2, cos=1e-3)
    _check("dSink", sg.grad, sf.grad, abs_tol=5e-1, sig=1e-3, med=5e-2, cos=1e-2)


def test_v4_gluon_v3_csa_adapter_matches_eager():
    B, H, S = 1, 32, 256
    P, K = max(S // 4, 1), 64
    g = torch.Generator(device="cuda").manual_seed(12)
    scale = 1.0 / math.sqrt(D)
    q = torch.randn(B, H, S, D, generator=g, device="cuda", dtype=torch.bfloat16)
    lat = torch.randn(B, S, D, generator=g, device="cuda", dtype=torch.bfloat16)
    pool = torch.randn(B, P, D, generator=g, device="cuda", dtype=torch.bfloat16)
    do = torch.randn(B, H, S, D, generator=g, device="cuda", dtype=torch.bfloat16)
    sink = torch.full((H,), -1.0, dtype=torch.float32, device="cuda")
    topk_idxs = torch.randint(0, P, (B, S, K), generator=g, device="cuda", dtype=torch.int32)
    topk_idxs = torch.where(
        torch.rand(B, S, K, generator=g, device="cuda") < 0.125, torch.full_like(topk_idxs, -1), topk_idxs
    )
    idx = topk_idxs.clamp(0, P - 1).long()
    bidx = torch.arange(B, device="cuda").view(B, 1, 1)
    sparse_mask_inf = torch.where(
        topk_idxs < 0, torch.tensor(float("-inf"), device="cuda"), torch.zeros((), device="cuda")
    )

    qg, latg, poolg, sg = (
        _leaf(q, fp32=False),
        _leaf(lat, fp32=False),
        _leaf(pool, fp32=False),
        _leaf(sink, fp32=False),
    )
    klg = latg.unsqueeze(1).expand(B, H, S, D)
    og = v4_csa_attention_gluon_v3(
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

    qf, latf, poolf, sf = (
        _leaf(q, fp32=True),
        _leaf(lat, fp32=True),
        _leaf(pool, fp32=True),
        _leaf(sink, fp32=True),
    )
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

    print(f"\n[gluon_v3 V4 CSA adapter] B={B} H={H} S={S} P={P} K={K}", flush=True)
    _check("O", og, of, abs_tol=3e-2, med=2e-2, cos=1e-3)
    _check("dQ", qg.grad, qf.grad, abs_tol=5e-2, sig=1e-3, med=2e-2, cos=1e-3)
    _check("dlatent", latg.grad, latf.grad, abs_tol=1e-1, sig=1e-3, med=2e-2, cos=1e-3)
    _check("dpool", poolg.grad, poolf.grad, abs_tol=2e-1, sig=1e-3, med=2e-2, cos=1e-3)
    _check("dSink", sg.grad, sf.grad, abs_tol=5e-1, sig=1e-3, med=5e-2, cos=1e-2)


@pytest.mark.parametrize("H,pool_k,T", [(64, 512, 64), (128, 1024, 64)], ids=["h64_topk640", "h128_topk1152"])
def test_v4_gluon_v3_round9_csa_formula_path_matches_eager(H, pool_k, T):
    g = torch.Generator(device="cuda").manual_seed(13 + H)
    scale = 1.0 / math.sqrt(D)
    q512 = torch.randn(T, H, D, generator=g, device="cuda", dtype=torch.bfloat16)
    kv512 = torch.randn(T + pool_k, 1, D, generator=g, device="cuda", dtype=torch.bfloat16)
    q = torch.cat([q512, torch.zeros(T, H, _ROPE, device="cuda", dtype=torch.bfloat16)], dim=-1).contiguous()
    kv = torch.cat(
        [kv512, torch.zeros(T + pool_k, 1, _ROPE, device="cuda", dtype=torch.bfloat16)], dim=-1
    ).contiguous()
    do = torch.randn(T, H, D, generator=g, device="cuda", dtype=torch.bfloat16)
    sink = torch.randn(H, generator=g, device="cuda", dtype=torch.float32) * 0.1

    ti = torch.arange(T, device="cuda").view(T, 1)
    win = ti - _SWA + 1 + torch.arange(_SWA, device="cuda").view(1, _SWA)
    win = torch.where(win >= 0, win, torch.full_like(win, -1))
    pool_topk = T + torch.randint(0, pool_k, (T, pool_k), generator=g, device="cuda", dtype=torch.int64)
    topk = torch.cat([win, pool_topk], dim=1).to(torch.int32).contiguous()

    qg, kvg, sg = _leaf(q, fp32=False), _leaf(kv, fp32=False), _leaf(sink, fp32=False)
    og, lseg = sparse_mla_fwd_v4_gluon_v3(qg, kvg, topk, attn_sink=sg, kv_lora_rank=D, scale=scale)
    dqg, dkvg, dsg = sparse_mla_bwd_v4_gluon_v3(
        qg, kvg, og, do, topk, lseg, attn_sink=sg, kv_lora_rank=D, scale=scale
    )

    qf, kvf, sf = _leaf(q, fp32=True), _leaf(kv, fp32=True), _leaf(sink, fp32=True)
    of, lsef = _eager_sparse_mla(qf, kvf, topk, sf, scale=scale)
    of.backward(do.float())

    print(f"\n[gluon_v3 round9 CSA formula] T={T} H={H} TOPK={topk.shape[1]}", flush=True)
    _check("O", og, of, abs_tol=5e-2, med=3e-2, cos=2e-3)
    _check("LSE", lseg, lsef, abs_tol=1e-2, sig=1e-3, med=1e-3, cos=1e-6)
    _check("dQ", dqg[:, :, :D], qf.grad[:, :, :D], abs_tol=1e-1, sig=1e-3, med=3e-2, cos=2e-3)
    _check("dKV", dkvg[:, :, :D], kvf.grad[:, :, :D], abs_tol=3e-1, sig=1e-3, med=5e-2, cos=5e-3)
    _check("dSink", dsg, sf.grad, abs_tol=1.0, sig=1e-3, med=8e-2, cos=2e-2)
