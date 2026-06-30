###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Gluon DeepSeek-V4 sparse-MLA attention fwd / bwd correctness (gfx950).

Validates the ported Gluon backend (``_gluon_dsa``, from ROCm/aiter PR #2922)
against an eager-Python sparse-MLA reference:

* forward: gluon ``(O, LSE)`` vs an fp32 reference that gathers the MQA latent
  ``kv`` by ``topk_indices`` and runs a sink-augmented softmax;
* backward: gluon ``(dq, dkv, d_sink)`` vs torch autograd of the same fp32
  reference forward.

Interface (sparse-MLA latent, NOT the CSA gathered/sparse_mask layout):

* ``q``  : ``[T, H, d_qk=576]`` bf16   (``d_qk = kv_lora_rank 512 + rope 64``)
* ``kv`` : ``[T, 1, d_qk]``     bf16   (single MQA latent; ``V = K_lora[:512]``)
* ``topk`` : ``[T, TOPK]`` int32       (absolute token indices; ``-1`` invalid)
* ``sink`` : ``[H]`` fp32 optional

Tolerances mirror aiter's own ``test_sparse_mla_v4_bwd_gluon`` (bf16 kernel vs
fp32 autograd). GPU-only; skipped off gfx950 / when Gluon is unavailable.
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

from primus.backends.megatron.core.transformer.v4_attention_kernels._gluon_dsa import (  # noqa: E402
    sparse_mla_bwd_v4_gluon,
    sparse_mla_fwd_v4_gluon,
)

D_V, D_ROPE = 512, 64
D_QK = D_V + D_ROPE


# ---------------------------------------------------------------------------
# Eager fp32 sparse-MLA reference (bit-for-bit aiter's test reference)
# ---------------------------------------------------------------------------


def _ref_fwd(q, kv, topk, sink, scale):
    T, H, _ = q.shape
    inv = (topk < 0) | (topk >= T)
    idx = topk.clamp(0, T - 1).long()
    gk = kv.squeeze(1)[idx]  # [T, K, d_qk]
    S = torch.einsum("thd,tkd->thk", q, gk) * scale
    S = S.masked_fill(inv[:, None, :], float("-inf"))
    if sink is not None:
        sc = sink.view(1, H, 1).expand(T, H, 1)
        lse = torch.logsumexp(torch.cat([S, sc], dim=-1), dim=-1)
    else:
        lse = torch.logsumexp(S, dim=-1)
    P = torch.exp(S - lse[:, :, None])
    P = torch.where(inv[:, None, :], torch.zeros_like(P), P)
    o = torch.einsum("thk,tkd->thd", P, gk[..., :D_V])
    return o, lse


def _ref_grads(q, kv, topk, sink, do, scale):
    qf = q.detach().float().requires_grad_(True)
    kf = kv.detach().float().requires_grad_(True)
    sf = sink.detach().float().requires_grad_(True) if sink is not None else None
    o, _ = _ref_fwd(qf, kf, topk, sf, scale)
    (o * do.float()).sum().backward()
    return qf.grad, kf.grad, (sf.grad if sf is not None else None)


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
    print(f"    {name:4s} max_abs={max_abs:.3e} median_rel={med_rel:.3e} cos_err={cos_err:.3e}", flush=True)
    assert max_abs < abs_tol, f"{name} max_abs {max_abs:.3e} >= {abs_tol}"
    if med is not None:
        assert med_rel < med, f"{name} median_rel {med_rel:.3e} >= {med}"
    if cos is not None:
        assert cos_err < cos, f"{name} cos_err {cos_err:.3e} >= {cos}"


# (S, H, TOPK, sink_init)
_CASES = [
    (256, 64, 128, None),
    (256, 128, 256, 1.0),
    (512, 128, 640, -1.0),
]


def _make(S, H, TOPK, sink_init, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(S, H, D_QK, generator=g, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(S, 1, D_QK, generator=g, device="cuda", dtype=torch.bfloat16)
    topk = torch.randint(0, S, (S, TOPK), generator=g, device="cuda", dtype=torch.int32)
    sink = torch.full((H,), sink_init, dtype=torch.float32, device="cuda") if sink_init is not None else None
    return q, kv, topk, sink


@pytest.mark.parametrize("S,H,TOPK,sink_init", _CASES, ids=lambda v: str(v))
def test_gluon_dsa_fwd_matches_reference(S, H, TOPK, sink_init):
    """Gluon sparse-MLA forward O / LSE vs fp32 eager reference."""
    q, kv, topk, sink = _make(S, H, TOPK, sink_init)
    scale = 1.0 / math.sqrt(D_QK)

    o_ref, lse_ref = _ref_fwd(q.float(), kv.float(), topk, sink, scale)
    o_g, lse_g = sparse_mla_fwd_v4_gluon(q, kv, topk, attn_sink=sink, kv_lora_rank=D_V, scale=scale)

    print(f"\n[gluon DSA fwd] S={S} H={H} TOPK={TOPK} sink={sink_init}", flush=True)
    assert o_g.shape == (S, H, D_V)
    assert o_g.dtype == torch.bfloat16
    _check("O", o_g, o_ref, abs_tol=3e-2, sig=1e-2, med=2e-2, cos=1e-3)
    _check("LSE", lse_g, lse_ref, abs_tol=5e-3, sig=1e-2, med=1e-3, cos=1e-5)


@pytest.mark.parametrize("S,H,TOPK,sink_init", _CASES, ids=lambda v: str(v))
def test_gluon_dsa_bwd_matches_autograd(S, H, TOPK, sink_init):
    """Gluon sparse-MLA backward dq / dkv / d_sink vs fp32 autograd reference."""
    q, kv, topk, sink = _make(S, H, TOPK, sink_init, seed=1)
    do = torch.randn(S, H, D_V, device="cuda", dtype=torch.bfloat16)
    scale = 1.0 / math.sqrt(D_QK)

    # Use the gluon forward's own (o, lse) as the bwd inputs (sink-inclusive lse).
    o_g, lse_g = sparse_mla_fwd_v4_gluon(q, kv, topk, attn_sink=sink, kv_lora_rank=D_V, scale=scale)
    dq_g, dkv_g, dsk_g = sparse_mla_bwd_v4_gluon(
        q, kv, o_g, do, topk, lse_g, attn_sink=sink, kv_lora_rank=D_V, scale=scale
    )
    dq_r, dkv_r, dsk_r = _ref_grads(q, kv, topk, sink, do, scale)

    print(f"\n[gluon DSA bwd] S={S} H={H} TOPK={TOPK} sink={sink_init}", flush=True)
    _check("dQ", dq_g, dq_r, abs_tol=5e-2, sig=1e-3, med=2e-2, cos=1e-3)
    _check("dKV", dkv_g, dkv_r.view_as(dkv_g), abs_tol=2e-1, sig=1e-3, med=2e-2, cos=1e-3)
    if sink is not None:
        _check("dSk", dsk_g, dsk_r, abs_tol=5e-1, sig=1e-3, med=5e-2, cos=1e-3)
