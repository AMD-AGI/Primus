###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Plan-4 P25 G24 — `v4_attention` Triton BWD equivalence to autograd-on-eager.

Asserts that gradients (``dq``, ``dk``, ``dv``, ``dsink``) returned by
:func:`v4_attention`'s autograd Function match the gradients computed
by autograd-on-:func:`eager_v4_attention` within the plan-4 tolerance
budget across the same shape envelope as G23 (V4-Flash + V4-Pro,
``compress_ratio ∈ {0, 128}``, fp32 + bf16, sink_on / sink_off, MQA /
MHA layouts).

The sink gradient is asserted **per-head**: ``dsink`` is shape ``[H]``
and each head's gradient must match independently.
"""

from __future__ import annotations

import math
from typing import Optional

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("v4_attention Triton kernel requires CUDA / HIP", allow_module_level=True)

pytest.importorskip("triton", reason="Triton not installed")

from primus.backends.megatron.core.transformer.sliding_window_kv import (  # noqa: E402
    sliding_window_causal_mask,
)
from primus.backends.megatron.core.transformer.v4_attention_kernels import (  # noqa: E402
    eager_v4_attention,
    v4_attention,
)

# ---------------------------------------------------------------------------
# Shape envelope (same as G23)
# ---------------------------------------------------------------------------


# See ``test_v4_p25_v4_attention_fwd._BASE_SHAPES`` for the fast vs
# release tier rationale (G28 plan-4 release-tier shape gate).
_BASE_SHAPES = [
    ("v4_flash_small", 1, 8, 64, 64, 32),
    ("v4_pro_small", 1, 4, 64, 64, 32),
    pytest.param(
        "v4_flash_release",
        1,
        64,
        1024,
        512,
        128,
        marks=pytest.mark.slow,
    ),
    pytest.param(
        "v4_pro_release",
        1,
        128,
        512,
        512,
        128,
        marks=pytest.mark.slow,
    ),
]
_DTYPES = [torch.float32, torch.bfloat16]
_SINK_MODES = [True, False]
_KV_LAYOUTS = ["mqa", "mha"]


def _is_release_tier(variant: str) -> bool:
    """Release-tier shapes are tagged by name (``*_release``)."""
    return variant.endswith("_release")


def _bwd_tol(dtype: torch.dtype, *, release: bool = False) -> dict:
    """Plan-4 BWD tolerance budget.

    Release-tier ``head_dim=512`` causes:
    * ~sqrt(8) ≈ 2.8x more matmul-accumulation noise than the
      ``head_dim=64`` fast tier;
    * non-deterministic ``tl.atomic_add`` contributions to ``dk / dv``
      (sliding-window cells receive ~SWA contributions, summed in a
      non-deterministic thread order) which adds another ~``sqrt(SWA)``
      bf16 jitter on top of matmul noise.

    Empirically the worst-case outlier at V4-Flash / V4-Pro release
    sits around ``0.15-0.18`` for bf16 ``dk``; we set ``atol=2e-1`` to
    absorb the long tail while keeping fast-tier shapes tight.
    """
    if dtype == torch.float32:
        return {"atol": 1e-4, "rtol": 1e-4}
    if dtype == torch.bfloat16:
        return {"atol": 2e-1, "rtol": 2e-1} if release else {"atol": 5e-2, "rtol": 5e-2}
    raise ValueError(f"unsupported dtype {dtype!r}")


def _sink_tol(dtype: torch.dtype, *, release: bool = False) -> dict:
    """Per-head sink gradient tolerance.

    ``dsink[h] = sum_{b, t} dprobs_at_sink_column[b, h, t]`` — the
    reduction is over ``B * Sq`` softmax-derivative terms, so cumulative
    bf16 rounding scales linearly with ``Sq``. Release-tier ``Sq=1024``
    is 16x the fast-tier ``Sq=64``; loosen the bf16 budget accordingly.
    """
    if dtype == torch.float32:
        return {"atol": 1e-4, "rtol": 1e-4}
    if dtype == torch.bfloat16:
        return {"atol": 5e-2, "rtol": 5e-2} if release else {"atol": 5e-3, "rtol": 5e-3}
    raise ValueError(f"unsupported dtype {dtype!r}")


def _build_inputs(
    *,
    B: int,
    H: int,
    S: int,
    D: int,
    swa_window: int,
    sink_on: bool,
    dtype: torch.dtype,
    kv_layout: str,
    use_hca: bool = False,
    seed: int = 4321,
):
    """Build (q, k, v, sink, additive_mask, swa_window) with leaves on requires_grad.

    The dense path uses ``swa_window > 0, additive_mask=None``. The HCA
    path concatenates a tiny pool to the keys / values and supplies a
    pre-built ``[Sq, Sq+P]`` joint additive mask.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    device = "cuda"
    K_H = 1 if kv_layout == "mqa" else H

    q = torch.randn(B, H, S, D, generator=g, device=device, dtype=dtype, requires_grad=True)
    if use_hca:
        P = 4
        k_local = torch.randn(B, K_H, S, D, generator=g, device=device, dtype=dtype)
        v_local = torch.randn(B, K_H, S, D, generator=g, device=device, dtype=dtype)
        pool_k = torch.randn(B, K_H, P, D, generator=g, device=device, dtype=dtype)
        pool_v = torch.randn(B, K_H, P, D, generator=g, device=device, dtype=dtype)
        k = torch.cat([k_local, pool_k], dim=2).requires_grad_(True)
        v = torch.cat([v_local, pool_v], dim=2).requires_grad_(True)
        # Joint additive mask: local SWA-causal + pool causal-on-stride
        local_mask = sliding_window_causal_mask(S, swa_window, device=device, dtype=dtype)
        ratio = 4
        t = torch.arange(S, device=device).unsqueeze(1)
        s_end = (torch.arange(P, device=device).unsqueeze(0) + 1) * ratio - 1
        pool_mask = torch.where(s_end <= t, 0.0, float("-inf")).to(dtype)
        full_mask = torch.cat([local_mask, pool_mask], dim=-1)
        kernel_swa_window = swa_window
        eager_swa_window = 0
        eager_mask: Optional[torch.Tensor] = full_mask
        kernel_mask: Optional[torch.Tensor] = pool_mask
        hca_local_seqlen = S
    else:
        k = torch.randn(B, K_H, S, D, generator=g, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(B, K_H, S, D, generator=g, device=device, dtype=dtype, requires_grad=True)
        # Eager and kernel both use the in-kernel SWA mask (eager via
        # pre-built additive_mask, kernel via swa_window).
        eager_mask = sliding_window_causal_mask(S, swa_window, device=device, dtype=dtype)
        kernel_mask = None
        eager_swa_window = 0
        kernel_swa_window = swa_window
        hca_local_seqlen = 0

    sink = (
        torch.randn(H, generator=g, device=device, dtype=torch.float32, requires_grad=True) * 0.1
        if sink_on
        else None
    )
    return dict(
        q=q,
        k=k,
        v=v,
        sink=sink,
        eager_mask=eager_mask,
        eager_swa_window=eager_swa_window,
        kernel_mask=kernel_mask,
        kernel_swa_window=kernel_swa_window,
        hca_local_seqlen=hca_local_seqlen,
    )


def _grads_from(model_out: torch.Tensor, *leaves: torch.Tensor) -> list[Optional[torch.Tensor]]:
    """Run ``model_out.sum().backward()`` and pull ``leaf.grad`` into a list.

    Cleans up ``leaf.grad`` on each leaf so the caller can re-use the
    leaves for a second autograd pass.
    """
    real_leaves = [lf for lf in leaves if lf is not None]
    grads_out = torch.ones_like(model_out)
    grads = torch.autograd.grad(
        outputs=model_out,
        inputs=real_leaves,
        grad_outputs=grads_out,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )
    # Re-pad with None for missing leaves
    out: list[Optional[torch.Tensor]] = []
    j = 0
    for lf in leaves:
        if lf is None:
            out.append(None)
        else:
            out.append(grads[j])
            j += 1
    return out


# ---------------------------------------------------------------------------
# G24 — dense (compress_ratio == 0)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant,B,H,S,D,swa_window", _BASE_SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES, ids=lambda d: str(d).rsplit(".", 1)[-1])
@pytest.mark.parametrize("sink_on", _SINK_MODES, ids=["sink_on", "sink_off"])
@pytest.mark.parametrize("kv_layout", _KV_LAYOUTS)
def test_g24_dense_bwd_matches_eager(
    variant: str,
    B: int,
    H: int,
    S: int,
    D: int,
    swa_window: int,
    dtype: torch.dtype,
    sink_on: bool,
    kv_layout: str,
):
    """Dense path: kernel BWD matches autograd-on-eager."""
    # Build two independent leaf sets so the two BWD passes do not
    # interfere with each other's ``grad`` slots.
    ref_inp = _build_inputs(
        B=B,
        H=H,
        S=S,
        D=D,
        swa_window=swa_window,
        sink_on=sink_on,
        dtype=dtype,
        kv_layout=kv_layout,
        use_hca=False,
        seed=4321,
    )
    cand_inp = _build_inputs(
        B=B,
        H=H,
        S=S,
        D=D,
        swa_window=swa_window,
        sink_on=sink_on,
        dtype=dtype,
        kv_layout=kv_layout,
        use_hca=False,
        seed=4321,
    )

    scale = 1.0 / math.sqrt(D)

    out_ref = eager_v4_attention(
        ref_inp["q"],
        ref_inp["k"],
        ref_inp["v"],
        sink=ref_inp["sink"],
        swa_window=ref_inp["eager_swa_window"],
        additive_mask=ref_inp["eager_mask"],
        attn_dropout=0.0,
        training=False,
        scale=scale,
    )
    dq_ref, dk_ref, dv_ref, dsink_ref = _grads_from(
        out_ref, ref_inp["q"], ref_inp["k"], ref_inp["v"], ref_inp["sink"]
    )

    out_cand = v4_attention(
        cand_inp["q"],
        cand_inp["k"],
        cand_inp["v"],
        sink=cand_inp["sink"],
        swa_window=cand_inp["kernel_swa_window"],
        additive_mask=cand_inp["kernel_mask"],
        attn_dropout=0.0,
        training=False,
        scale=scale,
        hca_local_seqlen=cand_inp["hca_local_seqlen"],
    )
    dq_cand, dk_cand, dv_cand, dsink_cand = _grads_from(
        out_cand, cand_inp["q"], cand_inp["k"], cand_inp["v"], cand_inp["sink"]
    )

    release = _is_release_tier(variant)
    tol = _bwd_tol(dtype, release=release)
    torch.testing.assert_close(dq_cand, dq_ref, **tol)
    torch.testing.assert_close(dk_cand, dk_ref, **tol)
    torch.testing.assert_close(dv_cand, dv_ref, **tol)
    if sink_on:
        # Per-head sink gradient: shape [H]
        assert dsink_ref.shape == dsink_cand.shape == (H,)
        torch.testing.assert_close(dsink_cand, dsink_ref, **_sink_tol(dtype, release=release))
    else:
        assert dsink_ref is None and dsink_cand is None


# ---------------------------------------------------------------------------
# G24 — HCA (compress_ratio == 128)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant,B,H,S,D,swa_window", _BASE_SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES, ids=lambda d: str(d).rsplit(".", 1)[-1])
@pytest.mark.parametrize("sink_on", _SINK_MODES, ids=["sink_on", "sink_off"])
@pytest.mark.parametrize("kv_layout", _KV_LAYOUTS)
def test_g24_hca_style_bwd_matches_eager(
    variant: str,
    B: int,
    H: int,
    S: int,
    D: int,
    swa_window: int,
    dtype: torch.dtype,
    sink_on: bool,
    kv_layout: str,
):
    """HCA path: caller-supplied additive mask, kernel BWD matches autograd-on-eager."""
    ref_inp = _build_inputs(
        B=B,
        H=H,
        S=S,
        D=D,
        swa_window=swa_window,
        sink_on=sink_on,
        dtype=dtype,
        kv_layout=kv_layout,
        use_hca=True,
        seed=4321,
    )
    cand_inp = _build_inputs(
        B=B,
        H=H,
        S=S,
        D=D,
        swa_window=swa_window,
        sink_on=sink_on,
        dtype=dtype,
        kv_layout=kv_layout,
        use_hca=True,
        seed=4321,
    )

    scale = 1.0 / math.sqrt(D)

    out_ref = eager_v4_attention(
        ref_inp["q"],
        ref_inp["k"],
        ref_inp["v"],
        sink=ref_inp["sink"],
        swa_window=0,
        additive_mask=ref_inp["eager_mask"],
        attn_dropout=0.0,
        training=False,
        scale=scale,
    )
    dq_ref, dk_ref, dv_ref, dsink_ref = _grads_from(
        out_ref, ref_inp["q"], ref_inp["k"], ref_inp["v"], ref_inp["sink"]
    )

    out_cand = v4_attention(
        cand_inp["q"],
        cand_inp["k"],
        cand_inp["v"],
        sink=cand_inp["sink"],
        swa_window=cand_inp["kernel_swa_window"],
        additive_mask=cand_inp["kernel_mask"],
        attn_dropout=0.0,
        training=False,
        scale=scale,
        hca_local_seqlen=cand_inp["hca_local_seqlen"],
    )
    dq_cand, dk_cand, dv_cand, dsink_cand = _grads_from(
        out_cand, cand_inp["q"], cand_inp["k"], cand_inp["v"], cand_inp["sink"]
    )

    release = _is_release_tier(variant)
    tol = _bwd_tol(dtype, release=release)
    torch.testing.assert_close(dq_cand, dq_ref, **tol)
    torch.testing.assert_close(dk_cand, dk_ref, **tol)
    torch.testing.assert_close(dv_cand, dv_ref, **tol)
    if sink_on:
        assert dsink_ref.shape == dsink_cand.shape == (H,)
        torch.testing.assert_close(dsink_cand, dsink_ref, **_sink_tol(dtype, release=release))
    else:
        assert dsink_ref is None and dsink_cand is None


# ---------------------------------------------------------------------------
# Sanity: dq, dk, dv finite-ness (no NaN / Inf)
# ---------------------------------------------------------------------------


def test_g24_no_nan_in_grads_dense_fp32():
    """Sanity: kernel BWD does not introduce NaN / Inf for a typical fp32 dense case."""
    inp = _build_inputs(
        B=1,
        H=4,
        S=64,
        D=64,
        swa_window=32,
        sink_on=True,
        dtype=torch.float32,
        kv_layout="mha",
        use_hca=False,
        seed=4321,
    )
    out_cand = v4_attention(
        inp["q"],
        inp["k"],
        inp["v"],
        sink=inp["sink"],
        swa_window=inp["kernel_swa_window"],
        additive_mask=inp["kernel_mask"],
        attn_dropout=0.0,
        training=False,
        scale=1.0 / math.sqrt(64),
        hca_local_seqlen=inp["hca_local_seqlen"],
    )
    grads = _grads_from(out_cand, inp["q"], inp["k"], inp["v"], inp["sink"])
    for name, g in zip(("q", "k", "v", "sink"), grads):
        if g is None:
            continue
        assert torch.isfinite(g).all(), f"d{name} contains NaN/Inf"
