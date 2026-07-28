###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Plan-4 P25 G23 — `v4_attention_v1` Triton FWD equivalence to eager.

Asserts that :func:`v4_attention_v1` (Triton kernel from
``primus...transformer.v4_attention_kernels.v4_attention``) produces
forward output equal to :func:`eager_v4_attention` within the plan-4
tolerance budget across:

* V4-Flash and V4-Pro shape envelopes (head_dim=512, H ∈ {64, 128});
* ``compress_ratio ∈ {0, 128}`` — dense + SWA + sink (no bias) and HCA
  (joint-softmax additive bias, no in-kernel SWA);
* fp32 and bf16 inputs;
* ``sink ∈ {None, learned [H]}``;
* MQA (``K_H == 1``) and MHA (``K_H == HQ``) layouts.

The test is GPU-only (Triton requires CUDA / HIP); CPU runs are
``pytest.skip``-ed at module collection time.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("v4_attention_v1 Triton kernel requires CUDA / HIP", allow_module_level=True)

# Triton import (must be importable in the env).
pytest.importorskip("triton", reason="Triton not installed")

from primus.backends.megatron.core.transformer.sliding_window_kv import (  # noqa: E402
    sliding_window_causal_mask,
)
from primus.backends.megatron.core.transformer.v4_attention_kernels import (  # noqa: E402
    eager_v4_attention,
    v4_attention_v1,
)

# ---------------------------------------------------------------------------
# Shape envelope (V4-Flash / V4-Pro)
# ---------------------------------------------------------------------------


# Two tiers of shapes are exposed to the parametrise decorator below:
#
# * Fast tier — toy ``head_dim=64`` / ``H ∈ {4, 8}`` / ``S=64`` shapes
#   that exercise every code path in the kernel (in-kernel SWA mask,
#   caller-supplied additive_mask, MQA/MHA, sink) in milliseconds. Run
#   on every ``pytest`` invocation.
#
# * Release tier (``pytest.mark.slow``) — production V4 dimensions
#   (``head_dim=512``, real ``H``, real ``swa_window``) calibrated so
#   the eager fp32 reference fits MI355X HBM. The release tier is the
#   plan-4 G28 gate that empirically confirms kernel correctness at the
#   exact ``head_dim`` that plan-4 exists to solve. ``S`` is calibrated
#   per variant: V4-Flash @ S=1024 / V4-Pro @ S=512 keep peak fp32
#   memory below ~1 GiB per test even for the MHA layout.
#
# The full ``S=4096`` smoke is owned by G30 (P27 smoke gate); these
# unit tests intentionally stay below that to keep the per-test cost
# in the ``-m slow`` budget.
_BASE_SHAPES = [
    # (variant, B, H, S, head_dim, swa_window)
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
_KV_LAYOUTS = ["mqa", "mha"]  # K_H == 1 vs K_H == HQ


def _is_release_tier(variant: str) -> bool:
    """Release-tier shapes are tagged by name (``*_release``)."""
    return variant.endswith("_release")


def _fwd_tol(dtype: torch.dtype, *, release: bool = False) -> dict:
    if dtype == torch.float32:
        return {"atol": 1e-4, "rtol": 1e-4}
    if dtype == torch.bfloat16:
        # Release-tier ``head_dim=512`` accumulates ~8x more terms than
        # the ``head_dim=64`` fast tier; bf16 long-tail rounding scales
        # as ~sqrt(N). Empirically the worst-case outlier sits between
        # 2.0e-2 and 5.0e-2 at release-tier dims; we loosen by ~2.5x.
        return {"atol": 5e-2, "rtol": 5e-2} if release else {"atol": 2e-2, "rtol": 2e-2}
    raise ValueError(f"unsupported dtype {dtype!r}")


def _make_dense_inputs(
    *,
    B: int,
    H: int,
    S: int,
    D: int,
    swa_window: int,
    sink_on: bool,
    dtype: torch.dtype,
    kv_layout: str,
    seed: int = 1234,
):
    """Build inputs for a dense (compress_ratio=0) test case.

    The kernel is invoked with ``swa_window > 0, additive_mask=None``;
    the eager reference is invoked with the equivalent ``additive_mask``
    pre-built (so both go through identical mask math).
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    device = "cuda"
    K_H = 1 if kv_layout == "mqa" else H

    q = torch.randn(B, H, S, D, generator=g, device=device, dtype=dtype)
    k = torch.randn(B, K_H, S, D, generator=g, device=device, dtype=dtype)
    v = torch.randn(B, K_H, S, D, generator=g, device=device, dtype=dtype)
    sink = torch.randn(H, generator=g, device=device, dtype=torch.float32) * 0.1 if sink_on else None
    return dict(B=B, H=H, S=S, D=D, swa_window=swa_window, q=q, k=k, v=v, sink=sink)


def _make_hca_inputs(
    *,
    B: int,
    H: int,
    S: int,
    P: int,
    D: int,
    swa_window: int,
    sink_on: bool,
    dtype: torch.dtype,
    kv_layout: str,
    seed: int = 1234,
):
    """Build inputs for an HCA (compress_ratio=128) test case.

    HCA pre-concatenates a length-``P`` compressed-pool key/value to the
    length-``S`` local key/value, and the caller pre-builds the joint
    additive mask so the kernel does NOT apply SWA / causal in-kernel.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    device = "cuda"
    K_H = 1 if kv_layout == "mqa" else H

    q = torch.randn(B, H, S, D, generator=g, device=device, dtype=dtype)
    k_local = torch.randn(B, K_H, S, D, generator=g, device=device, dtype=dtype)
    v_local = torch.randn(B, K_H, S, D, generator=g, device=device, dtype=dtype)
    pool_k = torch.randn(B, K_H, P, D, generator=g, device=device, dtype=dtype)
    pool_v = torch.randn(B, K_H, P, D, generator=g, device=device, dtype=dtype)
    k_full = torch.cat([k_local, pool_k], dim=2)
    v_full = torch.cat([v_local, pool_v], dim=2)

    # Local SWA-causal mask: [S, S]
    local_mask = sliding_window_causal_mask(S, swa_window, device=device, dtype=dtype)
    # Pool causal-on-stride mask: pool[s] is visible at query t iff
    # (s+1)*ratio - 1 <= t. Use a fixed ratio of 4 for the test.
    ratio = 4
    t = torch.arange(S, device=device).unsqueeze(1)
    s_end = (torch.arange(P, device=device).unsqueeze(0) + 1) * ratio - 1
    pool_mask = torch.where(s_end <= t, 0.0, float("-inf")).to(dtype)
    full_mask = torch.cat([local_mask, pool_mask], dim=-1)  # [S, S+P]

    sink = torch.randn(H, generator=g, device=device, dtype=torch.float32) * 0.1 if sink_on else None
    return dict(
        B=B,
        H=H,
        S=S,
        P=P,
        D=D,
        q=q,
        k=k_full,
        v=v_full,
        sink=sink,
        full_mask=full_mask,
        pool_mask=pool_mask,
    )


# ---------------------------------------------------------------------------
# G23 — dense (compress_ratio == 0)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant,B,H,S,D,swa_window", _BASE_SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES, ids=lambda d: str(d).rsplit(".", 1)[-1])
@pytest.mark.parametrize("sink_on", _SINK_MODES, ids=["sink_on", "sink_off"])
@pytest.mark.parametrize("kv_layout", _KV_LAYOUTS)
def test_g23_dense_fwd_matches_eager(
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
    """Dense path: kernel uses in-kernel SWA-causal mask + optional sink."""
    toy = _make_dense_inputs(
        B=B,
        H=H,
        S=S,
        D=D,
        swa_window=swa_window,
        sink_on=sink_on,
        dtype=dtype,
        kv_layout=kv_layout,
    )

    scale = 1.0 / math.sqrt(D)

    # Reference: eager_v4_attention with pre-built SWA additive_mask
    eager_mask = sliding_window_causal_mask(S, swa_window, device=toy["q"].device, dtype=dtype)
    out_ref = eager_v4_attention(
        toy["q"],
        toy["k"],
        toy["v"],
        sink=toy["sink"],
        swa_window=0,
        additive_mask=eager_mask,
        attn_dropout=0.0,
        training=False,
        scale=scale,
    )

    # Candidate: v4_attention_v1 with swa_window > 0, additive_mask=None
    out_cand = v4_attention_v1(
        toy["q"],
        toy["k"],
        toy["v"],
        sink=toy["sink"],
        swa_window=swa_window,
        additive_mask=None,
        attn_dropout=0.0,
        training=False,
        scale=scale,
    )

    assert out_ref.shape == out_cand.shape == toy["q"].shape
    assert out_ref.dtype == out_cand.dtype == dtype
    torch.testing.assert_close(
        out_cand,
        out_ref,
        **_fwd_tol(dtype, release=_is_release_tier(variant)),
    )


# ---------------------------------------------------------------------------
# G23 — HCA (compress_ratio == 128) — caller-supplied additive mask
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant,B,H,S,D,swa_window", _BASE_SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES, ids=lambda d: str(d).rsplit(".", 1)[-1])
@pytest.mark.parametrize("sink_on", _SINK_MODES, ids=["sink_on", "sink_off"])
@pytest.mark.parametrize("kv_layout", _KV_LAYOUTS)
def test_g23_hca_style_fwd_matches_eager(
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
    """HCA path: caller pre-concatenates pool keys + supplies joint additive mask."""
    P = 4  # tiny pool — exercises the additive_mask branch w/o full HCA setup
    toy = _make_hca_inputs(
        B=B,
        H=H,
        S=S,
        P=P,
        D=D,
        swa_window=swa_window,
        sink_on=sink_on,
        dtype=dtype,
        kv_layout=kv_layout,
    )

    scale = 1.0 / math.sqrt(D)

    out_ref = eager_v4_attention(
        toy["q"],
        toy["k"],
        toy["v"],
        sink=toy["sink"],
        swa_window=0,
        additive_mask=toy["full_mask"],
        attn_dropout=0.0,
        training=False,
        scale=scale,
    )

    out_cand = v4_attention_v1(
        toy["q"],
        toy["k"],
        toy["v"],
        sink=toy["sink"],
        swa_window=swa_window,
        additive_mask=toy["pool_mask"],
        attn_dropout=0.0,
        training=False,
        scale=scale,
        hca_local_seqlen=S,
    )

    assert out_ref.shape == out_cand.shape == toy["q"].shape
    assert out_ref.dtype == out_cand.dtype == dtype
    torch.testing.assert_close(
        out_cand,
        out_ref,
        **_fwd_tol(dtype, release=_is_release_tier(variant)),
    )


# ---------------------------------------------------------------------------
# G25 — determinism with attn_dropout=0.0
# ---------------------------------------------------------------------------


def test_g25_determinism_fp32_mha():
    """Repeated FWD calls with the same inputs produce bit-identical output (fp32 / MHA)."""
    toy = _make_dense_inputs(
        B=1,
        H=4,
        S=64,
        D=64,
        swa_window=32,
        sink_on=True,
        dtype=torch.float32,
        kv_layout="mha",
    )
    scale = 1.0 / math.sqrt(toy["D"])

    out_a = v4_attention_v1(
        toy["q"],
        toy["k"],
        toy["v"],
        sink=toy["sink"],
        swa_window=toy["swa_window"],
        additive_mask=None,
        attn_dropout=0.0,
        training=False,
        scale=scale,
    )
    out_b = v4_attention_v1(
        toy["q"],
        toy["k"],
        toy["v"],
        sink=toy["sink"],
        swa_window=toy["swa_window"],
        additive_mask=None,
        attn_dropout=0.0,
        training=False,
        scale=scale,
    )
    assert torch.equal(out_a, out_b), "v4_attention_v1 FWD is non-deterministic at fp32 / MHA"


def test_g25_dropout_with_training_is_rejected():
    """``attn_dropout > 0`` with ``training=True`` raises (kernel does not implement dropout)."""
    toy = _make_dense_inputs(
        B=1,
        H=4,
        S=64,
        D=64,
        swa_window=32,
        sink_on=False,
        dtype=torch.float32,
        kv_layout="mha",
    )
    scale = 1.0 / math.sqrt(toy["D"])
    with pytest.raises(NotImplementedError, match="dropout"):
        v4_attention_v1(
            toy["q"],
            toy["k"],
            toy["v"],
            sink=None,
            swa_window=toy["swa_window"],
            additive_mask=None,
            attn_dropout=0.1,
            training=True,
            scale=scale,
        )
