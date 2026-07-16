###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Plan-4 P26 G27 — `v4_csa_attention_v0` Triton BWD equivalence to autograd-on-eager.

Asserts that gradients (``dq``, ``dk_local``, ``dv_local``,
``dgathered``, ``dsink``) returned by :func:`v4_csa_attention_v0`'s
autograd Function match the gradients computed by autograd-on-
:func:`eager_v4_csa_attention` within the plan-4 tolerance budget.

The eager reference autograd does:

* ``gathered`` is a leaf tensor with ``requires_grad=True``;
  ``dgathered`` is the gradient w.r.t. that exact tensor (the
  per-query top-K gather + scatter back to the pool happens *outside*
  the function under test, in
  :meth:`DeepseekV4Attention._csa_forward` — see plan-4 P26 design
  notes).
* ``sparse_mask`` is built from the indexer's ``topk_idxs >= 0`` test;
  it is NOT a learnable input and does not need a gradient.

The sink gradient is asserted **per-head**: ``dsink`` is shape ``[H]``
and each head's gradient must match independently.
"""

from __future__ import annotations

import math
from typing import Optional

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("v4_csa_attention_v0 Triton kernel requires CUDA / HIP", allow_module_level=True)

pytest.importorskip("triton", reason="Triton not installed")

from primus.backends.megatron.core.transformer.v4_attention_kernels import (  # noqa: E402
    eager_v4_csa_attention,
    v4_csa_attention_v0,
)

# ---------------------------------------------------------------------------
# Shape envelope (mirrors G26)
# ---------------------------------------------------------------------------


# See ``test_v4_p26_v4_csa_attention_fwd._BASE_SHAPES`` for the fast vs
# release tier rationale (G28 plan-4 release-tier shape gate).
_BASE_SHAPES = [
    ("v4_flash_small", 1, 8, 64, 64, 16, 32),
    ("v4_pro_small", 1, 4, 64, 64, 16, 32),
    # See ``test_v4_p26_v4_csa_attention_fwd._BASE_SHAPES`` for the
    # rationale on V4-Pro release ``K_topk=512`` (eager broadcast OOM).
    pytest.param(
        "v4_flash_release",
        1,
        64,
        1024,
        512,
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
        512,
        128,
        marks=pytest.mark.slow,
    ),
]
_DTYPES = [torch.float32, torch.bfloat16]
_SINK_MODES = [True, False]


def _is_release_tier(variant: str) -> bool:
    """Release-tier shapes are tagged by name (``*_release``)."""
    return variant.endswith("_release")


def _bwd_tol(dtype: torch.dtype, *, release: bool = False) -> dict:
    """Plan-4 BWD tolerance budget.

    See ``test_v4_p25_v4_attention_bwd._bwd_tol`` for the
    matmul-noise + ``atomic_add`` jitter rationale at release-tier
    ``head_dim=512``.
    """
    if dtype == torch.float32:
        return {"atol": 1e-4, "rtol": 1e-4}
    if dtype == torch.bfloat16:
        return {"atol": 2e-1, "rtol": 2e-1} if release else {"atol": 5e-2, "rtol": 5e-2}
    raise ValueError(f"unsupported dtype {dtype!r}")


def _sink_tol(dtype: torch.dtype, *, release: bool = False) -> dict:
    """Per-head sink gradient tolerance (CSA path).

    ``dsink[h]`` reduces over ``B * Sq`` softmax-derivative terms;
    cumulative bf16 rounding scales linearly with ``Sq``. Release-tier
    ``Sq`` ∈ {512, 1024} is 8x–16x the fast-tier ``Sq=64``.
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
    K_topk: int,
    sink_on: bool,
    dtype: torch.dtype,
    seed: int = 4321,
):
    """Build (q, k_local, v_local, gathered, sparse_mask, sink) leaves with requires_grad."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    device = "cuda"

    q = torch.randn(B, H, S, D, generator=g, device=device, dtype=dtype, requires_grad=True)
    k_local = torch.randn(B, H, S, D, generator=g, device=device, dtype=dtype, requires_grad=True)
    v_local = torch.randn(B, H, S, D, generator=g, device=device, dtype=dtype, requires_grad=True)

    # Build sparse_mask first (deterministic from `valid` mask), then
    # use it to also zero the gathered rows for masked positions so the
    # eager reference's autograd sees the same physical data the kernel
    # sees.
    valid = torch.rand(B, S, K_topk, generator=g, device=device) > 0.25
    sparse_mask = torch.where(
        valid,
        torch.zeros((), dtype=dtype, device=device),
        torch.tensor(float("-inf"), dtype=dtype, device=device),
    )
    gathered_raw = torch.randn(B, S, K_topk, D, generator=g, device=device, dtype=dtype)
    gathered_raw = gathered_raw * valid.unsqueeze(-1).to(dtype)
    gathered = gathered_raw.detach().clone().requires_grad_(True)

    sink = (
        torch.randn(H, generator=g, device=device, dtype=torch.float32, requires_grad=True) * 0.1
        if sink_on
        else None
    )
    return dict(
        q=q,
        k_local=k_local,
        v_local=v_local,
        gathered=gathered,
        sparse_mask=sparse_mask,
        sink=sink,
    )


def _grads_from(model_out: torch.Tensor, *leaves: torch.Tensor) -> list[Optional[torch.Tensor]]:
    """Run ``model_out.sum().backward()`` and pull ``leaf.grad`` into a list.

    Uses ``torch.autograd.grad`` so the input tensors don't need to
    have their ``.grad`` slot wiped between FWDs.
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
# G27 — CSA bwd equivalence (cr == 4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant,B,H,S,D,K_topk,swa_window", _BASE_SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES, ids=lambda d: str(d).rsplit(".", 1)[-1])
@pytest.mark.parametrize("sink_on", _SINK_MODES, ids=["sink_on", "sink_off"])
def test_g27_csa_bwd_matches_eager(
    variant: str,
    B: int,
    H: int,
    S: int,
    D: int,
    K_topk: int,
    swa_window: int,
    dtype: torch.dtype,
    sink_on: bool,
):
    """CSA bwd: kernel BWD matches autograd-on-eager."""
    # Build two independent leaf sets so the two BWD passes do not
    # interfere with each other's autograd graphs.
    ref_inp = _build_inputs(
        B=B,
        H=H,
        S=S,
        D=D,
        K_topk=K_topk,
        sink_on=sink_on,
        dtype=dtype,
        seed=4321,
    )
    cand_inp = _build_inputs(
        B=B,
        H=H,
        S=S,
        D=D,
        K_topk=K_topk,
        sink_on=sink_on,
        dtype=dtype,
        seed=4321,
    )

    scale = 1.0 / math.sqrt(D)

    out_ref = eager_v4_csa_attention(
        ref_inp["q"],
        ref_inp["k_local"],
        ref_inp["v_local"],
        ref_inp["gathered"],
        sink=ref_inp["sink"],
        swa_window=swa_window,
        sparse_mask=ref_inp["sparse_mask"],
        attn_dropout=0.0,
        training=False,
        scale=scale,
    )
    dq_ref, dkl_ref, dvl_ref, dg_ref, dsink_ref = _grads_from(
        out_ref,
        ref_inp["q"],
        ref_inp["k_local"],
        ref_inp["v_local"],
        ref_inp["gathered"],
        ref_inp["sink"],
    )

    out_cand = v4_csa_attention_v0(
        cand_inp["q"],
        cand_inp["k_local"],
        cand_inp["v_local"],
        cand_inp["gathered"],
        sink=cand_inp["sink"],
        swa_window=swa_window,
        sparse_mask=cand_inp["sparse_mask"],
        attn_dropout=0.0,
        training=False,
        scale=scale,
    )
    dq_cand, dkl_cand, dvl_cand, dg_cand, dsink_cand = _grads_from(
        out_cand,
        cand_inp["q"],
        cand_inp["k_local"],
        cand_inp["v_local"],
        cand_inp["gathered"],
        cand_inp["sink"],
    )

    release = _is_release_tier(variant)
    tol = _bwd_tol(dtype, release=release)
    torch.testing.assert_close(dq_cand, dq_ref, **tol)
    torch.testing.assert_close(dkl_cand, dkl_ref, **tol)
    torch.testing.assert_close(dvl_cand, dvl_ref, **tol)
    torch.testing.assert_close(dg_cand, dg_ref, **tol)
    if sink_on:
        assert dsink_ref.shape == dsink_cand.shape == (H,)
        torch.testing.assert_close(dsink_cand, dsink_ref, **_sink_tol(dtype, release=release))
    else:
        assert dsink_ref is None and dsink_cand is None


# ---------------------------------------------------------------------------
# G27 — gradients are finite (no NaN / Inf from the all-masked tile case)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", _DTYPES, ids=lambda d: str(d).rsplit(".", 1)[-1])
def test_g27_csa_bwd_grads_finite_with_masked_topk(dtype: torch.dtype):
    """Heavy-masked sparse branch must not blow up to NaN / Inf in bwd."""
    B, H, S, D, K_topk = 1, 4, 32, 64, 16
    swa_window = 16

    g = torch.Generator(device="cuda").manual_seed(20260507)
    device = "cuda"
    q = torch.randn(B, H, S, D, generator=g, device=device, dtype=dtype, requires_grad=True)
    k_local = torch.randn(B, H, S, D, generator=g, device=device, dtype=dtype, requires_grad=True)
    v_local = torch.randn(B, H, S, D, generator=g, device=device, dtype=dtype, requires_grad=True)

    # Mask 90% of sparse slots — forces many tiles to be fully -inf so
    # the all-masked online-softmax case is exercised.
    valid = torch.rand(B, S, K_topk, generator=g, device=device) > 0.9
    sparse_mask = torch.where(
        valid,
        torch.zeros((), dtype=dtype, device=device),
        torch.tensor(float("-inf"), dtype=dtype, device=device),
    )
    gathered = (
        (
            torch.randn(B, S, K_topk, D, generator=g, device=device, dtype=dtype)
            * valid.unsqueeze(-1).to(dtype)
        )
        .detach()
        .clone()
        .requires_grad_(True)
    )
    sink = torch.randn(H, generator=g, device=device, dtype=torch.float32, requires_grad=True) * 0.1

    out = v4_csa_attention_v0(
        q,
        k_local,
        v_local,
        gathered,
        sink=sink,
        swa_window=swa_window,
        sparse_mask=sparse_mask,
        attn_dropout=0.0,
        training=False,
        scale=1.0 / math.sqrt(D),
    )
    grads = torch.autograd.grad(
        outputs=out,
        inputs=[q, k_local, v_local, gathered, sink],
        grad_outputs=torch.ones_like(out),
    )
    for tensor in [out, *grads]:
        assert torch.isfinite(tensor).all(), "NaN / Inf observed in CSA bwd output / grads"
