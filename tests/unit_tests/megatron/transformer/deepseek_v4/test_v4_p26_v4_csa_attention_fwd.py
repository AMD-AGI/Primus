###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Plan-4 P26 G26 — `v4_csa_attention` Triton FWD equivalence to eager.

Asserts that :func:`v4_csa_attention` (Triton kernel from
``primus...transformer.v4_attention_kernels.v4_csa_attention``)
produces forward output equal to :func:`eager_v4_csa_attention` within
the plan-4 tolerance budget across:

* V4-Flash and V4-Pro shape envelopes (head_dim=512 in production, but
  small / fast tier here for CI time);
* ``compress_ratio == 4`` only — the dense / HCA paths are covered by
  G23 / G24 in :mod:`test_v4_p25_v4_attention_fwd`;
* fp32 and bf16 inputs;
* ``sink ∈ {None, learned [H]}``;
* ``K_topk == 0`` short-circuit — the wrapper falls through to the
  dense :func:`v4_attention` kernel and the result must still match
  the eager reference's degenerate (``gathered.shape[2] == 0``) limit.

The test is GPU-only (Triton requires CUDA / HIP); CPU runs are
``pytest.skip``-ed at module collection time.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("v4_csa_attention Triton kernel requires CUDA / HIP", allow_module_level=True)

pytest.importorskip("triton", reason="Triton not installed")

from primus.backends.megatron.core.transformer.v4_attention_kernels import (  # noqa: E402
    eager_v4_csa_attention,
    v4_csa_attention,
)

# ---------------------------------------------------------------------------
# Shape envelope (V4-Flash / V4-Pro, small / fast tier for CI)
# ---------------------------------------------------------------------------


# Shapes are intentionally smaller than the full V4 yamls so the suite
# completes in seconds on a single MI355X. The numerical contract being
# tested is the kernel's, not the model's full size.
_BASE_SHAPES = [
    # (variant, B, H, S, D, K_topk, swa_window)
    ("v4_flash_small", 1, 8, 64, 64, 16, 32),
    ("v4_pro_small", 1, 4, 64, 64, 16, 32),
]
_DTYPES = [torch.float32, torch.bfloat16]
_SINK_MODES = [True, False]


def _fwd_tol(dtype: torch.dtype) -> dict:
    if dtype == torch.float32:
        return {"atol": 1e-4, "rtol": 1e-4}
    if dtype == torch.bfloat16:
        return {"atol": 2e-2, "rtol": 2e-2}
    raise ValueError(f"unsupported dtype {dtype!r}")


def _make_csa_inputs(
    *,
    B: int,
    H: int,
    S: int,
    D: int,
    K_topk: int,
    sink_on: bool,
    dtype: torch.dtype,
    seed: int = 1234,
):
    """Build inputs for a CSA test case.

    The kernel + eager reference share the exact same ``[B, H, Sq, D]``
    K_local / V_local layout (CSA always has K_H == HQ — the V4
    forward broadcast-expanded MQA single-latent KV across heads
    before reaching ``_csa_forward``). ``gathered`` is per-query
    (no H dim); ``sparse_mask`` flips a few entries to ``-inf`` to
    exercise the topk == -1 path.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    device = "cuda"

    q = torch.randn(B, H, S, D, generator=g, device=device, dtype=dtype)
    k_local = torch.randn(B, H, S, D, generator=g, device=device, dtype=dtype)
    v_local = torch.randn(B, H, S, D, generator=g, device=device, dtype=dtype)
    gathered = torch.randn(B, S, K_topk, D, generator=g, device=device, dtype=dtype)

    # Sparse mask: 0 for valid entries, -inf for "indexer dropped" entries.
    # Drop a randomized 25% of slots so both branches in the kernel see
    # masked entries.
    valid = torch.rand(B, S, K_topk, generator=g, device=device) > 0.25
    sparse_mask = torch.where(
        valid,
        torch.zeros((), dtype=dtype, device=device),
        torch.tensor(float("-inf"), dtype=dtype, device=device),
    )
    # Also zero the gathered rows for masked positions so the eager
    # reference's autograd sees the same physical data the kernel sees.
    gathered = gathered * valid.unsqueeze(-1).to(dtype)

    sink = torch.randn(H, generator=g, device=device, dtype=torch.float32) * 0.1 if sink_on else None
    return dict(
        B=B,
        H=H,
        S=S,
        D=D,
        K_topk=K_topk,
        q=q,
        k_local=k_local,
        v_local=v_local,
        gathered=gathered,
        sparse_mask=sparse_mask,
        sink=sink,
    )


# ---------------------------------------------------------------------------
# G26 — CSA fwd equivalence (cr == 4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant,B,H,S,D,K_topk,swa_window", _BASE_SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES, ids=lambda d: str(d).rsplit(".", 1)[-1])
@pytest.mark.parametrize("sink_on", _SINK_MODES, ids=["sink_on", "sink_off"])
def test_g26_csa_fwd_matches_eager(
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
    """CSA fwd: kernel matches eager joint local-SWA + sparse-topK + sink path."""
    toy = _make_csa_inputs(
        B=B,
        H=H,
        S=S,
        D=D,
        K_topk=K_topk,
        sink_on=sink_on,
        dtype=dtype,
    )

    scale = 1.0 / math.sqrt(D)

    out_ref = eager_v4_csa_attention(
        toy["q"],
        toy["k_local"],
        toy["v_local"],
        toy["gathered"],
        sink=toy["sink"],
        swa_window=swa_window,
        sparse_mask=toy["sparse_mask"],
        attn_dropout=0.0,
        training=False,
        scale=scale,
    )

    out_cand = v4_csa_attention(
        toy["q"],
        toy["k_local"],
        toy["v_local"],
        toy["gathered"],
        sink=toy["sink"],
        swa_window=swa_window,
        sparse_mask=toy["sparse_mask"],
        attn_dropout=0.0,
        training=False,
        scale=scale,
    )

    assert out_ref.shape == out_cand.shape == toy["q"].shape
    assert out_ref.dtype == out_cand.dtype == dtype
    torch.testing.assert_close(out_cand, out_ref, **_fwd_tol(dtype))


# ---------------------------------------------------------------------------
# G26 — K_topk == 0 short-circuit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", _DTYPES, ids=lambda d: str(d).rsplit(".", 1)[-1])
@pytest.mark.parametrize("sink_on", _SINK_MODES, ids=["sink_on", "sink_off"])
def test_g26_csa_fwd_short_circuits_when_k_topk_zero(dtype: torch.dtype, sink_on: bool):
    """``v4_csa_attention(..., gathered.shape[2] == 0, ...)`` falls through to dense."""
    B, H, S, D = 1, 4, 32, 64
    swa_window = 16

    g = torch.Generator(device="cuda").manual_seed(2026)
    device = "cuda"
    q = torch.randn(B, H, S, D, generator=g, device=device, dtype=dtype)
    k_local = torch.randn(B, H, S, D, generator=g, device=device, dtype=dtype)
    v_local = torch.randn(B, H, S, D, generator=g, device=device, dtype=dtype)
    gathered = torch.empty(B, S, 0, D, device=device, dtype=dtype)
    sparse_mask = torch.empty(B, S, 0, device=device, dtype=dtype)
    sink = torch.randn(H, generator=g, device=device, dtype=torch.float32) * 0.1 if sink_on else None

    scale = 1.0 / math.sqrt(D)

    # Reference: eager CSA with empty gathered. The eager path's
    # ``torch.cat([local_logits, sparse_logits], dim=-1)`` reduces to
    # just ``local_logits`` because ``sparse_logits.shape[-1] == 0``,
    # and the joint softmax collapses to the local-only softmax.
    out_ref = eager_v4_csa_attention(
        q,
        k_local,
        v_local,
        gathered,
        sink=sink,
        swa_window=swa_window,
        sparse_mask=sparse_mask,
        attn_dropout=0.0,
        training=False,
        scale=scale,
    )

    # Candidate: wrapper short-circuits to dense v4_attention.
    out_cand = v4_csa_attention(
        q,
        k_local,
        v_local,
        gathered,
        sink=sink,
        swa_window=swa_window,
        sparse_mask=sparse_mask,
        attn_dropout=0.0,
        training=False,
        scale=scale,
    )

    assert out_ref.shape == out_cand.shape == (B, H, S, D)
    torch.testing.assert_close(out_cand, out_ref, **_fwd_tol(dtype))


# ---------------------------------------------------------------------------
# G26 — explicit dropout-with-training rejection
# ---------------------------------------------------------------------------


def test_g26_csa_dropout_with_training_is_rejected():
    """``v4_csa_attention`` raises NotImplementedError on dropout + training=True.

    V4 trains with attn_dropout=0; the kernel does not implement
    in-kernel attention dropout. We refuse explicitly so a stray
    non-zero dropout configuration raises rather than silently dropping
    the kernel path.
    """
    B, H, S, D, K_topk = 1, 2, 16, 64, 4
    swa_window = 8
    dtype = torch.float32
    g = torch.Generator(device="cuda").manual_seed(42)
    device = "cuda"

    q = torch.randn(B, H, S, D, generator=g, device=device, dtype=dtype)
    k_local = torch.randn(B, H, S, D, generator=g, device=device, dtype=dtype)
    v_local = torch.randn(B, H, S, D, generator=g, device=device, dtype=dtype)
    gathered = torch.randn(B, S, K_topk, D, generator=g, device=device, dtype=dtype)
    sparse_mask = torch.zeros(B, S, K_topk, device=device, dtype=dtype)

    with pytest.raises(NotImplementedError, match="does not implement in-kernel"):
        v4_csa_attention(
            q,
            k_local,
            v_local,
            gathered,
            sink=None,
            swa_window=swa_window,
            sparse_mask=sparse_mask,
            attn_dropout=0.1,
            training=True,
            scale=1.0 / math.sqrt(D),
        )
