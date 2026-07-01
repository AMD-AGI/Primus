###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""FlyDSL CSA (compress_ratio == 4) attention fwd / bwd correctness.

This module checks the **FlyDSL** CSA kernels against the eager-Python
reference (:func:`eager_v4_csa_attention`, the bit-exact "truth" defined
in ``v4_attention_kernels/reference.py``) and, for context, reports the
in-tree **Triton** kernel error against the same golden so the two
backends can be compared side by side.

Scope / constraints (driven by the FlyDSL CSA kernel preconditions):

* ``compress_ratio == 4`` (CSA) only.
* bf16 only, ``head_dim == 512`` (``D % 64 == 0`` for bwd), ``swa_window
  > 0``, ``scale == 1/sqrt(D)`` — these mirror the asserts inside
  ``_launch_v4_attention_fwd_csa`` / ``flydsl_v4_csa_attention_bwd``.
* K/V are MQA single-latent (head-identical): the FlyDSL CSA backward
  reads head 0 only (``k_local[:, :1]``), so the inputs are built by
  broadcasting a ``[B, 1, S, D]`` latent across heads — exactly what the
  V4 forward feeds the kernel in production.
* The FlyDSL backward only runs its own kernel when
  ``V4_FLYDSL_CSA_BWD_FLY_DQ=1`` and ``V4_FLYDSL_CSA_BWD_FLY_DKV=1``
  (otherwise the launcher transparently falls back to Triton). The bwd
  test sets both knobs so the FlyDSL kernel is actually exercised.

The golden reference is evaluated in fp32 on the **same bf16-rounded
input bits** (``x.float()``) so the measured error reflects kernel
compute precision (bf16 tensor-core matmul + fp32 softmax) rather than
input-quantisation noise.

GPU-only; skipped at collection time on CPU or when FlyDSL is absent.
"""

from __future__ import annotations

import math
from typing import Tuple

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("FlyDSL CSA kernels require CUDA / HIP", allow_module_level=True)

pytest.importorskip("triton", reason="Triton baseline not installed")

# Import the local primus reference + Triton launchers FIRST so they are
# cached in sys.modules before the FlyDSL kernel wrappers (which insert
# ``/workspace/Primus`` onto sys.path at import time) can shadow them.
from primus.backends.megatron.core.transformer.v4_attention_kernels import (  # noqa: E402
    eager_v4_csa_attention,
)
from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_v0_deprecated.v4_csa_attention_bwd import (  # noqa: E402
    _launch_v4_csa_attention_bwd,
)
from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_v0_deprecated.v4_csa_attention_fwd import (  # noqa: E402
    _launch_v4_csa_attention_fwd,
)

try:
    from primus.backends.megatron.core.transformer.v4_attention_kernels._flydsl_v0_deprecated.kernels.v4_attention_fwd_flydsl_csa import (  # noqa: E402
        _launch_v4_attention_fwd_csa,
    )
    from primus.backends.megatron.core.transformer.v4_attention_kernels._flydsl_v0_deprecated.kernels.v4_csa_attention_bwd_flydsl_mqa import (  # noqa: E402
        flydsl_v4_csa_attention_bwd,
    )
except Exception as exc:  # pragma: no cover - environment-dependent
    pytest.skip(f"FlyDSL CSA kernels unavailable: {exc!r}", allow_module_level=True)


# ---------------------------------------------------------------------------
# Shapes — small (fp32 golden must fit HBM) but with the production
# head_dim=512 that FlyDSL hard-requires.
# ---------------------------------------------------------------------------

# (variant, B, H, S, D, K_topk, swa_window)
_SHAPES = [
    ("flash_like", 1, 8, 256, 512, 128, 128),
    ("pro_like", 1, 16, 256, 512, 128, 128),
]
_SINK_MODES = [True, False]


def _make_inputs(
    *,
    B: int,
    H: int,
    S: int,
    D: int,
    K_topk: int,
    sink_on: bool,
    seed: int = 1234,
):
    """Build bf16 CSA inputs with MQA (head-identical) K/V.

    Returns the bf16 tensors the kernels consume plus the scalar config.
    ``sparse_mask`` drops ~25% of slots (``-inf``) and the matching
    ``gathered`` rows are zeroed so eager / Triton / FlyDSL all see the
    same physical data on the masked positions.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    device = "cuda"
    dtype = torch.bfloat16

    q = torch.randn(B, H, S, D, generator=g, device=device, dtype=dtype)
    # MQA single latent. ``k_mqa`` / ``v_mqa`` are the [B, 1, S, D] tensors
    # the FlyDSL forward consumes (it uses the mqa_kv stride trick); the
    # [B, H, S, D] broadcast (``k_full`` / ``v_full``) is what eager / Triton
    # consume and is the autograd leaf, so per-head grads can be compared
    # against the FlyDSL [B, H, S, D] dk/dv buffers. Heads are identical, so
    # all three backends see the same physical K/V.
    k_mqa = torch.randn(B, 1, S, D, generator=g, device=device, dtype=dtype)
    v_mqa = torch.randn(B, 1, S, D, generator=g, device=device, dtype=dtype)
    k_full = k_mqa.expand(B, H, S, D).contiguous()
    v_full = v_mqa.expand(B, H, S, D).contiguous()

    gathered = torch.randn(B, S, K_topk, D, generator=g, device=device, dtype=dtype)
    valid = torch.rand(B, S, K_topk, generator=g, device=device) > 0.25
    sparse_mask = torch.where(
        valid,
        torch.zeros((), dtype=dtype, device=device),
        torch.tensor(float("-inf"), dtype=dtype, device=device),
    )
    gathered = gathered * valid.unsqueeze(-1).to(dtype)

    sink = torch.randn(H, generator=g, device=device, dtype=torch.float32) * 0.1 if sink_on else None
    return q, k_full, v_full, k_mqa, v_mqa, gathered, sparse_mask, sink


def _err(cand: torch.Tensor, ref: torch.Tensor) -> Tuple[float, float]:
    """Max abs error and max relative error (vs |ref|, eps-floored)."""
    cand = cand.float()
    ref = ref.float()
    abs_err = (cand - ref).abs()
    max_abs = abs_err.max().item()
    denom = ref.abs().clamp_min(1e-6)
    max_rel = (abs_err / denom).max().item()
    return max_abs, max_rel


# ---------------------------------------------------------------------------
# Forward correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant,B,H,S,D,K_topk,swa", _SHAPES, ids=[s[0] for s in _SHAPES])
@pytest.mark.parametrize("sink_on", _SINK_MODES, ids=["sink_on", "sink_off"])
def test_flydsl_csa_fwd_matches_eager(
    variant: str,
    B: int,
    H: int,
    S: int,
    D: int,
    K_topk: int,
    swa: int,
    sink_on: bool,
):
    """FlyDSL CSA forward vs fp32 eager golden (Triton reported for context)."""
    q, k_full, v_full, k_mqa, v_mqa, gathered, sparse_mask, sink = _make_inputs(
        B=B, H=H, S=S, D=D, K_topk=K_topk, sink_on=sink_on
    )
    scale = 1.0 / math.sqrt(D)

    # fp32 golden on the same bf16-rounded bits.
    out_gold = eager_v4_csa_attention(
        q.float(),
        k_full.float(),
        v_full.float(),
        gathered.float(),
        sink=None if sink is None else sink.float(),
        swa_window=swa,
        sparse_mask=sparse_mask.float(),
        attn_dropout=0.0,
        training=False,
        scale=scale,
    )

    # FlyDSL forward consumes the MQA [B, 1, S, D] latent (mqa_kv path).
    out_fly, _lse_fly = _launch_v4_attention_fwd_csa(
        q,
        k_mqa,
        v_mqa,
        gathered,
        sink=sink,
        swa_window=swa,
        sparse_mask=sparse_mask,
        scale=scale,
    )
    # Triton consumes the [B, H, S, D] broadcast (its production input).
    out_tri, _lse_tri = _launch_v4_csa_attention_fwd(
        q,
        k_full,
        v_full,
        gathered,
        sparse_mask,
        sink=sink,
        swa_window=swa,
        scale=scale,
    )

    fly_abs, fly_rel = _err(out_fly, out_gold)
    tri_abs, tri_rel = _err(out_tri, out_gold)
    print(
        f"\n[CSA fwd] {variant} sink={sink_on} H={H} S={S} D={D} K={K_topk}\n"
        f"    FlyDSL  vs golden: max_abs={fly_abs:.3e} max_rel={fly_rel:.3e}\n"
        f"    Triton  vs golden: max_abs={tri_abs:.3e} max_rel={tri_rel:.3e}",
        flush=True,
    )

    assert out_fly.shape == out_gold.shape == q.shape
    assert out_fly.dtype == torch.bfloat16
    # bf16 tolerance; Triton must pass (sanity), FlyDSL failing is the signal.
    torch.testing.assert_close(out_tri.float(), out_gold, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(out_fly.float(), out_gold, atol=2e-2, rtol=2e-2)


# ---------------------------------------------------------------------------
# Backward correctness
# ---------------------------------------------------------------------------


def _eager_grads(q, k_local, v_local, gathered, sparse_mask, sink, *, swa, scale, dout):
    """fp32 eager autograd grads on the same bf16-rounded input bits."""
    qg = q.float().detach().requires_grad_(True)
    kg = k_local.float().detach().requires_grad_(True)
    vg = v_local.float().detach().requires_grad_(True)
    gg = gathered.float().detach().requires_grad_(True)
    sg = None if sink is None else sink.float().detach().requires_grad_(True)

    out = eager_v4_csa_attention(
        qg,
        kg,
        vg,
        gg,
        sink=sg,
        swa_window=swa,
        sparse_mask=sparse_mask.float(),
        attn_dropout=0.0,
        training=False,
        scale=scale,
    )
    out.backward(dout.float())
    return qg.grad, kg.grad, vg.grad, gg.grad, (None if sg is None else sg.grad)


@pytest.mark.parametrize("variant,B,H,S,D,K_topk,swa", _SHAPES, ids=[s[0] for s in _SHAPES])
@pytest.mark.parametrize("sink_on", _SINK_MODES, ids=["sink_on", "sink_off"])
def test_flydsl_csa_bwd_matches_eager(
    variant: str,
    B: int,
    H: int,
    S: int,
    D: int,
    K_topk: int,
    swa: int,
    sink_on: bool,
    monkeypatch,
):
    """FlyDSL CSA backward vs fp32 eager golden (Triton reported for context).

    Forces ``V4_FLYDSL_CSA_BWD_FLY_DQ=1`` + ``V4_FLYDSL_CSA_BWD_FLY_DKV=1``
    so a single FlyDSL full-kernel launch produces all five grads (else
    the launcher would fall back to Triton).
    """
    q, k_full, v_full, k_mqa, v_mqa, gathered, sparse_mask, sink = _make_inputs(
        B=B, H=H, S=S, D=D, K_topk=K_topk, sink_on=sink_on, seed=2025
    )
    scale = 1.0 / math.sqrt(D)
    gen = torch.Generator(device="cuda").manual_seed(99)
    dout = torch.randn(B, H, S, D, generator=gen, device="cuda", dtype=torch.bfloat16)

    # Golden grads (eager autograd over the [B, H, S, D] leaves).
    dq_g, dk_g, dv_g, dg_g, ds_g = _eager_grads(
        q, k_full, v_full, gathered, sparse_mask, sink, swa=swa, scale=scale, dout=dout
    )

    # FlyDSL fwd (MQA latent) -> out + lse feed FlyDSL bwd. The bwd consumes
    # the [B, H, S, D] tensor and slices head 0 internally (mqa_kv=True).
    out_fly, lse_fly = _launch_v4_attention_fwd_csa(
        q,
        k_mqa,
        v_mqa,
        gathered,
        sink=sink,
        swa_window=swa,
        sparse_mask=sparse_mask,
        scale=scale,
    )
    monkeypatch.setenv("V4_FLYDSL_CSA_BWD_FLY_DQ", "1")
    monkeypatch.setenv("V4_FLYDSL_CSA_BWD_FLY_DKV", "1")
    dq_f, dk_f, dv_f, dg_f, ds_f = flydsl_v4_csa_attention_bwd(
        q,
        k_full,
        v_full,
        gathered,
        sparse_mask,
        out_fly,
        dout,
        lse_fly,
        sink=sink,
        swa_window=swa,
        scale=scale,
    )

    # Triton baseline (its own fwd out+lse -> its own bwd).
    out_tri, lse_tri = _launch_v4_csa_attention_fwd(
        q,
        k_full,
        v_full,
        gathered,
        sparse_mask,
        sink=sink,
        swa_window=swa,
        scale=scale,
    )
    dq_t, dk_t, dv_t, dg_t, ds_t = _launch_v4_csa_attention_bwd(
        q,
        k_full,
        v_full,
        gathered,
        sparse_mask,
        out_tri,
        dout,
        lse_tri,
        sink=sink,
        swa_window=swa,
        scale=scale,
    )

    def _report(name, fly, tri, gold):
        fa, fr = _err(fly, gold)
        ta, tr = _err(tri, gold)
        print(
            f"    {name:10s} FlyDSL max_abs={fa:.3e} max_rel={fr:.3e} | "
            f"Triton max_abs={ta:.3e} max_rel={tr:.3e}",
            flush=True,
        )
        return fa, fr

    print(
        f"\n[CSA bwd] {variant} sink={sink_on} H={H} S={S} D={D} K={K_topk}",
        flush=True,
    )
    _report("dq", dq_f, dq_t, dq_g)
    _report("dk_local", dk_f, dk_t, dk_g)
    _report("dv_local", dv_f, dv_t, dv_g)
    _report("dgathered", dg_f, dg_t, dg_g)
    if sink_on:
        _report("dsink", ds_f, ds_t, ds_g)

    # bf16 gradient tolerance (looser than fwd: grads accumulate more
    # terms). Triton must pass as a sanity floor; FlyDSL failing flags a
    # precision problem to report (do NOT auto-fix).
    bwd_tol = dict(atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(dq_t.float(), dq_g, **bwd_tol)
    torch.testing.assert_close(dq_f.float(), dq_g, **bwd_tol)
    torch.testing.assert_close(dk_f.float(), dk_g, **bwd_tol)
    torch.testing.assert_close(dv_f.float(), dv_g, **bwd_tol)
    torch.testing.assert_close(dg_f.float(), dg_g, **bwd_tol)
    if sink_on:
        torch.testing.assert_close(ds_f.float(), ds_g, **bwd_tol)
