###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Plan-5 P31 — CSA in-kernel top-K gather equivalence.

The P26 CSA kernel consumed a materialised ``gathered`` tensor. P31 moves
that gather into the Triton kernel and scatters gradients directly back
to the compressed pool. These tests compare the new pool/topk API against
the eager reference plus PyTorch gather autograd.
"""

from __future__ import annotations

import math
from typing import Optional

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("v4_csa_attention Triton kernel requires CUDA / HIP", allow_module_level=True)

pytest.importorskip("triton", reason="Triton not installed")

from primus.backends.megatron.core.transformer.v4_attention_kernels import (  # noqa: E402
    eager_v4_csa_attention,
    v4_csa_attention_from_pool,
)

_SHAPES = [
    ("fast", 1, 4, 32, 64, 32, 16, 16),
    pytest.param("release_head_dim", 1, 16, 128, 512, 128, 64, 64, marks=pytest.mark.slow),
]
_DTYPES = [torch.float32, torch.bfloat16]
_SINK_MODES = [True, False]


def _tol(dtype: torch.dtype, *, release: bool = False) -> dict:
    if dtype == torch.float32:
        return {"atol": 1e-4, "rtol": 1e-4}
    if dtype == torch.bfloat16:
        return {"atol": 2e-1, "rtol": 2e-1} if release else {"atol": 5e-2, "rtol": 5e-2}
    raise ValueError(f"unsupported dtype {dtype!r}")


def _sink_tol(dtype: torch.dtype, *, release: bool = False) -> dict:
    if dtype == torch.float32:
        return {"atol": 1e-4, "rtol": 1e-4}
    if dtype == torch.bfloat16:
        return {"atol": 5e-2, "rtol": 5e-2} if release else {"atol": 5e-3, "rtol": 5e-3}
    raise ValueError(f"unsupported dtype {dtype!r}")


def _make_inputs(
    *,
    B: int,
    H: int,
    S: int,
    D: int,
    P: int,
    K_topk: int,
    sink_on: bool,
    dtype: torch.dtype,
    requires_grad: bool,
    seed: int = 20260509,
):
    g = torch.Generator(device="cuda").manual_seed(seed)
    device = "cuda"

    q = torch.randn(B, H, S, D, generator=g, device=device, dtype=dtype).requires_grad_(requires_grad)
    k_local = torch.randn(B, H, S, D, generator=g, device=device, dtype=dtype).requires_grad_(requires_grad)
    v_local = torch.randn(B, H, S, D, generator=g, device=device, dtype=dtype).requires_grad_(requires_grad)
    pool = torch.randn(B, P, D, generator=g, device=device, dtype=dtype).requires_grad_(requires_grad)

    topk_idxs = torch.randint(0, P, (B, S, K_topk), generator=g, device=device, dtype=torch.int64)
    if K_topk >= 4:
        # Exercise both invalid slots and duplicate slots, which are the
        # two cases the in-kernel scatter-add must handle correctly.
        topk_idxs[..., 0] = -1
        topk_idxs[..., 1] = topk_idxs[..., 2]

    sink = None
    if sink_on:
        sink_raw = torch.randn(H, generator=g, device=device, dtype=torch.float32) * 0.1
        sink = sink_raw.detach().clone().requires_grad_(requires_grad)

    return dict(q=q, k_local=k_local, v_local=v_local, pool=pool, topk_idxs=topk_idxs, sink=sink)


def _gather_from_pool(pool: torch.Tensor, topk_idxs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    B, P, D = pool.shape
    _, S, K = topk_idxs.shape
    valid = topk_idxs >= 0
    safe_idx = topk_idxs.clamp(min=0)
    gathered = torch.gather(
        pool.unsqueeze(1).expand(B, S, P, D),
        dim=2,
        index=safe_idx.unsqueeze(-1).expand(B, S, K, D),
    )
    gathered = gathered * valid.unsqueeze(-1).to(pool.dtype)
    sparse_mask = torch.where(valid, 0.0, float("-inf")).to(pool.dtype)
    return gathered, sparse_mask


def _grads_from(model_out: torch.Tensor, *leaves: Optional[torch.Tensor]) -> list[Optional[torch.Tensor]]:
    real_leaves = [leaf for leaf in leaves if leaf is not None]
    grads = torch.autograd.grad(
        outputs=model_out,
        inputs=real_leaves,
        grad_outputs=torch.ones_like(model_out),
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )
    out: list[Optional[torch.Tensor]] = []
    idx = 0
    for leaf in leaves:
        if leaf is None:
            out.append(None)
        else:
            out.append(grads[idx])
            idx += 1
    return out


@pytest.mark.parametrize("variant,B,H,S,D,P,K_topk,swa_window", _SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES, ids=lambda d: str(d).rsplit(".", 1)[-1])
@pytest.mark.parametrize("sink_on", _SINK_MODES, ids=["sink_on", "sink_off"])
def test_p31_csa_pool_fwd_matches_gathered_reference(
    variant: str,
    B: int,
    H: int,
    S: int,
    D: int,
    P: int,
    K_topk: int,
    swa_window: int,
    dtype: torch.dtype,
    sink_on: bool,
):
    inp = _make_inputs(
        B=B,
        H=H,
        S=S,
        D=D,
        P=P,
        K_topk=K_topk,
        sink_on=sink_on,
        dtype=dtype,
        requires_grad=False,
    )
    gathered, sparse_mask = _gather_from_pool(inp["pool"], inp["topk_idxs"])
    scale = 1.0 / math.sqrt(D)

    out_ref = eager_v4_csa_attention(
        inp["q"],
        inp["k_local"],
        inp["v_local"],
        gathered,
        sink=inp["sink"],
        swa_window=swa_window,
        sparse_mask=sparse_mask,
        attn_dropout=0.0,
        training=False,
        scale=scale,
    )
    out_cand = v4_csa_attention_from_pool(
        inp["q"],
        inp["k_local"],
        inp["v_local"],
        inp["pool"],
        topk_idxs=inp["topk_idxs"],
        sink=inp["sink"],
        swa_window=swa_window,
        attn_dropout=0.0,
        training=False,
        scale=scale,
    )

    torch.testing.assert_close(out_cand, out_ref, **_tol(dtype, release=variant != "fast"))


@pytest.mark.parametrize("variant,B,H,S,D,P,K_topk,swa_window", _SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES, ids=lambda d: str(d).rsplit(".", 1)[-1])
@pytest.mark.parametrize("sink_on", _SINK_MODES, ids=["sink_on", "sink_off"])
def test_p31_csa_pool_bwd_matches_gathered_reference(
    variant: str,
    B: int,
    H: int,
    S: int,
    D: int,
    P: int,
    K_topk: int,
    swa_window: int,
    dtype: torch.dtype,
    sink_on: bool,
):
    ref_inp = _make_inputs(
        B=B,
        H=H,
        S=S,
        D=D,
        P=P,
        K_topk=K_topk,
        sink_on=sink_on,
        dtype=dtype,
        requires_grad=True,
        seed=4242,
    )
    cand_inp = _make_inputs(
        B=B,
        H=H,
        S=S,
        D=D,
        P=P,
        K_topk=K_topk,
        sink_on=sink_on,
        dtype=dtype,
        requires_grad=True,
        seed=4242,
    )
    scale = 1.0 / math.sqrt(D)

    gathered, sparse_mask = _gather_from_pool(ref_inp["pool"], ref_inp["topk_idxs"])
    out_ref = eager_v4_csa_attention(
        ref_inp["q"],
        ref_inp["k_local"],
        ref_inp["v_local"],
        gathered,
        sink=ref_inp["sink"],
        swa_window=swa_window,
        sparse_mask=sparse_mask,
        attn_dropout=0.0,
        training=False,
        scale=scale,
    )
    dq_ref, dkl_ref, dvl_ref, dpool_ref, dsink_ref = _grads_from(
        out_ref,
        ref_inp["q"],
        ref_inp["k_local"],
        ref_inp["v_local"],
        ref_inp["pool"],
        ref_inp["sink"],
    )

    out_cand = v4_csa_attention_from_pool(
        cand_inp["q"],
        cand_inp["k_local"],
        cand_inp["v_local"],
        cand_inp["pool"],
        topk_idxs=cand_inp["topk_idxs"],
        sink=cand_inp["sink"],
        swa_window=swa_window,
        attn_dropout=0.0,
        training=False,
        scale=scale,
    )
    dq_cand, dkl_cand, dvl_cand, dpool_cand, dsink_cand = _grads_from(
        out_cand,
        cand_inp["q"],
        cand_inp["k_local"],
        cand_inp["v_local"],
        cand_inp["pool"],
        cand_inp["sink"],
    )

    release = variant != "fast"
    tol = _tol(dtype, release=release)
    torch.testing.assert_close(dq_cand, dq_ref, **tol)
    torch.testing.assert_close(dkl_cand, dkl_ref, **tol)
    torch.testing.assert_close(dvl_cand, dvl_ref, **tol)
    torch.testing.assert_close(dpool_cand, dpool_ref, **tol)
    if sink_on:
        torch.testing.assert_close(dsink_cand, dsink_ref, **_sink_tol(dtype, release=release))
    else:
        assert dsink_ref is None and dsink_cand is None
