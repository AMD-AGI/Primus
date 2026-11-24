###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Persistent router backward (``PersistRouterFn``: route backward, d(hidden), d(W) in one launch) against the
kernel chain it replaces (route backward kernel, ``router_dgrad``, ``router_wgrad``, accumulation of the two incoming
gradients).

* Forward outputs (top-k weights / idx / loss), d(hidden) and d(W) are byte-identical for the learned router with
  loss, the learned router and the hash router.
* The relaxed last-arriver hand-off of the single launch is race free: graph replay of [GEMM, router-bwd] (also with
  a concurrent GEMM on a side stream), with NaN-poisoned main_grad before each replay, gives the same bytes every
  time and the module's error counter stays clean.
"""

import pytest
import torch

from .helpers import bits_equal, requires_gfx1250

pytestmark = requires_gfx1250

DEV = "cuda"
N, H, E, K = 512, 512, 32, 6
SCALE = 1.5
V = 1000


def _make(case):
    from primus.backends.megatron.core.transformer.moe._triton.v4_route_compact import (
        pad_rows,
    )

    torch.manual_seed(0)
    hidden = (torch.randn(N, 1, H, device=DEV) * 0.5).to(torch.bfloat16)
    w = (torch.randn(E, H, device=DEV) * 0.02).float()
    bias = (torch.randn(E, device=DEV) * 0.01).float()
    dtopw = torch.randn(N, K, device=DEV) * 1e-3
    g_e = (torch.randn(N, 1, H, device=DEV) * 1e-2).to(torch.bfloat16)
    g_s = (torch.randn(N, 1, H, device=DEV) * 1e-2).to(torch.bfloat16)
    hash_ = case == "hash"
    loss_cfg = (0.0001, 1, N) if case == "loss" else None
    idx_tab = ids = None
    if hash_:
        tab = torch.stack([torch.randperm(E, device=DEV)[:K] for _ in range(V)]).to(torch.int64)
        idx_tab = pad_rows(tab)
        ids = torch.randint(0, V, (N,), device=DEV)
    return hidden, w, bias, dtopw, g_e, g_s, hash_, loss_cfg, idx_tab, ids


def _ref_path(hidden, w, bias, hash_, loss_cfg, idx_tab, ids):
    from primus.backends.megatron.core.transformer.moe._triton.v4_route_compact import (
        route_compact,
    )
    from primus.backends.megatron.core.transformer.moe._triton.v4_router_fused import (
        router_fwd,
    )

    flat = hidden.reshape(-1, H)
    if hash_:
        logits, _, _ = router_fwd(flat, w)
        idx_rows = idx_tab
    else:
        logits, idx_rows, _ = router_fwd(flat, w, bias, K, "sqrtsoftplus", SCALE, idx_pad=True)
    lg = logits.detach().requires_grad_()
    topw, topi, loss = route_compact(lg, idx_rows, ids, K, SCALE, False, loss_cfg)
    return lg, topw, topi, loss, flat


def _ref_bwd(lg, topw, flat, w, dtopw, g_e, g_s):
    from primus.backends.megatron.core.transformer.moe._triton.v4_router_fused import (
        router_dgrad,
        router_wgrad,
    )

    (dl,) = torch.autograd.grad(topw, lg, dtopw, retain_graph=True)
    acc = g_s.reshape(-1, H) + g_e.reshape(-1, H)
    r = router_dgrad(dl.contiguous().float(), w, dtype=flat.dtype)
    dw = router_wgrad(dl.contiguous().float(), flat, dtype=w.dtype)
    acc.add_(r)
    return acc, dw


@pytest.mark.parametrize("case", ["loss", "learned", "hash"])
def test_persist_router_backward_bytes(case):
    from primus.backends.megatron.core.transformer.moe._triton import (
        persist_router_bwd as prb,
    )

    hidden, w, bias, dtopw, g_e, g_s, hash_, loss_cfg, idx_tab, ids = _make(case)
    lg, topw_r, topi_r, loss_r, flat = _ref_path(hidden, w, bias, hash_, loss_cfg, idx_tab, ids)
    dx_r, dw_r = _ref_bwd(lg, topw_r, flat, w, dtopw, g_e, g_s)

    h = hidden.detach().requires_grad_()
    wp = w.detach().clone().requires_grad_()
    st = prb.RouterStash()
    a_r, a_e, a_s = prb.PersistForkFn.apply(h, st)
    topw, topi, loss = prb.PersistRouterFn.apply(
        a_r, wp, bias, idx_tab, ids, K, "sqrtsoftplus", SCALE, False, loss_cfg, hash_, False, st
    )
    torch.autograd.backward([topw, a_e, a_s], [dtopw, g_e, g_s])
    torch.cuda.synchronize()
    assert not prb.check_error()
    assert bits_equal(topw, topw_r) and bits_equal(topi, topi_r)
    assert (loss is None and loss_r is None) or bits_equal(loss, loss_r)
    assert bits_equal(h.grad, dx_r.view(h.shape))
    assert bits_equal(wp.grad, dw_r)


@pytest.mark.parametrize("case", ["loss", "hash"])
def test_persist_router_backward_graph_replay_is_race_free(case):
    from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler

    from primus.backends.megatron.core.transformer.moe._triton import (
        persist_router_bwd as prb,
    )

    hidden, w, bias, dtopw, g_e, g_s, hash_, loss_cfg, idx_tab, ids = _make(case)
    lg, topw_r, _, _, flat = _ref_path(hidden, w, bias, hash_, loss_cfg, idx_tab, ids)
    dx_r, dw_r = _ref_bwd(lg, topw_r, flat, w, dtopw, g_e, g_s)
    logits_s, idx_rows_s, ids_s, wsave, denom, cnt = topw_r.grad_fn.saved_tensors
    has_loss = loss_cfg is not None
    dscale = MoEAuxLossAutoScaler.main_loss_backward_scale if has_loss else logits_s
    route = (
        logits_s, idx_rows_s, ids_s, wsave, denom, cnt, dscale, K, N if has_loss else 1,
        topw_r.grad_fn.lcoef if has_loss else 0.0, SCALE, hash_, has_loss,
    )  # fmt: skip
    mg = torch.zeros(E, H, device=DEV)
    out = {}
    a = torch.randn(2048, 2048, device=DEV, dtype=torch.bfloat16)

    def fn():
        out["dx"] = prb.router_bwd_mega(
            route, flat, w, dtopw, g_s.reshape(-1, H), g_e.reshape(-1, H), mg, w.dtype, False, sync_key="test"
        )

    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        fn()
        fn()
    torch.cuda.current_stream().wait_stream(st)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        a @ a
        fn()
    side = torch.cuda.Stream()
    for i in range(60):
        mg.fill_(float("nan"))
        if i % 2:
            with torch.cuda.stream(side):
                a @ a  # concurrent work on another stream
        gr.replay()
        torch.cuda.synchronize()
        assert bits_equal(out["dx"], dx_r.view(out["dx"].shape)), f"dx, replay {i}"
        assert bits_equal(mg, dw_r), f"main_grad, replay {i}"
    assert not prb.check_error()
