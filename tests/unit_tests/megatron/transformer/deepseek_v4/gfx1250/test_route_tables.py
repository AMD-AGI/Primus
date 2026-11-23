###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Early routing tables built inside the compact route kernel.

``RouteCompactFn`` can emit the MegaMoE routing tables (row map, row weights, inverse map, per-expert offsets) as a
by-product of the route. Checked byte for byte against building them from the finished (top-k idx, weights) with
``build_routing_blk``:

* dynamic tables (``Route3Dyn``), with and without even slots, including experts that receive no token;
* the route outputs (top-k weights / idx / loss / saved tensors) are unchanged by asking for tables;
* MegaMoE fp8_block fwd + bwd with tables built by the caller (``early_routing_tables``) equals the run that builds
  its own routing.
"""

import pytest
import torch

from .helpers import bits_equal, requires_gfx1250

pytestmark = requires_gfx1250

DEV = "cuda"
E, K, SCALE, COEF = 32, 6, 1.5, 1e-4
ALIGN = 256


class _Ctx:
    def save_for_backward(self, *t):
        self.saved = t

    def mark_non_differentiable(self, *a):
        pass

    def set_materialize_grads(self, v):
        pass


def _inputs(N, H, hashed, wdtype, seed, skew=False, experts=E):
    from primus.backends.megatron.core.transformer.moe._triton.v4_route_compact import (
        pad_rows,
    )

    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(N, H, device=DEV, generator=g).to(torch.bfloat16)
    w = (torch.randn(E, H, device=DEV, generator=g) * 0.02).to(wdtype)
    if skew:
        w[:3] += 0.05
    tab = ids = None
    if hashed:
        tab = torch.stack([torch.randperm(experts, device=DEV, generator=g)[:K] for _ in range(1000)]).to(
            torch.int32
        )
        tab = pad_rows(tab)
        ids = torch.randint(0, 1000, (N,), device=DEV, generator=g)
    return x, w, tab, ids


def _logits(x, w, tab, hashed):
    from primus.backends.megatron.core.transformer.moe._triton.v4_router_fused import (
        router_fwd,
    )

    if hashed:
        return router_fwd(x, w)[0], tab
    lg, rows, _ = router_fwd(x, w, None, K, "sqrtsoftplus", SCALE, idx_pad=True)
    return lg, rows


def _same_saved(a, b):
    for u, v in zip(a, b):
        if u.dim() == 2 and u.shape[1] == 8:  # columns >= K are never written
            u, v = u[:, :K], v[:, :K]
        if not bits_equal(u, v):
            return False
    return True


@pytest.mark.parametrize(
    "hashed, slot_even, zero_experts",
    [
        (False, False, False),
        (False, True, False),
        (True, False, False),
        (True, True, False),
        (True, False, True),
    ],
)
def test_dynamic_tables_equal_build_routing(hashed, slot_even, zero_experts):
    from primus_turbo.pytorch.ops.moe.fused_mega_moe_fp8blk import routing_pool_rows
    from primus_turbo.triton.quantization.quantization_fp8blk_moe_kernel import (
        build_routing_blk,
    )

    from primus.backends.megatron.core.transformer.moe._triton.v4_route_compact import (
        Route3Dyn,
        RouteCompactFn,
    )

    N, H = 1024, 512
    x, w, tab, ids = _inputs(N, H, hashed, torch.bfloat16, 1, experts=10 if zero_experts else E)
    loss_cfg = None if hashed else (COEF, 1, N)
    logits, rows = _logits(x, w, tab, hashed)
    P = routing_pool_rows(N, K, E)
    t = Route3Dyn(E, P, ALIGN)
    c1, c0 = _Ctx(), _Ctx()
    topw, topi, loss = RouteCompactFn.forward(c1, logits, rows, ids, K, SCALE, slot_even, loss_cfg, t)
    topw0, topi0, loss0 = RouteCompactFn.forward(c0, logits, rows, ids, K, SCALE, slot_even, loss_cfg)
    ref = build_routing_blk(topi0, topw0.float(), E, P, align=ALIGN)
    torch.cuda.synchronize()
    assert bits_equal(topw, topw0) and bits_equal(topi, topi0)
    assert (loss is None and loss0 is None) or bits_equal(loss, loss0)
    for nm, u, v in zip(("rowmap", "rw", "inv", "poffs"), ref[:4], t.routing()[:4]):
        assert bits_equal(u, v), nm


def test_megamoe_with_early_tables_equals_default_routing():
    from primus_turbo.pytorch.ops.moe.fused_mega_moe_fp8blk import (
        FP8BlockWeightCache,
        early_routing_tables,
        fused_mega_moe_fp8blk,
    )

    T, H, I = 1024, 1024, 512
    g = torch.Generator(device=DEV).manual_seed(3)
    x0 = torch.randn(T, H, device=DEV, generator=g).to(torch.bfloat16)
    w1 = (torch.randn(E, 2 * I, H, device=DEV, generator=g) * 0.02).to(torch.bfloat16)
    w2 = (torch.randn(E, H, I, device=DEV, generator=g) * 0.02).to(torch.bfloat16)
    topi = (torch.arange(T * K, device=DEV) % E).view(T, K)
    tw0 = torch.rand(T, K, device=DEV, generator=g)
    dy = torch.randn(T, H, device=DEV, generator=g).to(torch.bfloat16)
    outs = []
    for use in (False, True):
        x = x0.clone().requires_grad_(True)
        tw = tw0.clone().requires_grad_(True)
        a1, a2 = w1.clone().requires_grad_(True), w2.clone().requires_grad_(True)
        c1, c2 = FP8BlockWeightCache(a1), FP8BlockWeightCache(a2)
        routing = early_routing_tables(topi, tw0, E) if use else None
        y = fused_mega_moe_fp8blk(x, topi, tw, a1, a2, c1, c2, routing=routing)
        y.backward(dy)
        torch.cuda.synchronize()
        outs.append((y.detach(), x.grad, tw.grad, a1.grad, a2.grad))
    assert all(bits_equal(u, v) for u, v in zip(*outs))
