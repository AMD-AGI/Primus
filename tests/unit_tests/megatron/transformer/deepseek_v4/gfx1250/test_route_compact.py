###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Compact routing (``route_compact``) against the dense chain it replaces.

Dense chain: fused router -> ``V4RouterPostFn`` -> sequence-balance loss (learned router only) -> force-load-balance
"even" slots -> ``probs.topk``. The compact path computes the top-k weights / indices (and the loss) in one kernel
from the router logits and the routed experts. Checked for the learned router (with auxiliary loss) and the hash
router, with and without even slots:

* top-k weights and indices are bit-identical to the dense chain;
* d(logits) of the compact backward matches an fp64 reference within the FP32 error of the dense chain;
* the loss matches the dense chain within FP32 rounding; dx / dW are finite.

The compact path is selected by ``PRIMUS_FUSE_ROUTE``; the dense chain needs ``PRIMUS_V4_ROUTER_DENSE_OUT=1``.
"""

import pytest
import torch

from .helpers import bits_equal, requires_gfx1250

pytestmark = requires_gfx1250

DEV = "cuda"
N, H, E, K = 1024, 512, 32, 6
COEF, SCALE = 1e-4, 1.0


def _setup(monkeypatch):
    from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler

    monkeypatch.setenv("PRIMUS_V4_ROUTER_DENSE_OUT", "1")
    MoEAuxLossAutoScaler.main_loss_backward_scale = torch.tensor(1.0, device=DEV)


def _dense(x, w, bias, hash_ids, tid2eid, slot_even):
    from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler

    from primus.backends.megatron.core.transformer.moe._triton.v4_router_fused import (
        v4_router_fused,
    )
    from primus.backends.megatron.core.transformer.moe._triton.v4_router_post import (
        RowsFromSlotsFn,
        v4_router_post_triton,
    )
    from primus.backends.megatron.core.transformer.moe.v4_seq_balance_loss import (
        normalise_affinities,
        sequence_balance_loss,
    )
    from primus.backends.megatron.core.transformer.moe.v4_topk_router import v4_score_fn

    if hash_ids is None:
        logits, idx, _ = v4_router_fused(x, w, bias, K, "sqrtsoftplus", SCALE)
    else:
        logits, _, _ = v4_router_fused(x, w)
        idx = tid2eid[hash_ids].long()
    logits.retain_grad()
    probs, rmap = v4_router_post_triton(
        logits, idx, score_function="sqrtsoftplus", topk_scaling_factor=SCALE, out_dtype=logits.dtype
    )
    loss = None
    if hash_ids is None:
        aff = normalise_affinities(v4_score_fn(logits, score_function="sqrtsoftplus"))
        loss = sequence_balance_loss(
            scores=aff.view(1, N, E), routing_map=rmap.view(1, N, E), topk=K, coeff=COEF
        )
        probs = MoEAuxLossAutoScaler.apply(probs, loss)
    if slot_even:
        slot = torch.arange(N * K, device=DEV).view(N, K) % E
        vals, _ = torch.topk(probs, K, dim=1)
        probs, rmap = RowsFromSlotsFn.apply(slot, vals.to(probs.dtype), E)
    tw, ti = probs.topk(K, dim=-1)
    return tw.to(torch.float32), ti, loss, logits, idx


def _compact(x, w, bias, hash_ids, tid2eid, slot_even):
    from primus.backends.megatron.core.transformer.moe._triton.v4_route_compact import (
        pad_rows,
        route_compact,
    )
    from primus.backends.megatron.core.transformer.moe._triton.v4_router_fused import (
        v4_router_fused,
    )

    if hash_ids is None:
        logits, idx, _ = v4_router_fused(x, w, bias, K, "sqrtsoftplus", SCALE, idx_pad=True)
        logits.retain_grad()
        tw, ti, loss = route_compact(logits, idx, None, K, SCALE, slot_even, (COEF, 1, N))
    else:
        logits, _, _ = v4_router_fused(x, w)
        logits.retain_grad()
        tw, ti, _ = route_compact(logits, pad_rows(tid2eid), hash_ids, K, SCALE, slot_even, None)
        loss = None
    return tw, ti, loss, logits, None


def _ref64(logits, idx, dtw):
    l = logits.detach().double().requires_grad_(True)
    s = torch.nn.functional.softplus(l).sqrt()
    g = s.gather(1, idx)
    wts = g / g.sum(1, keepdim=True).clamp_min(1e-12) * SCALE
    order = torch.argsort(wts, dim=1, descending=True, stable=True)
    (wts.gather(1, order) * dtw.double()).sum().backward()
    return l.grad


@pytest.mark.parametrize("slot_even", [True, False], ids=["even", "free"])
@pytest.mark.parametrize("hashed", [False, True], ids=["learned", "hash"])
@pytest.mark.parametrize("seed", [0, 1])
def test_compact_route_matches_dense_chain(monkeypatch, seed, hashed, slot_even):
    _setup(monkeypatch)
    torch.manual_seed(seed)
    x = (torch.randn(N, H, device=DEV) * 0.5).to(torch.bfloat16)
    w = torch.randn(E, H, device=DEV) * 0.02
    bias = torch.randn(E, device=DEV) * 0.01
    ids = torch.randint(0, 1000, (N,), device=DEV)
    tid = torch.stack([torch.randperm(E, device=DEV)[:K] for _ in range(1000)]).to(torch.int32)
    dtw = torch.randn(N, K, device=DEV)
    res = []
    for fn in (_dense, _compact):
        xx = x.clone().requires_grad_(True)
        ww = w.clone().requires_grad_(True)
        tw, ti, loss, logits, idx = fn(
            xx, ww, bias, ids if hashed else None, tid if hashed else None, slot_even
        )
        tw.backward(dtw)
        res.append((tw.detach(), ti, loss, xx.grad, ww.grad, logits))
    (tw0, ti0, l0, dx0, dw0, lg0), (tw1, ti1, l1, dx1, dw1, lg1) = res
    assert torch.equal(ti0, ti1)
    assert bits_equal(tw0, tw1)
    assert torch.isfinite(dx1).all() and torch.isfinite(dw1).all()
    if l0 is not None:
        assert abs(l0.item() - l1.item()) <= 1e-5 * abs(l0.item())
    # d(logits): the compact backward is as close to fp64 as the dense chain
    if hashed:
        idx_ref = tid[ids].long()
    else:
        from primus.backends.megatron.core.transformer.moe._triton.v4_router_fused import (
            v4_router_fused,
        )

        idx_ref = v4_router_fused(x, w, bias, K, "sqrtsoftplus", SCALE)[1]
    d64 = _ref64(lg1, idx_ref, dtw)

    def rel(a):
        return ((a.double() - d64).norm() / d64.norm()).item()

    assert rel(lg1.grad) <= max(2.0 * rel(lg0.grad), 1e-5)
