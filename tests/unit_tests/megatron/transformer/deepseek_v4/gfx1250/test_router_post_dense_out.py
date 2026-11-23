###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Dense-output router post kernel (``V4RouterPostFn``) and ``RowsFromSlotsFn``.

* ``PRIMUS_V4_ROUTER_DENSE_OUT`` selects the dense-output kernel variant of router post; probs, routing map and
  d(logits) equal the other variant bit for bit for every score function and output dtype.
* Rebuilding dense rows from (slot, value) pairs (force load-balance rows) equals the scatter reference, forward and
  backward.
"""

import pytest
import torch

from .helpers import bits_equal, requires_gfx1250

pytestmark = requires_gfx1250

DEV = "cuda"
N, E, K = 1024, 32, 6


def _run(monkeypatch, dense, score_fn, out_dtype):
    from primus.backends.megatron.core.transformer.moe._triton import (
        v4_router_post as RP,
    )

    monkeypatch.setenv("PRIMUS_V4_ROUTER_DENSE_OUT", "1" if dense else "0")
    torch.manual_seed(1)
    logits = torch.randn(N, E, device=DEV, dtype=torch.float32, requires_grad=True)
    idx = torch.rand(N, E, device=DEV).topk(K, dim=-1).indices
    probs, rmap = RP.V4RouterPostFn.apply(logits, idx, score_fn, 2.5, out_dtype)
    g = torch.randn(N, E, device=DEV, dtype=out_dtype)
    (dl,) = torch.autograd.grad(probs, logits, g)
    return probs, rmap, dl


@pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("score_fn", ["sqrtsoftplus", "softmax", "sigmoid"])
def test_dense_out_variant_is_bit_equal(monkeypatch, score_fn, out_dtype):
    a = _run(monkeypatch, False, score_fn, out_dtype)
    b = _run(monkeypatch, True, score_fn, out_dtype)
    for x, y in zip(a, b):
        assert torch.equal(x, y) if x.dtype == torch.bool else bits_equal(x, y)


@pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16])
def test_rows_from_slots_matches_scatter(out_dtype):
    from primus.backends.megatron.core.transformer.moe._triton.v4_router_post import (
        RowsFromSlotsFn,
    )

    torch.manual_seed(0)
    probs = torch.rand(N, E, device=DEV).to(out_dtype).requires_grad_(True)
    slot = torch.rand(N, E, device=DEV).topk(K, dim=-1).indices
    tv, _ = torch.topk(probs, K, dim=1)
    rm0 = torch.zeros(N, E, dtype=torch.bool, device=DEV).scatter_(
        1, slot, torch.ones_like(slot, dtype=torch.bool)
    )
    p0 = torch.zeros_like(probs).scatter_(1, slot, tv.to(out_dtype))
    g = torch.randn(N, E, device=DEV, dtype=out_dtype)
    (d0,) = torch.autograd.grad(p0, probs, g)
    tv, _ = torch.topk(probs, K, dim=1)
    p1, rm1 = RowsFromSlotsFn.apply(slot, tv.to(out_dtype), E)
    (d1,) = torch.autograd.grad(p1, probs, g)
    assert bits_equal(p0, p1) and torch.equal(rm0, rm1) and bits_equal(d0, d1)
