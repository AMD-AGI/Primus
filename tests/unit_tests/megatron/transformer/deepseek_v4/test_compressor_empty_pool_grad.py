###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""An empty compressed pool must still be connected to the compressor's parameters.

A sequence contributes ``len // ratio`` compressed windows, so a pack whose segments are
all shorter than ``ratio`` produces NO windows. That is routine rather than exceptional:
HCA uses ratio=128 while Alpaca's median sample is ~53 tokens, so on packed instruction
data the HCA pool is empty for whole batches at a time.

The pool must then be zeros -- but zeros that still DEPEND on wkv_gate / ape / kv_norm.
A detached ``new_zeros`` has the identical value and shape, and is wrong in a way nothing
local can see: those parameters simply receive no gradient. Megatron's DDP catches it one
iteration later, via an assert that names no parameter and points at its own grad buffer;
without that check it would be entirely silent, and the layer would never train.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")))

from primus.backends.megatron.core.transformer.compressor import (  # noqa: E402
    Compressor,
)


def _all_shorter_than(ratio):
    """A packed layout in which no segment reaches one full window."""
    lens = [ratio - 1, max(1, ratio // 2), ratio - 2, 1]
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int64)
    return cu, int(cu[-1])


@pytest.mark.parametrize("ratio", [4, 128])
def test_empty_pool_still_propagates_gradient(ratio):
    C = 32
    cu, S = _all_shorter_than(ratio)

    torch.manual_seed(0)
    comp = Compressor(hidden_size=C, head_dim=16, ratio=ratio).double()
    hidden = torch.randn(1, S, C, dtype=torch.float64, requires_grad=True)

    # Precondition: this layout really does yield no windows, or the test proves nothing.
    row_idx, comp_ids, _ = comp.thd_compact_plan(cu, 0, S)
    assert row_idx is None, (
        f"ratio={ratio}: expected an empty pool for lengths all below the ratio, but the "
        f"plan produced windows -- pick shorter segments"
    )

    pooled = comp(hidden, cu_seqlens=cu, global_start=0)
    assert pooled.shape == (1, comp.thd_capacity(S), comp.head_dim)
    assert torch.count_nonzero(pooled) == 0, "an empty pool must be all zeros"

    pooled.sum().backward()

    # The value is zero, so every gradient is zero -- the point is that they EXIST.
    for name, p in comp.named_parameters():
        assert p.grad is not None, (
            f"ratio={ratio}: {name} got no gradient from an empty pool, so it is detached "
            f"from the graph and will never train on batches like this one"
        )
        assert torch.isfinite(p.grad).all(), f"{name} gradient is not finite"


@pytest.mark.parametrize("ratio", [4, 128])
def test_empty_pool_value_matches_plain_zeros(ratio):
    """Keeping the graph must not change the numbers."""
    C = 32
    cu, S = _all_shorter_than(ratio)
    torch.manual_seed(0)
    comp = Compressor(hidden_size=C, head_dim=16, ratio=ratio).double()
    hidden = torch.randn(1, S, C, dtype=torch.float64)

    pooled = comp(hidden, cu_seqlens=cu, global_start=0)
    torch.testing.assert_close(
        pooled,
        torch.zeros_like(pooled),
        rtol=0,
        atol=0,
        msg=lambda m: f"empty pool is no longer exactly zero:\n{m}",
    )
