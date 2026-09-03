###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CSA's overlap stitch must survive a context-parallel shard boundary.

The compressor pools `ratio` rows into one compressed key, and in overlap mode (CSA,
ratio=4) window `i` is stitched with window `i-1`. `_overlap_transform` takes that
predecessor from the PRECEDING SLOT of the compact buffer, and slot 0's predecessor is
hardcoded zeros -- correct only when slot 0 really is a sequence's first window.

Under CP it usually is not: a rank whose first owned window is window k>0 of some
sequence has its predecessor on the left neighbour. Both regressions pinned here made
that predecessor vanish, and both are invisible without a cross-CP comparison, because
each rank's output is perfectly self-consistent -- it is simply stitched to zeros.

Neither shows up as an error, a shape mismatch, or a NaN. They show up as one compressed
key per rank per layer being ~100% wrong, forever.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")))

from primus.backends.megatron.core.transformer.compressor import (  # noqa: E402
    Compressor,
)
from primus.backends.megatron.core.transformer.deepseek_v4_cp import (  # noqa: E402
    compressor_boundary_rows,
)

# The pooling runs a GEMM whose shape differs between the sharded and unsharded runs, and
# LocalRMSNorm accumulates in fp32, so even a bit-perfect algorithm leaves ~1e-8 here. The
# defect being pinned is O(1) -- a zeroed predecessor -- so this threshold separates them
# by seven orders of magnitude.
TOL = 1e-6


def _pool_thd(comp, cu, hidden, global_start, l_local, d_window):
    """One rank's THD pool, with the left-neighbour rows it is entitled to."""
    local = hidden[:, global_start : global_start + l_local]
    lo = global_start - d_window
    bnd = (
        hidden[:, lo:global_start]
        if lo >= 0
        else torch.cat([hidden.new_zeros(1, -lo, hidden.shape[2]), hidden[:, :global_start]], dim=1)
    )
    return comp(local, cu_seqlens=cu, global_start=global_start, boundary_hidden=bnd)


@pytest.mark.parametrize("cp_size", [2, 4, 8])
@pytest.mark.parametrize("lens", [[32], [1024], [13, 19, 32], [100, 156]])
def test_thd_cp_pool_matches_single_rank(cp_size, lens):
    """Every owned window must pool to the same value it does without CP.

    This is the regression for the missing overlap predecessor: before the shadow slot,
    the FIRST owned window of every rank whose first window was not a sequence start came
    out stitched to zeros, giving a relative error of order 1 on exactly one key per rank.
    """
    ratio, C = 4, 32
    total = sum(lens)
    if total % cp_size:
        pytest.skip(f"total {total} not divisible by cp_size {cp_size}")
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int64)

    torch.manual_seed(0)
    comp = Compressor(hidden_size=C, head_dim=16, ratio=ratio).double()
    assert comp.overlap, "this test is about the overlap stitch"
    hidden = torch.randn(1, total, C, dtype=torch.float64)
    d_window = compressor_boundary_rows(ratio, True)

    ref = _pool_thd(comp, cu, hidden, 0, total, d_window)
    _, ref_ids, ref_seq = comp.thd_compact_plan(cu, 0, total)
    want = {
        (int(s), int(c)): ref[0, j]
        for j, (s, c) in enumerate(zip(ref_seq.tolist(), ref_ids.tolist()))
        if c >= 0
    }

    l_local = total // cp_size
    seen = 0
    for r in range(cp_size):
        gs = r * l_local
        got = _pool_thd(comp, cu, hidden, gs, l_local, d_window)
        _, ids, seqs = comp.thd_compact_plan(cu, gs, l_local)
        assert got.shape[1] == ids.numel(), (
            "the emitted pool must stay exactly thd_capacity wide -- a shadow slot that "
            "leaks out desynchronises the all-gather and shifts every column number"
        )
        for j, (s, c) in enumerate(zip(seqs.tolist(), ids.tolist())):
            if c < 0:
                continue
            seen += 1
            d = (got[0, j] - want[(int(s), int(c))]).abs().max().item()
            assert d < TOL, (
                f"cp={cp_size} rank={r} slot={j} window=(seq {s}, comp {c}) differs from "
                f"the single-rank pool by {d:.3e}. slot 0 of a rank is the tell: its "
                f"overlap predecessor lives on the left neighbour."
            )
    assert seen == len(want), f"ranks together own {seen} windows, single rank owns {len(want)}"


@pytest.mark.parametrize("ratio,overlap", [(4, True), (8, False)])
def test_bshd_cp_pool_drops_exactly_the_prepended_windows(ratio, overlap):
    """Prepending `nb` rows creates `nb // ratio` extra windows -- drop that many.

    The BSHD CP path prepends the left neighbour's rows so the boundary window can be
    built, then slices the extra leading pool entries off. Dropping a fixed ONE was right
    only while ``compressor_boundary_rows`` returned ``ratio``; overlap now asks for
    ``2 * ratio``, and the stale slice left a duplicate of the neighbour's last window in
    every rank's contribution -- widening the global pool by cp_size, shifting the
    compressed RoPE phases (which are a bare ``arange(P)``) and letting one key be
    attended to twice.
    """
    C, S, cp_size = 32, 64, 2
    torch.manual_seed(0)
    comp = Compressor(hidden_size=C, head_dim=16, ratio=ratio).double()
    assert comp.overlap == overlap
    nb = compressor_boundary_rows(ratio, overlap)
    hidden = torch.randn(1, S, C, dtype=torch.float64)

    ref = comp(hidden)
    l_local = S // cp_size
    pooled = []
    for r in range(cp_size):
        gs = r * l_local
        bnd = (
            hidden[:, gs - nb : gs]
            if gs >= nb
            else torch.cat([hidden.new_zeros(1, nb - gs, C), hidden[:, :gs]], dim=1)
        )
        # Mirrors deepseek_v4_attention / indexer's CP branch.
        pooled.append(comp(torch.cat([bnd, hidden[:, gs : gs + l_local]], dim=1))[:, nb // ratio :])
    pooled = torch.cat(pooled, dim=1)

    assert pooled.shape[1] == S // ratio, (
        f"global pool has {pooled.shape[1]} columns, expected {S // ratio}; "
        f"{nb // ratio} leading windows are produced by the prepend and all must be dropped"
    )
    # Rank 0 has no real left neighbour, so only ranks > 0 are comparable.
    d = (pooled[:, l_local // ratio :] - ref[:, l_local // ratio :]).abs().max().item()
    assert d < TOL, f"sharded pool differs from the single-rank pool by {d:.3e}"
