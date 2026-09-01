###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Primus's compact window plan must agree with the reference ownership rule.

Dropping the sequence-alignment requirement means a pooling window can straddle a CP
shard boundary, and ownership has to be decided without communication. The rule -- a
window belongs to the rank holding its LAST row -- comes from upstream's
``_compressed_groups``; this pins that Primus's vectorised reimplementation selects
exactly the same windows, over ragged layouts and every shard offset.

Getting this wrong is silent: a window owned by nobody drops compressed history, one
owned by two ranks double-counts it, and neither shows up as an error.
"""

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")))

from primus.backends.megatron.core.transformer.compressor import (  # noqa: E402
    Compressor,
)


def _reference_groups(cu, global_start, l_local, ratio, d_comp):
    """Upstream _compressed_groups, transcribed."""
    cu = [int(x) for x in cu.tolist()]
    groups = []
    g_end = global_start + l_local
    for seq, (s, e) in enumerate(zip(cu[:-1], cu[1:])):
        lo, hi = max(s, global_start), min(e, g_end)
        if lo >= hi:
            continue
        n_full = (e - s) // ratio
        first_numer = max(0, global_start - d_comp - s)
        first = (first_numer + ratio - 1) // ratio if first_numer > 0 else 0
        stop = min((hi - s) // ratio, n_full)
        for comp_id in range(first, max(first, stop)):
            groups.append((seq, comp_id))
    return groups


def _plan(ratio, cu, global_start, l_local):
    c = Compressor(hidden_size=8, head_dim=8, ratio=ratio)
    _, comp_ids, seq_ids = c.thd_compact_plan(cu, global_start, l_local)
    keep = comp_ids >= 0
    return list(zip(seq_ids[keep].tolist(), comp_ids[keep].tolist())), c


@pytest.mark.parametrize("ratio", [4, 8, 128])
@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_ownership_matches_reference(ratio, cp_size):
    """Every window owned exactly once across all ranks, and the same set as upstream."""
    torch.manual_seed(ratio * 100 + cp_size)
    lens = [int(x) for x in torch.randint(1, 6 * ratio, (24,))]
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int64)
    total = int(cu[-1])
    l_local = math.ceil(total / cp_size)
    total_padded = l_local * cp_size
    cu[-1] = max(int(cu[-1]), total_padded)  # last sequence absorbs the tail, as in packing

    owned, comp = [], None
    for r in range(cp_size):
        got, comp = _plan(ratio, cu, r * l_local, l_local)
        owned.extend(got)

    ref = _reference_groups(cu, 0, l_local * cp_size, ratio, comp.thd_d_comp)
    assert sorted(owned) == sorted(ref), (
        f"ownership differs (ratio={ratio}, cp={cp_size}): "
        f"missing={sorted(set(ref) - set(owned))[:5]} extra={sorted(set(owned) - set(ref))[:5]}"
    )
    assert len(owned) == len(set(owned)), "a window is owned by more than one rank"


@pytest.mark.parametrize("ratio", [4, 128])
def test_capacity_is_never_exceeded(ratio):
    """The fixed c_cap must bound the owned count for every shard offset."""
    c = Compressor(hidden_size=8, head_dim=8, ratio=ratio)
    torch.manual_seed(7)
    for trial in range(50):
        lens = [int(x) for x in torch.randint(1, 5 * ratio, (16,))]
        cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int64)
        total = int(cu[-1])
        for cp in (1, 2, 4):
            l_local = math.ceil(total / cp)
            cu2 = cu.clone()
            cu2[-1] = max(int(cu2[-1]), l_local * cp)
            cap = c.thd_capacity(l_local)
            for r in range(cp):
                got, _ = _plan(ratio, cu2, r * l_local, l_local)
                assert len(got) <= cap, f"trial {trial} cp={cp} rank={r}: {len(got)} windows > capacity {cap}"


def test_ragged_layout_gives_ranks_different_window_counts():
    """Guard the guard: the CP equivalence check is only meaningful if the ranks actually
    own different numbers of windows.

    With lengths [333, 191, 277, 223] the counts happen to come out EQUAL at cp=2
    (127/127 and 3/3) and only differ at cp=4 (64/63/64/63 and 2/1/1/2). A cp=2-only
    equivalence run therefore passes without ever exercising the fixed-capacity path --
    which is exactly the trap this pins shut.
    """
    lens = [333, 191, 277, 223]
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int64)
    total = int(cu[-1])
    for ratio in (4, 128):
        c = Compressor(hidden_size=8, head_dim=8, ratio=ratio)
        L = total // 4
        counts = [int((c.thd_compact_plan(cu, r * L, L)[1] >= 0).sum().item()) for r in range(4)]
        assert len(set(counts)) > 1, (
            f"ratio={ratio}: cp=4 counts {counts} are uniform, so this layout no longer "
            f"exercises the ragged path -- pick different lengths."
        )
        assert max(counts) <= c.thd_capacity(L)


def test_capacity_is_at_least_one():
    """A rank owning zero windows must still contribute a slot, or the pool all-gather
    has mismatched participants and the job hangs instead of failing."""
    for ratio in (4, 128):
        c = Compressor(hidden_size=8, head_dim=8, ratio=ratio)
        assert c.thd_capacity(1) >= 1


def test_d_comp_is_larger_for_overlap():
    """Overlap (CSA) needs the predecessor window too, so twice the lookahead."""
    assert Compressor(hidden_size=8, head_dim=8, ratio=4).thd_d_comp == 8  # overlap
    assert Compressor(hidden_size=8, head_dim=8, ratio=128).thd_d_comp == 128  # non-overlap
