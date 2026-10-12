###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Mask and offset metadata, checked bitwise against the rule they encode.

Two things are pinned here.

**The mask itself.** ``compressed_causal_mask`` is built with broadcast arange arithmetic,
which is easy to get right and equally easy to get off by one in a way that no loss curve
would reveal -- an extra visible column leaks one window of future tokens, a missing one
quietly shortens the context. The reference is the rule written out as a double loop.

**The offset identity the streaming top-K rests on.** Scoring pool columns
``[p_off, p_off + Pc)`` is claimed to be the same as scoring columns ``[0, Pc)`` with the
query origin shifted by ``p_off * ratio``:

    (p_global + 1) * ratio - 1 <= t_global
    (p_local + p_off + 1) * ratio - 1 <= t_global
    (p_local + 1) * ratio - 1 <= t_global - p_off * ratio

That substitution is why the chunked path can reuse ``q_offset`` and needs no
chunk-awareness inside the Triton kernels. It is an algebraic claim carrying real weight:
if it were false, chunked scoring would silently apply the wrong mask to every chunk but
the first, and the only symptom would be a model conditioned on the wrong history.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")))

import primus.backends.megatron.core.models.deepseek_v4  # noqa: E402,F401
from primus.backends.megatron.core.transformer.compressor import (  # noqa: E402
    Compressor,
)
from primus.backends.megatron.core.transformer.deepseek_v4_cp import (  # noqa: E402
    compressed_causal_mask,
)
from primus.backends.megatron.core.transformer.indexer import Indexer  # noqa: E402

NEG = float("-inf")


def _reference_mask(s_local, p_global, global_start, ratio):
    """The rule as a double loop: query t may attend to slot s once s's window has closed."""
    m = torch.empty(s_local, p_global, dtype=torch.float32)
    for row in range(s_local):
        t = global_start + row
        for s in range(p_global):
            m[row, s] = 0.0 if (s + 1) * ratio - 1 <= t else NEG
    return m


@pytest.mark.parametrize("ratio", [1, 4, 128])
@pytest.mark.parametrize("global_start", [0, 1, 7, 512])
def test_compressed_causal_mask_matches_loop_reference(ratio, global_start):
    s_local, p_global = 24, 9
    got = compressed_causal_mask(s_local, p_global, global_start, ratio, device="cpu", dtype=torch.float32)
    want = _reference_mask(s_local, p_global, global_start, ratio)
    assert torch.equal(got, want), (
        f"ratio={ratio} global_start={global_start}: mask differs from the rule.\n"
        f"first differing row: "
        f"{int((got != want).any(dim=1).nonzero()[0]) if (got != want).any() else '-'}"
    )


def test_compressed_causal_mask_is_monotone_in_global_start():
    """A query further along the sequence can never see FEWER compressed slots."""
    ratio, s_local, p_global = 4, 8, 16
    prev = None
    for gs in range(0, 40, 4):
        m = compressed_causal_mask(s_local, p_global, gs, ratio, device="cpu", dtype=torch.float32)
        visible = torch.isfinite(m).sum(dim=1)
        if prev is not None:
            assert bool(
                (visible >= prev).all()
            ), f"global_start={gs}: some query sees fewer slots than at global_start={gs - 4}"
        prev = visible


@pytest.mark.parametrize("ratio", [4, 128])
@pytest.mark.parametrize("p_off", [0, 1, 3, 8])
def test_pool_column_offset_equals_query_offset_shift(ratio, p_off):
    """The substitution the streaming top-K depends on, checked bitwise.

    Slicing the full mask at columns ``[p_off, p_off + Pc)`` must equal building a mask of
    width ``Pc`` with the query origin moved back by ``p_off * ratio``.
    """
    S, P, Pc = 40, 16, 5
    if p_off + Pc > P:
        pytest.skip("chunk does not fit in the pool")

    idx = Indexer(hidden_size=8, index_head_dim=8, index_n_heads=1, index_topk=2, compress_ratio=ratio)
    q_base = idx._cp_query_offset(S)

    full = idx._causal_mask(S, P, torch.device("cpu"), torch.float32, q_offset=q_base)
    sliced = full[:, p_off : p_off + Pc]
    chunked = idx._causal_mask(S, Pc, torch.device("cpu"), torch.float32, q_offset=q_base - p_off * ratio)

    assert torch.equal(chunked, sliced), (
        f"ratio={ratio} p_off={p_off}: a pool-column offset is NOT equivalent to shifting "
        f"the query origin, so chunked scoring applies a different mask than one-shot "
        f"scoring for every chunk after the first."
    )


@pytest.mark.parametrize("ratio", [4, 128])
def test_chunked_mask_reassembles_the_full_mask(ratio):
    """Walking the pool in chunks must rebuild the one-shot mask exactly."""
    S, P, chunk = 40, 16, 5
    idx = Indexer(hidden_size=8, index_head_dim=8, index_n_heads=1, index_topk=2, compress_ratio=ratio)
    q_base = idx._cp_query_offset(S)

    full = idx._causal_mask(S, P, torch.device("cpu"), torch.float32, q_offset=q_base)
    parts = [
        idx._causal_mask(
            S,
            min(lo + chunk, P) - lo,
            torch.device("cpu"),
            torch.float32,
            q_offset=q_base - lo * ratio,
        )
        for lo in range(0, P, chunk)
    ]
    assert torch.equal(torch.cat(parts, dim=1), full)


@pytest.mark.skipif(
    not hasattr(Compressor, "thd_capacity"),
    reason="packed (THD) pooling not present on this revision",
)
@pytest.mark.parametrize("ratio", [4, 128])
def test_thd_mask_chunk_equals_slice_of_full(ratio):
    """Same property for the PACKED mask, which cannot use the offset shift.

    The packed mask keys on each column's ``(seq_id, comp_id)`` rather than on a scalar
    origin, so the streaming path slices the identity instead of shifting the query. That
    slice has to agree with the full mask column for column.
    """
    lens = [37, 19, 51, 21]
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int64)
    S = int(cu[-1])
    idx = Indexer(hidden_size=8, index_head_dim=8, index_n_heads=1, index_topk=2, compress_ratio=ratio)
    P = idx.indexer_compressor.thd_capacity(S)

    full = idx._thd_causal_mask(S, P, cu, torch.device("cpu"), torch.float32)
    chunk = 8
    parts = [
        idx._thd_causal_mask(S, P, cu, torch.device("cpu"), torch.float32, p_slice=(lo, min(lo + chunk, P)))
        for lo in range(0, P, chunk)
    ]
    assert torch.equal(
        torch.cat(parts, dim=1), full
    ), f"ratio={ratio}: the packed mask built per chunk differs from the full mask"
