###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The packed indexer's streaming top-K must select exactly what one-shot scoring does.

The packed path cannot materialise ``[B, S, P]``: P is the GLOBAL compressed pool, so it
grows with the packed segment count, and holding a full row (plus its ``[S, P]`` mask) is
what capped 128k packing at 256 segments per window -- roughly 9% supervised tokens, which
defeats the purpose of packing. So the row is consumed in chunks with a running top-K.

Chunked selection is exact only because top-K of the per-chunk top-Ks is the global top-K.
That is easy to get subtly wrong -- forgetting to offset chunk-local column indices to
global ones, or letting a chunk's ``-inf`` masked entries outrank real scores from another
chunk -- and every such error is silent: the model still trains, just conditioned on the
wrong compressed history. This pins the selection against single-chunk scoring, which is
the same arithmetic without the chunking.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")))

# Import the model package FIRST. _thd_causal_mask imports deepseek_v4_attention lazily,
# and deepseek_v4_attention <-> deepseek_v4_block is a real import cycle: whichever is
# entered first wins, so triggering it from inside a method fails when this file happens
# to be the first V4 test collected. Importing the package here resolves the cycle once.
import primus.backends.megatron.core.models.deepseek_v4  # noqa: E402,F401
from primus.backends.megatron.core.transformer.indexer import Indexer  # noqa: E402


def _run(indexer, hidden, cu, chunk):
    prev = os.environ.get("PRIMUS_INDEXER_TOPK_CHUNK")
    os.environ["PRIMUS_INDEXER_TOPK_CHUNK"] = str(chunk)
    try:
        torch.manual_seed(0)
        with torch.no_grad():
            return indexer(hidden, cu)  # (topk_idxs, topk_scores)
    finally:
        if prev is None:
            os.environ.pop("PRIMUS_INDEXER_TOPK_CHUNK", None)
        else:
            os.environ["PRIMUS_INDEXER_TOPK_CHUNK"] = prev


@pytest.mark.parametrize("chunk", [8, 16, 32])
@pytest.mark.parametrize("lens", [[64, 32, 96, 64], [37, 19, 51, 21]])
def test_streaming_topk_matches_single_chunk(chunk, lens):
    """Chunked selection == unchunked selection, for the same scores."""
    C, ratio, topk = 64, 4, 8
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int64)
    S = int(cu[-1])

    torch.manual_seed(0)
    idx = Indexer(
        hidden_size=C,
        index_head_dim=16,
        index_n_heads=4,
        index_topk=topk,
        compress_ratio=ratio,
    ).double()
    hidden = torch.randn(1, S, C, dtype=torch.float64)

    ref_i, ref_s = _run(idx, hidden, cu, 10**9)  # one chunk: the unchunked reference
    got_i, got_s = _run(idx, hidden, cu, chunk)

    assert got_i.shape == ref_i.shape

    # Compare the selected SCORES, not the selected column numbers.
    #
    # Column numbers are not a well-defined output here: ReLU drives a large fraction of
    # the scores to exactly 0.0, so ties are the norm rather than the exception, and which
    # of several equal-scoring columns top-k returns is arbitrary in both paths. Comparing
    # indices would fail on ties that are genuinely the same model. Scores are the thing
    # the selection is defined by, and comparing them is STRICTER, not weaker: a
    # chunk-local index never offset to a global column, or a leaked -inf outranking a
    # real score, both change the score vector.
    torch.testing.assert_close(
        got_s,
        ref_s,
        rtol=0,
        atol=0,
        msg=lambda m: f"chunk={chunk}: selected scores differ from unchunked scoring:\n{m}",
    )

    # ...and the indices must still POINT at those scores -- equal score vectors with
    # mismatched indices would mean the merge lost track of which column it kept.
    P = idx.indexer_compressor.thd_capacity(S)
    mask = idx._thd_causal_mask(S, P, cu, hidden.device, torch.float64)
    for q in range(ref_i.shape[1]):
        for c, sc in zip(got_i[0, q].tolist(), got_s[0, q].tolist()):
            if c < 0:
                continue
            assert torch.isfinite(mask[q, c]) or sc == float(
                "-inf"
            ), f"chunk={chunk} query {q}: column {c} scored {sc} but the mask forbids it"


def test_selected_columns_are_visible_to_attention():
    """Every selected column must be one the attention will actually honour.

    The indexer's top-K addresses the same pool the attention masks, so a column that is
    selected but invisible is a wasted top-K slot -- the query silently attends to fewer
    keys than index_topk. Zero-length or cross-sequence columns are the ways that happens.
    """
    C, ratio, topk = 64, 4, 8
    lens = [37, 19, 51, 21]
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int64)
    S = int(cu[-1])

    torch.manual_seed(0)
    idx = Indexer(
        hidden_size=C, index_head_dim=16, index_n_heads=4, index_topk=topk, compress_ratio=ratio
    ).double()
    hidden = torch.randn(1, S, C, dtype=torch.float64)
    sel, sel_s = _run(idx, hidden, cu, 16)

    P = idx.indexer_compressor.thd_capacity(S)
    mask = idx._thd_causal_mask(S, P, cu, hidden.device, torch.float64)  # [S, P]
    for q in range(S):
        for c, sc in zip(sel[0, q].tolist(), sel_s[0, q].tolist()):
            if c < 0 or sc == float("-inf"):
                continue  # padding slot: the query has fewer than K visible columns
            assert torch.isfinite(mask[q, c]), (
                f"query {q} selected pool column {c}, which the attention mask forbids "
                f"(different sequence, or not yet complete at this position)"
            )
