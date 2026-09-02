###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The indexer's selected columns must equal a pure-Python top-K of its own scores.

Every other test of the selection compares one implementation against another one in the
same file -- chunked against unchunked, fused against eager. Those catch drift between
paths but would all agree if the shared scoring formula were wrong. This re-derives the
score in plain Python from the module's weights,

    I[t, s] = sum_h w[t, h] * relu(q[t, h] . k[s])          (masked to causal positions)

picks the top-K with ``sorted``, and asserts the indices match bitwise.

Comparing INDICES only makes sense when the top-K scores are distinct, which ReLU makes
untrue in general -- it drives a large fraction of the scores to exactly 0.0, and which of
several zero-scoring columns ``torch.topk`` returns is arbitrary. The fixture below is
built so that every selected score is strictly positive, and asserts that precondition
rather than assuming it.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")))

import primus.backends.megatron.core.models.deepseek_v4  # noqa: E402,F401
from primus.backends.megatron.core.transformer.indexer import Indexer  # noqa: E402


def _python_scores(idx, hidden, cu=None):
    """Recompute [S, P] indexer scores in plain Python from the module's own weights."""
    with torch.no_grad():
        k_pool = (
            idx.indexer_compressor(hidden, cu_seqlens=cu)
            if cu is not None
            else idx.indexer_compressor(hidden)
        )
        # w_dq and w_w are fused into one GEMM by default; split it the same way.
        if idx._fuse_qw_proj:
            dqw = idx.w_dq_w(hidden)
            q_lr, w = dqw[..., : idx.dq_rank], dqw[..., idx.dq_rank :]
        else:
            q_lr, w = idx.w_dq(hidden), idx.w_w(hidden)
        q = idx.w_iuq(q_lr)

    S = hidden.shape[1]
    H, Hd = idx.index_n_heads, idx.index_head_dim
    P = k_pool.shape[1]
    q = q.reshape(1, S, H, Hd)
    w = w.reshape(1, S, H)

    scores = [[0.0] * P for _ in range(S)]
    for t in range(S):
        for s in range(P):
            acc = 0.0
            for h in range(H):
                dot = float(torch.dot(q[0, t, h], k_pool[0, s]))
                acc += float(w[0, t, h]) * (dot if dot > 0.0 else 0.0)
            scores[t][s] = acc
    return scores, P


def _python_topk(scores, mask_ok, K):
    """Reference selection: sorted() over the visible columns, highest score first."""
    out = []
    for t, row in enumerate(scores):
        cand = [(row[s], s) for s in range(len(row)) if mask_ok(t, s)]
        cand.sort(key=lambda v: (-v[0], v[1]))
        out.append([s for _, s in cand[:K]])
    return out


@pytest.mark.parametrize("topk", [2, 4])
def test_contiguous_topk_matches_python_reference(topk):
    C, ratio, H, Hd, S = 32, 4, 2, 8, 32
    torch.manual_seed(3)
    idx = Indexer(
        hidden_size=C, index_head_dim=Hd, index_n_heads=H, index_topk=topk, compress_ratio=ratio
    ).double()
    hidden = torch.randn(1, S, C, dtype=torch.float64)

    scores, P = _python_scores(idx, hidden)

    # Causal rule, stated independently of the module: pool slot s covers raw tokens
    # [s*ratio, (s+1)*ratio), so query t may attend to it once the window has closed.
    def visible(t, s):
        return (s + 1) * ratio - 1 <= t

    expected = _python_topk(scores, visible, topk)

    # Precondition: the comparison is only well-posed where the selected scores are
    # distinct and nonzero. Restrict to the queries where that holds, and require that
    # there are enough of them for the test to mean something.
    usable = []
    for t in range(S):
        vals = [scores[t][s] for s in expected[t]]
        if len(vals) == topk and all(v > 0 for v in vals) and len(set(vals)) == topk:
            usable.append(t)
    assert len(usable) >= topk, (
        f"only {len(usable)} queries have {topk} distinct positive scores; the fixture no "
        f"longer exercises a well-posed index comparison"
    )

    with torch.no_grad():
        got, _ = idx(hidden)

    for t in usable:
        assert torch.equal(
            torch.tensor(sorted(got[0, t].tolist())),
            torch.tensor(sorted(expected[t])),
        ), (
            f"query {t}: indexer selected {sorted(got[0, t].tolist())}, pure-Python top-{topk} "
            f"of the same scores is {sorted(expected[t])}"
        )


def test_topk_marks_unavailable_slots_with_minus_one():
    """Early queries cannot fill index_topk, and the shortfall must be -1, not slot 0.

    A query at position 0 has no closed compression window yet. Returning a real column
    number there would make it attend to a key built from tokens it has not seen -- and
    since the sparse-MLA kernel honours -1 by zeroing the value and forcing the score to
    -inf, the sentinel is the mechanism, not a cosmetic marker.
    """
    C, ratio, topk, S = 32, 4, 8, 16
    torch.manual_seed(0)
    idx = Indexer(
        hidden_size=C, index_head_dim=8, index_n_heads=2, index_topk=topk, compress_ratio=ratio
    ).double()
    hidden = torch.randn(1, S, C, dtype=torch.float64)

    with torch.no_grad():
        got, _ = idx(hidden)

    for t in range(S):
        n_visible = min(topk, max(0, (t + 1) // ratio))
        row = got[0, t].tolist()
        n_real = sum(1 for c in row if c >= 0)
        assert n_real == n_visible, (
            f"query {t}: {n_real} real columns selected, but only {n_visible} compression "
            f"windows have closed by then"
        )
        for c in row:
            if c >= 0:
                assert (c + 1) * ratio - 1 <= t, f"query {t} selected future column {c}"
