###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for the CSA indexer score temperature.

The index score ``I_{t,s}`` carries a ``sqrt(index_n_heads * index_head_dim)``
temperature that the open-source reference applies in two places:
``index_n_heads ** -0.5`` when building the per-head weights, plus the indexer's
own ``index_head_dim ** -0.5`` on the way into the distillation loss.

The scale is invisible to ``topk`` -- which is why it went missing -- and only
becomes load-bearing once the scores enter a softmax. Both properties are
pinned here: the scale must be exactly the reference value, it must not perturb
the selection, and it must actually change the softmax it feeds.
"""

from __future__ import annotations

import copy
import math

import pytest

torch = pytest.importorskip("torch")

from primus.backends.megatron.core.transformer.indexer import Indexer  # noqa: E402

_HIDDEN = 64
_HEAD_DIM = 16
_N_HEADS = 4
_TOPK = 4
_RATIO = 4


def _make_indexer() -> Indexer:
    return Indexer(
        hidden_size=_HIDDEN,
        index_head_dim=_HEAD_DIM,
        index_n_heads=_N_HEADS,
        index_topk=_TOPK,
        compress_ratio=_RATIO,
    )


def test_score_scale_matches_reference_formula():
    """``score_scale == (index_n_heads * index_head_dim) ** -0.5``."""
    indexer = _make_indexer()
    expected = (_N_HEADS**-0.5) * (_HEAD_DIM**-0.5)
    assert indexer.score_scale == pytest.approx(expected, rel=1e-12)
    # Guard against the assertion going vacuous on a degenerate config.
    assert expected < 0.5, "pick a config where the scale is clearly not 1"


def test_scale_is_a_pure_constant_on_the_scores():
    """Scaling only changes the temperature, never the relative scores.

    Runs the same weights twice, once with the scale neutralised, and checks
    every finite score differs by exactly the constant.
    """
    torch.manual_seed(0)
    scaled = _make_indexer()
    unscaled = _make_indexer()
    unscaled.load_state_dict(copy.deepcopy(scaled.state_dict()))
    unscaled.score_scale = 1.0

    hidden = torch.randn(2, 32, _HIDDEN, dtype=torch.float32)
    with torch.no_grad():
        idx_scaled, scores_scaled = scaled(hidden)
        idx_unscaled, scores_unscaled = unscaled(hidden)

    # top-k selection is invariant under a positive constant.
    assert torch.equal(idx_scaled, idx_unscaled)

    finite = torch.isfinite(scores_scaled)
    assert finite.any(), "no finite scores to compare"
    torch.testing.assert_close(
        scores_scaled[finite],
        scores_unscaled[finite] * scaled.score_scale,
        rtol=1e-5,
        atol=1e-7,
    )


def test_scale_changes_the_softmax_it_feeds():
    """The distillation loss softmaxes these scores, so the scale must bite.

    Without the scale the distribution is ~90x too sharp at the V4 widths; a
    smaller config still has to show a clearly less peaked distribution.
    """
    torch.manual_seed(0)
    indexer = _make_indexer()
    hidden = torch.randn(1, 32, _HIDDEN, dtype=torch.float32)
    with torch.no_grad():
        _, scores = indexer(hidden)

    finite_rows = torch.isfinite(scores).all(dim=-1)
    assert finite_rows.any(), "need at least one fully-valid row"
    rows = scores[finite_rows].float()

    peaked = torch.softmax(rows / indexer.score_scale, dim=-1).max(dim=-1).values
    tempered = torch.softmax(rows, dim=-1).max(dim=-1).values

    assert (tempered <= peaked + 1e-6).all()
    assert tempered.mean() < peaked.mean(), "the scale must soften the distribution"


def test_sentinel_semantics_survive_the_scale():
    """Masked slots stay ``-inf`` / ``-1``: the scale is applied pre-mask."""
    torch.manual_seed(0)
    indexer = _make_indexer()
    # S small enough that early queries have fewer than topk legal entries.
    hidden = torch.randn(1, 8, _HIDDEN, dtype=torch.float32)
    with torch.no_grad():
        idxs, scores = indexer(hidden)

    invalid = idxs < 0
    assert invalid.any(), "expected some early-query sentinels in this shape"
    assert torch.isneginf(scores[invalid]).all()
    assert torch.isfinite(scores[~invalid]).all()


def test_scale_survives_the_fused_and_unfused_projection_layouts(monkeypatch):
    """``w_dq_w`` (fused) and ``w_dq`` + ``w_w`` must land on the same scores."""
    torch.manual_seed(0)
    monkeypatch.setenv("PRIMUS_INDEXER_FUSE_PROJ", "1")
    fused = _make_indexer()
    monkeypatch.setenv("PRIMUS_INDEXER_FUSE_PROJ", "0")
    split = _make_indexer()

    # ``_load_from_state_dict`` bridges the two layouts.
    split.load_state_dict(copy.deepcopy(fused.state_dict()))

    hidden = torch.randn(1, 16, _HIDDEN, dtype=torch.float32)
    with torch.no_grad():
        idx_fused, scores_fused = fused(hidden)
        idx_split, scores_split = split(hidden)

    assert torch.equal(idx_fused, idx_split)
    finite = torch.isfinite(scores_fused)
    torch.testing.assert_close(scores_fused[finite], scores_split[finite], rtol=1e-5, atol=1e-6)


def test_scale_is_not_double_counted_against_a_hand_computed_reference():
    """Pin the whole scoring chain against an explicit reference expression."""
    torch.manual_seed(0)
    indexer = _make_indexer()
    hidden = torch.randn(1, 16, _HIDDEN, dtype=torch.float32)

    with torch.no_grad():
        _, scores = indexer(hidden)

        k_icomp = indexer.indexer_compressor(hidden)  # [B, P, Hd]
        if indexer._fuse_qw_proj:
            dqw = indexer.w_dq_w(hidden)
            q_q = dqw[..., : indexer.dq_rank]
            w_i = dqw[..., indexer.dq_rank :]
        else:
            q_q = indexer.w_dq(hidden)
            w_i = indexer.w_w(hidden)
        B, S, _ = hidden.shape
        q_i = indexer.w_iuq(q_q).view(B, S, _N_HEADS, _HEAD_DIM)

        dot = torch.einsum("bshd,bpd->bshp", q_i, k_icomp)
        expected = (torch.relu(dot) * w_i.unsqueeze(-1)).sum(dim=2) * ((_N_HEADS * _HEAD_DIM) ** -0.5)
        P = k_icomp.shape[1]
        mask = indexer._causal_mask(S, P, expected.device, expected.dtype)
        expected = expected + mask.unsqueeze(0)

        topk_eff = min(_TOPK, P)
        expected_topk = expected.topk(topk_eff, dim=-1).values

    finite = torch.isfinite(expected_topk)
    torch.testing.assert_close(scores[..., :topk_eff][finite], expected_topk[finite], rtol=1e-5, atol=1e-6)
    # Sanity: the reference expression is not accidentally scale-free.
    assert math.isclose(indexer.score_scale, (_N_HEADS * _HEAD_DIM) ** -0.5, rel_tol=1e-12)
