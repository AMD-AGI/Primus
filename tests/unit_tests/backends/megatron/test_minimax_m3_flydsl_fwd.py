###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The flydsl MSA forward against the eager backend it has to match.

The eager backend is itself pinned to transformers' MiniMax-M3 reference (see
test_minimax_m3_msa.py), so matching it here is matching the reference. The
block tables are built the way the indexer builds them -- causal, own block
forced, top-k by score, -1 padded -- rather than taken from the indexer, so the
kernel is tested on exactly the contract it consumes.
"""

import pytest
import torch

pytest.importorskip("megatron")
pytest.importorskip("flydsl")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName,
    reason="the flydsl MSA kernel targets gfx950",
)

BLOCK, D = 128, 128


def _table(B, Hkv, S, topk, gen):
    n_blocks = -(-S // BLOCK)
    scores = torch.randn(B, Hkv, S, n_blocks, device="cuda", generator=gen)
    qb = torch.arange(S, device="cuda") // BLOCK
    future = torch.arange(n_blocks, device="cuda")[None, :] > qb[:, None]
    scores = scores.masked_fill(future, float("-inf"))
    scores.scatter_(-1, qb.view(1, 1, S, 1).expand(B, Hkv, S, 1), float("inf"))
    k = min(topk, n_blocks)
    vals, idx = scores.topk(k, dim=-1)
    idx = idx.masked_fill(vals == float("-inf"), -1)
    if k < topk:
        idx = torch.nn.functional.pad(idx, (0, topk - k), value=-1)
    return idx.to(torch.int32).contiguous()


def _inputs(S, B, Hkv, topk, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(S, B, 16 * Hkv, D, device="cuda", dtype=torch.bfloat16, generator=gen)
    k = torch.randn(S, B, Hkv, D, device="cuda", dtype=torch.bfloat16, generator=gen)
    v = torch.randn(S, B, Hkv, D, device="cuda", dtype=torch.bfloat16, generator=gen)
    return q, k, v, _table(B, Hkv, S, topk, gen)


def _reference(q, k, v, table):
    from primus.backends.megatron.core.transformer.minimax_m3.eager import (
        build_block_keep,
        eager_block_sparse_attention,
    )

    S, Hq, Hkv = q.shape[0], q.shape[2], k.shape[2]
    qb, kb, vb = (t.permute(1, 2, 0, 3) for t in (q, k, v))
    keep = build_block_keep(table.long(), S, BLOCK)
    out, dense = eager_block_sparse_attention(qb, kb, vb, keep, D**-0.5)
    lse = torch.logsumexp(dense.masked_fill(~keep.repeat_interleave(Hq // Hkv, dim=1), float("-inf")), dim=-1)
    return out, lse


def _check(o, lse, out_ref, lse_ref):
    # One bf16 ulp at |o| < 4: both sides round P and O to bf16.
    torch.testing.assert_close(o.permute(1, 2, 0, 3).float(), out_ref.float(), rtol=0, atol=1.6e-2)
    torch.testing.assert_close(lse.permute(1, 2, 0), lse_ref, rtol=0, atol=1e-4)


@pytest.mark.parametrize(
    "S, B, Hkv, topk",
    [
        (300, 1, 1, 16),  # partial last block, fewer blocks than topk (-1 slots)
        (1000, 1, 4, 4),  # topk below the visible block count
        (1000, 2, 2, 16),  # batch > 1
        (2148, 1, 4, 16),
        (4096, 2, 4, 8),
    ],
)
def test_matches_eager(S, B, Hkv, topk):
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_token_fwd import (
        msa_token_fwd,
    )

    q, k, v, table = _inputs(S, B, Hkv, topk)
    o, lse = msa_token_fwd(q, k, v, table)
    _check(o, lse, *_reference(q, k, v, table))


@pytest.mark.parametrize(
    "config",
    [
        dict(pipeline=False, barriers=True),
        dict(step_keys=32, pipeline=False),
        dict(k_via_lds=False, pipeline=False),
        dict(remap="block"),
    ],
)
def test_every_tuning_path_matches_eager(config):
    """The knobs change the schedule, never the maths."""
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_token_fwd import (
        msa_token_fwd,
    )

    q, k, v, table = _inputs(2048, 1, 4, 16, seed=1)
    o, lse = msa_token_fwd(q, k, v, table, **config)
    _check(o, lse, *_reference(q, k, v, table))
