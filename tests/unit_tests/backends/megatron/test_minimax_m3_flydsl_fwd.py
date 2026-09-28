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


def _snr_db(ref, x):
    """Signal-to-noise ratio in dB, as Primus-Turbo's kernel tests measure it."""
    ref, x = ref.double(), x.double()
    return (10 * torch.log10(ref.norm() ** 2 / ((ref - x).norm() ** 2 + 1e-12))).item()


def _reference(q, k, v, table, dtype):
    from primus.backends.megatron.core.transformer.minimax_m3.eager import (
        build_block_keep,
        eager_block_sparse_attention,
    )

    S, Hq, Hkv = q.shape[0], q.shape[2], k.shape[2]
    qb, kb, vb = (t.to(dtype).permute(1, 2, 0, 3) for t in (q, k, v))
    keep = build_block_keep(table.long(), S, BLOCK)
    out, dense = eager_block_sparse_attention(qb, kb, vb, keep, D**-0.5)
    lse = torch.logsumexp(dense.masked_fill(~keep.repeat_interleave(Hq // Hkv, dim=1), float("-inf")), dim=-1)
    return out.float(), lse


def _check(o, lse, q, k, v, table):
    """Global SNR against fp32, as Turbo gates bf16 attention; plus the worst element
    against eager bf16 -- one wrong row barely moves a global SNR, and a wrong
    diagonal or tail block is exactly the local bug this kernel could have."""
    out32, lse32 = _reference(q, k, v, table, torch.float32)
    out16, _ = _reference(q, k, v, table, torch.bfloat16)
    o = o.permute(1, 2, 0, 3).float()
    assert _snr_db(out32, o) >= 40.0, f"O SNR {_snr_db(out32, o):.1f} dB"
    # One bf16 ulp at |o| < 4: both sides round P and O to bf16.
    torch.testing.assert_close(o, out16, rtol=0, atol=1.6e-2)
    torch.testing.assert_close(lse.permute(1, 2, 0), lse32, rtol=0, atol=1e-4)


@pytest.mark.parametrize(
    "S, B, Hkv, topk",
    [
        (300, 1, 1, 16),  # partial last block, fewer blocks than topk (-1 slots)
        (1000, 1, 4, 4),  # topk below the visible block count
        (1000, 2, 2, 16),  # batch > 1
        (2148, 1, 4, 16),
        (4096, 2, 4, 8),
        (1001, 1, 2, 16),  # odd S: one token per work-group
        (1002, 2, 2, 16),  # S % 4 == 2: two tokens per work-group
    ],
)
def test_matches_eager(S, B, Hkv, topk):
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_token_fwd import (
        msa_token_fwd,
    )

    q, k, v, table = _inputs(S, B, Hkv, topk)
    o, lse = msa_token_fwd(q, k, v, table)
    _check(o, lse, q, k, v, table)


def _slot_mass_reference(q, k, v, table):
    """fp32 share of each head's attention that each table slot's block received."""
    from primus.backends.megatron.core.transformer.minimax_m3.eager import (
        build_block_keep,
        eager_block_sparse_attention,
    )

    S, B, Hkv = q.shape[0], q.shape[1], k.shape[2]
    keep = build_block_keep(table.long(), S, BLOCK)
    _, _, probs = eager_block_sparse_attention(
        *(t.float().permute(1, 2, 0, 3) for t in (q, k, v)), keep, D**-0.5, return_probs=True
    )
    n_blocks = -(-S // BLOCK)
    per_block = (
        torch.nn.functional.pad(probs, (0, n_blocks * BLOCK - S))
        .view(B, 16 * Hkv, S, n_blocks, BLOCK)
        .sum(-1)
    )
    slots = table.long().repeat_interleave(16, dim=1)
    return per_block.gather(-1, slots.clamp_min(0)).masked_fill(slots < 0, 0.0), slots


@pytest.mark.parametrize(
    "S, B, Hkv, topk",
    [
        (300, 1, 1, 16),
        (1000, 2, 2, 4),
        (1002, 1, 2, 6),  # two tokens per work-group, topk not a multiple of 4
    ],
)
def test_slot_lse_is_the_attention_each_slot_got(S, B, Hkv, topk):
    """exp(slot_lse - lse) is the sparse indexer loss's target, per head."""
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_token_fwd import (
        msa_token_fwd,
    )

    q, k, v, table = _inputs(S, B, Hkv, topk, seed=2)
    _, lse, slot_lse = msa_token_fwd(q, k, v, table, return_slot_lse=True)
    mass = torch.exp(slot_lse - lse.unsqueeze(-1)).permute(1, 2, 0, 3)
    ref, slots = _slot_mass_reference(q, k, v, table)

    torch.testing.assert_close(mass, ref, rtol=0, atol=1e-5)
    assert torch.isneginf(slot_lse.permute(1, 2, 0, 3)[slots < 0]).all(), "unvisited slots must be -inf"


@pytest.mark.parametrize("kind, value", [("scale", 16.0), ("shift", 100.0), ("shift", -100.0)])
def test_extreme_scores_stay_exact(kind, value):
    """The online softmax keeps a running maximum, so scores beyond exp's float
    range -- large ones, or rows whose every score is very negative -- neither
    overflow nor underflow."""
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_token_fwd import (
        msa_token_fwd,
    )

    q, k, v, table = _inputs(1000, 1, 2, 16, seed=4)
    if kind == "scale":
        q = (q.float() * value).bfloat16()  # scaled scores reach ~95
    else:
        # every key shares a direction u and every query points along it, so each
        # scaled score sits within a few units of `value`
        u = torch.randn(D, device="cuda", generator=torch.Generator(device="cuda").manual_seed(5))
        k = (u + 0.25 * k.float()).bfloat16()
        q = (value / (u.dot(u) * D**-0.5) * u + 0.25 * q.float()).bfloat16()

    o, lse, slot_lse = msa_token_fwd(q, k, v, table, return_slot_lse=True)
    assert torch.isfinite(o).all() and torch.isfinite(lse).all()
    out32, lse32 = _reference(q, k, v, table, torch.float32)
    o = o.permute(1, 2, 0, 3).float()
    assert _snr_db(out32, o) >= 40.0, f"O SNR {_snr_db(out32, o):.1f} dB"
    torch.testing.assert_close(lse.permute(1, 2, 0), lse32, rtol=0, atol=1e-3)
    mass = torch.exp(slot_lse - lse.unsqueeze(-1)).permute(1, 2, 0, 3)
    torch.testing.assert_close(mass, _slot_mass_reference(q, k, v, table)[0], rtol=0, atol=1e-4)


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
    _check(o, lse, q, k, v, table)


def test_refuses_topk_above_16():
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_token_fwd import (
        msa_token_fwd,
    )

    q, k, v, table = _inputs(4096, 1, 1, 17)
    with pytest.raises(AssertionError, match="at most 16"):
        msa_token_fwd(q, k, v, table)


def test_refuses_buffers_past_32_bit_offsets():
    """A 2 GiB q would wrap the kernels' 32-bit buffer offsets: refuse it up front."""
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_token_fwd import (
        MAX_BUFFER_BYTES,
        msa_token_fwd,
    )

    S, B, Hkv, topk = MAX_BUFFER_BYTES // (16 * 4 * D * 2), 1, 4, 16  # q is exactly 2 GiB
    q = torch.empty(S, B, 16 * Hkv, D, device="cuda", dtype=torch.bfloat16)
    k = torch.empty(S, B, Hkv, D, device="cuda", dtype=torch.bfloat16)
    table = torch.full((B, Hkv, S, topk), -1, device="cuda", dtype=torch.int32)
    with pytest.raises(AssertionError, match="32-bit"):
        msa_token_fwd(q, k, k, table)
