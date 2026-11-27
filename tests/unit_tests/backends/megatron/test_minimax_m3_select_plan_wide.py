###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``msa_select_plan_wide`` and ``msa_select_skip_fill``: the fused block selection past 64
blocks, and the indexer's scoring kernel without its fill launches.

Both are bit-identical: every output (block table, boosted scores, winning keys, dead mask, the
BlockPlan) must equal the unfused selection's bit for bit, and so must the gradients that go
back through it. Neither has a switch (gfx1250 takes them); the tests turn them on and off
through ``index_select._wide_enabled`` / ``_skip_fill_enabled``.
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron")
pytest.importorskip("flydsl")


def _gcn_arch() -> str:
    if not torch.cuda.is_available():
        return ""
    return torch.cuda.get_device_properties(0).gcnArchName


pytestmark = pytest.mark.skipif(
    "gfx950" not in _gcn_arch() and "gfx1250" not in _gcn_arch(),
    reason="the eager BlockPlan builder is a FlyDSL kernel (gfx950, gfx1250)",
)


def _block_scores(B, H, S, block, seed, hot):
    """Causally masked block scores and their argmax key positions (-1 where invisible);
    ``hot``: a few blocks that most queries select."""
    n_blocks = -(-S // block)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    raw = torch.randn(B, H, S, n_blocks, device="cuda", generator=gen)
    if hot:
        raw[..., :: max(1, n_blocks // 3)] += 6.0
    q_block = torch.arange(S, device="cuda") // block
    future = torch.arange(n_blocks, device="cuda").view(1, -1) > q_block.view(-1, 1)
    raw = raw.masked_fill(future, float("-inf"))
    keys = torch.randint(0, block, raw.shape, device="cuda", generator=gen)
    keys = (keys + torch.arange(n_blocks, device="cuda") * block).clamp_max(S - 1)
    return raw, keys.masked_fill(future, -1).to(torch.int32), future


def test_wide_supported_bounds():
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl import select_plan

    assert select_plan.wide_supported(1, 4, 16384, 128, 16)
    assert select_plan.wide_supported(1, 4, 32768, 256, 16)
    assert select_plan.wide_supported(4, 4, 32768, 256, 32)  # 4096 rows of blocks
    assert select_plan.wide_supported(1, 4, 65536, 512, 32)
    assert select_plan.wide_supported(1, 4, 131072, 1024, 16)
    assert select_plan.wide_supported(8, 4, 8192, 64, 16)  # MBS 8 at 8k: 2048 rows
    assert not select_plan.wide_supported(5, 4, 32768, 256, 16)  # 5120 rows of blocks
    assert not select_plan.wide_supported(1, 4, 262144, 2048, 16)
    assert not select_plan.wide_supported(1, 4, 4096, 32, 65)
    assert not select_plan.wide_supported(1, 4, 0, 0, 0)


# S not a multiple of the block or of the 128-token chunk, -1 padded rows (every early query),
# hot blocks, B > 1, block counts that are not powers of two, 16k / 32k / 64k-sized tables
_WIDE_CASES = [
    (1, 4, 16384, 128, 16, 1, 0, False),
    (1, 4, 32768, 128, 16, 1, 1, True),
    (1, 4, 8300, 128, 16, 1, 0, False),  # 65 blocks
    (2, 4, 4100, 16, 16, 1, 1, True),  # 257 blocks, 2056 rows
    (1, 4, 8192, 16, 16, 2, 1, False),  # 512 blocks
    (3, 2, 2000, 16, 8, 1, 0, True),  # 125 blocks, B 3
    (1, 4, 2148, 128, 16, 1, 0, False),  # a shape the narrow kernels serve too
    (8, 4, 8192, 128, 32, 1, 1, False),  # 64 blocks, 2048 rows, topk 32
]


# bounded: the plan's chunk rule (msa_bounded_plan_chunks), which the wide plan takes from plan_chunks
@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("B, H, S, block, topk, local_blocks, init_blocks, hot", _WIDE_CASES)
def test_select_plan_wide_matches_eager(B, H, S, block, topk, local_blocks, init_blocks, hot, bounded):
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl import select_plan
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_token_bwd import (
        build_block_plan,
        plan_chunk_size,
    )
    from primus.backends.megatron.core.transformer.minimax_m3.indexer import (
        _forced_blocks,
    )

    raw, argmax, _ = _block_scores(B, H, S, block, seed=S + B, hot=hot)
    n_blocks = raw.shape[-1]
    topk = min(topk, n_blocks)
    assert select_plan.wide_supported(B, H, S, n_blocks, topk)
    forced = _forced_blocks(S, n_blocks, block, local_blocks, init_blocks, raw.device)

    boosted = select_plan.boost_forced(raw, forced)
    ref_boosted = raw.scatter(-1, forced.view(1, 1, S, -1).expand(B, H, -1, -1), float("inf"))
    assert torch.equal(boosted, ref_boosted)

    topk_scores, topk_idx = boosted.topk(topk, dim=-1)
    outs = [
        select_plan.select_post_and_plan(
            topk_scores, topk_idx, argmax, boosted, block, True, bounded, wide=True
        )
        for _ in range(2)
    ]
    bi, keys, keys_c, dead, table, plan = outs[0]

    ref_bi = topk_idx.masked_fill(topk_scores == float("-inf"), -1)
    slots = ref_bi.clamp_min(0)
    ref_keys = argmax.gather(-1, slots)
    ref_dead = (ref_bi < 0) | torch.isposinf(boosted.gather(-1, slots)) | (ref_keys < 0)
    assert (ref_bi < 0).any() and ref_dead.any(), "the inputs must exercise padding and forced slots"
    assert torch.equal(bi, ref_bi)
    assert torch.equal(keys, ref_keys)
    assert torch.equal(keys_c, ref_keys.clamp_min(0))
    assert torch.equal(dead, ref_dead)
    assert torch.equal(table, ref_bi.to(torch.int32))

    ref = build_block_plan(ref_bi, block, bounded)
    assert plan.n_blocks == ref.n_blocks and plan.n_chunks == ref.n_chunks
    capacity = B * H * S * topk
    if bounded and plan_chunk_size(capacity, bounded=True) != plan_chunk_size(capacity):
        # the two chunk rules differ at this capacity: the plan must be the bounded one
        assert ref.n_chunks != build_block_plan(ref_bi, block).n_chunks
    assert torch.equal(plan.ptr, ref.ptr)
    live = int(ref.ptr[-1])
    assert live > 0
    assert torch.equal(plan.entries[:live], ref.entries[:live])
    assert torch.equal(plan.chunk_ptr, ref.chunk_ptr)
    live_chunks = int(ref.chunk_ptr[-1])
    assert torch.equal(plan.chunks[:live_chunks], ref.chunks[:live_chunks])
    assert torch.equal(plan.chunks[live_chunks:, 1], plan.chunks[live_chunks:, 2]), "spare chunks are empty"
    if hot:
        per_row = ref.ptr[1:] - ref.ptr[:-1]
        assert int(per_row.max()) > 4 * int(per_row.float().mean()), "the inputs must have hot blocks"

    # determinism: a second run gives the same plan
    plan2 = outs[1][5]
    assert torch.equal(plan2.ptr, plan.ptr) and torch.equal(plan2.entries[:live], plan.entries[:live])
    assert torch.equal(plan2.chunks[:live_chunks], plan.chunks[:live_chunks])


@pytest.mark.parametrize(
    "B, H, S, block, local_blocks, init_blocks",
    [(1, 4, 4096, 128, 1, 0), (2, 4, 1000, 128, 1, 1), (1, 4, 8300, 128, 1, 0), (2, 2, 4100, 16, 2, 1)],
)
def test_boost_causal_ignores_the_unwritten_blocks(B, H, S, block, local_blocks, init_blocks):
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl import select_plan
    from primus.backends.megatron.core.transformer.minimax_m3.indexer import (
        _forced_blocks,
    )

    raw, _, future = _block_scores(B, H, S, block, seed=3, hot=False)
    n_blocks = raw.shape[-1]
    forced = _forced_blocks(S, n_blocks, block, local_blocks, init_blocks, raw.device)
    ref = raw.scatter(-1, forced.view(1, 1, S, -1).expand(B, H, -1, -1), float("inf"))
    for poison in (float("nan"), float("inf"), 0.0):
        dirty = raw.masked_fill(future, poison)
        assert torch.equal(select_plan.boost_forced(dirty, forced, causal_block=block), ref)
    assert torch.equal(select_plan.boost_forced(raw, forced, causal_block=0), ref)


def _indexer(topk, block, local_blocks, init_blocks):
    """A MinimaxM3Indexer with the attributes its selection reads, without the projections."""
    from primus.backends.megatron.core.transformer.minimax_m3.indexer import (
        MinimaxM3Indexer,
    )

    ix = MinimaxM3Indexer.__new__(MinimaxM3Indexer)
    torch.nn.Module.__init__(ix)
    ix.config = SimpleNamespace(msa_fused_select_plan=True)
    ix.block_size, ix.topk_blocks = block, topk
    ix.local_blocks, ix.init_blocks = local_blocks, init_blocks
    return ix


def _paths(monkeypatch, wide, skip_fill):
    """Take (or not) the wide plan and the unfilled scoring tables; the latter is a gfx1250 path."""
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl import index_select

    monkeypatch.setattr(index_select, "_wide_enabled", lambda: wide)
    monkeypatch.setattr(index_select, "_skip_fill_enabled", lambda: skip_fill and "gfx1250" in _gcn_arch())


def _qk(S, B, H, seed, hot):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(S, B, H, 128, device="cuda", generator=gen)
    k = torch.randn(S, B, 1, 128, device="cuda", generator=gen)
    if hot:
        # a direction the queries share with the keys of a few positions: hot blocks
        u = torch.randn(128, device="cuda", generator=gen)
        u = u / u.norm()
        q = q + 30.0 * u
        k[:: max(1, S // 5)] += 4.0 * u
    return q.bfloat16(), k.bfloat16()


def _step(ix, q, k):
    """Selection, then the gradients of a fixed function of the selected slots' scores."""
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl.index_select import (
        fused_select_blocks,
    )

    q = q.detach().clone().requires_grad_(True)
    k = k.detach().clone().requires_grad_(True)
    idx, scores, plan = fused_select_blocks(q, k, ix.select_from_block_scores, ix.block_size)
    picked = scores.gather(-1, idx.clamp_min(0))
    w = torch.linspace(-1.0, 1.0, idx.shape[-1], device=q.device)
    loss = (torch.where(torch.isfinite(picked) & (idx >= 0), picked, torch.zeros_like(picked)) * w).sum()
    loss.backward()
    live, live_chunks = int(plan.ptr[-1]), int(plan.chunk_ptr[-1])
    return [
        idx,
        scores,
        plan.ptr,
        plan.entries[:live],
        plan.chunk_ptr,
        plan.chunks[:live_chunks],
        q.grad,
        k.grad,
    ]


# S, B, H, topk, local, init, hot: 16k, 32k, 8k + 1 block with S not a multiple of 64, and shapes
# the narrow plan serves (where only msa_select_skip_fill changes anything)
_E2E_CASES = [
    (16384, 1, 4, 16, 1, 0, False),
    (32768, 1, 4, 16, 1, 0, True),
    (8300, 2, 4, 16, 1, 1, False),
    (4096, 1, 4, 16, 1, 0, True),
    (1000, 2, 4, 4, 1, 1, False),
    (300, 3, 2, 16, 2, 0, False),
]


@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("S, B, H, topk, local_blocks, init_blocks, hot", _E2E_CASES)
def test_fused_select_bitwise_with_wide_plan_and_skip_fill(
    require_bitwise, monkeypatch, S, B, H, topk, local_blocks, init_blocks, hot, bounded
):
    from primus.backends.megatron.core.fusions import bitwise_check
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl import (
        index_select,
        select_plan,
    )

    q, k = _qk(S, B, H, seed=S, hot=hot)
    NB = -(-S // 128)
    wide_shape = not select_plan.supported(B, H, S, NB, min(topk, NB))
    fills = []
    real = index_select.index_block_max

    def spy(*args, fill=True, **kwargs):
        fills.append(fill)
        return real(*args, fill=fill, **kwargs)

    monkeypatch.setattr(index_select, "index_block_max", spy)
    monkeypatch.setattr(bitwise_check, "_results", dict(bitwise_check._results))
    # the same index dK kernel and plan chunking on both sides (bounded: the reference is then the
    # unfused selection with build_block_plan's bounded chunks; that chunking sums in another
    # order, these two paths keep every bit)
    monkeypatch.setattr(index_select, "index_dk_scatter_enabled", lambda: False)
    monkeypatch.setattr(index_select, "bounded_plan_chunks_enabled", lambda: bounded)

    _paths(monkeypatch, wide=False, skip_fill=False)
    ref = _step(_indexer(topk, 128, local_blocks, init_blocks), q, k)
    assert fills and all(f is True for f in fills)

    for wide, skip_fill in ((True, False), (False, True), (True, True)):
        _paths(monkeypatch, wide, skip_fill)
        ix = _indexer(topk, 128, local_blocks, init_blocks)
        first = _step(ix, q, k)  # the call that runs the self-checks
        del fills[:]
        second = _step(ix, q, k)
        key = (S, B, H, NB, 128, min(topk, NB), bounded)
        if wide and wide_shape:
            require_bitwise(
                bitwise_check._results.get(("msa_select_plan_wide", key), False),
                f"msa_select_plan_wide {key}",
            )
        fused = not wide_shape or wide
        if skip_fill and fused and "gfx1250" in _gcn_arch():
            require_bitwise(
                bitwise_check._results.get(("msa_select_skip_fill", key), False),
                f"msa_select_skip_fill {key}",
            )
            assert fills == [False], "the steady call scores once, into unfilled tables"
        else:
            assert fills == [True]
        for got in (first, second):
            assert len(got) == len(ref)
            for a, b in zip(got, ref):
                assert a.dtype == b.dtype and torch.equal(a, b)


def test_skip_fill_needs_forced_blocks(monkeypatch):
    """Without forced blocks there is no boost kernel to write the -inf: the fills stay."""
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl import index_select

    fills = []
    real = index_select.index_block_max

    def spy(*args, fill=True, **kwargs):
        fills.append(fill)
        return real(*args, fill=fill, **kwargs)

    monkeypatch.setattr(index_select, "index_block_max", spy)
    q, k = _qk(1000, 1, 4, seed=1, hot=False)
    _paths(monkeypatch, wide=False, skip_fill=False)
    ref = _step(_indexer(4, 128, 0, 0), q, k)
    del fills[:]
    _paths(monkeypatch, wide=True, skip_fill=True)
    got = _step(_indexer(4, 128, 0, 0), q, k)
    assert fills == [True]
    assert all(torch.equal(a, b) for a, b in zip(got, ref))


def test_wide_plan_chunk_size_is_plan_chunks_rule():
    """The wide plan takes its chunk size from ``plan_chunk_size_of``, the rule of ``plan_chunks``:
    the reference rule, the bounded rule, and the reference rule again where the bounded plan's
    partials would pass 2 GiB. Within ``wide_supported`` (at most 4096 rows) the bounded rule's
    chunk count stays below that limit."""
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl import (
        msa_token_bwd as bwd,
    )
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl import select_plan

    for S in (8192, 16384, 32768, 65536, 131072):
        for B in (1, 2, 4, 16):
            n_blocks, rows, cap = S // 128, B * 4 * (S // 128), B * 4 * S * 16
            assert bwd.plan_chunk_size_of(rows, cap) == bwd.plan_chunk_size(cap)
            size = bwd.plan_chunk_size_of(rows, cap, bounded=True)
            if select_plan.wide_supported(B, 4, S, n_blocks, 16):
                assert size == bwd.plan_chunk_size(cap, bounded=True)
                assert (rows + -(-cap // size)) * bwd.BOUNDED_CHUNK_WS_BYTES < 2**31
    # past the limit (8192 rows): the reference rule, as plan_chunks does
    cap = 8192 * 384 * 4
    assert bwd.plan_chunk_size_of(8192, cap, bounded=True) == bwd.plan_chunk_size(cap)
    assert bwd.plan_chunk_size_of(8191, cap, bounded=True) == bwd.plan_chunk_size(cap, bounded=True)
