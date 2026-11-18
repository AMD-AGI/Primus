###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""select_plan's selection bookkeeping and BlockPlan against the eager ops they replace.

Everything here is integer bookkeeping or a select of fp32 values, so the fused
outputs must equal the eager ones exactly.
"""

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

BLOCK = 128


def _block_scores(B, H, S, seed):
    """Causally masked block scores and their argmax key positions (-1 where invisible)."""
    n_blocks = -(-S // BLOCK)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    raw = torch.randn(B, H, S, n_blocks, device="cuda", generator=gen)
    q_block = torch.arange(S, device="cuda") // BLOCK
    future = torch.arange(n_blocks, device="cuda").view(1, -1) > q_block.view(-1, 1)
    raw = raw.masked_fill(future, float("-inf"))
    keys = torch.randint(0, BLOCK, raw.shape, device="cuda", generator=gen)
    keys = (keys + torch.arange(n_blocks, device="cuda") * BLOCK).clamp_max(S - 1)
    return raw, keys.masked_fill(future, -1).to(torch.int32)


def test_supported_bounds():
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl import select_plan

    assert select_plan.supported(4, 4, 8192, 64, 32)  # B * H * n_blocks == 1024
    assert not select_plan.supported(5, 4, 8192, 64, 18)  # 1280 rows of blocks
    assert not select_plan.supported(1, 4, 16384, 128, 18)  # more than 64 blocks
    assert not select_plan.supported(1, 4, 4096, 32, 33)
    assert not select_plan.supported(1, 4, 0, 0, 0)


@pytest.mark.parametrize(
    "B, H, S, topk, local_blocks, init_blocks",
    [(1, 4, 4096, 18, 1, 1), (2, 4, 1000, 4, 1, 0), (1, 2, 600, 16, 2, 1), (4, 4, 8192, 32, 1, 1)],
)
def test_select_plan_matches_eager(B, H, S, topk, local_blocks, init_blocks):
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl import select_plan
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_token_bwd import (
        build_block_plan,
    )
    from primus.backends.megatron.core.transformer.minimax_m3.indexer import (
        _forced_blocks,
    )

    raw, argmax = _block_scores(B, H, S, seed=S)
    n_blocks = raw.shape[-1]
    topk = min(topk, n_blocks)
    assert select_plan.supported(B, H, S, n_blocks, topk)
    forced = _forced_blocks(S, n_blocks, BLOCK, local_blocks, init_blocks, raw.device)

    # MinimaxM3Indexer._boost_always_visible
    boosted = select_plan.boost_forced(raw, forced)
    ref_boosted = raw.scatter(-1, forced.view(1, 1, S, -1).expand(B, H, -1, -1), float("inf"))
    assert torch.equal(boosted, ref_boosted)

    topk_scores, topk_idx = boosted.topk(topk, dim=-1)
    bi, keys, keys_c, dead, table, plan = select_plan.select_post_and_plan(
        topk_scores, topk_idx, argmax, boosted, BLOCK, True
    )

    # MinimaxM3Indexer.select_from_block_scores and _FusedSelect's unfused bookkeeping
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

    ref = build_block_plan(ref_bi, BLOCK)
    assert plan.n_blocks == ref.n_blocks and plan.n_chunks == ref.n_chunks
    assert torch.equal(plan.ptr, ref.ptr)
    live = int(ref.ptr[-1])
    assert torch.equal(plan.entries[:live], ref.entries[:live])
    assert torch.equal(plan.chunk_ptr, ref.chunk_ptr)
    live_chunks = int(ref.chunk_ptr[-1])
    assert torch.equal(plan.chunks[:live_chunks], ref.chunks[:live_chunks])
    assert torch.equal(plan.chunks[live_chunks:, 1], plan.chunks[live_chunks:, 2]), "spare chunks are empty"


def test_backward_g_matches_eager():
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl import select_plan

    B, H, S, n_blocks, topk = 1, 4, 1000, 8, 4
    gen = torch.Generator(device="cuda").manual_seed(7)
    grad_scores = torch.randn(B, H, S, n_blocks, device="cuda", generator=gen)
    bi = torch.randint(-1, n_blocks, (B, H, S, topk), device="cuda", generator=gen)
    dead = (bi < 0) | (torch.rand(bi.shape, device="cuda", generator=gen) < 0.1)

    ref = grad_scores.gather(-1, bi.clamp_min(0)).masked_fill_(dead, 0.0).float()
    assert torch.equal(select_plan.backward_g(grad_scores, bi, dead), ref)
