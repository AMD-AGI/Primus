###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The fused sparse indexer loss against ``compute_sparse_indexer_loss``: the same loss
and the same block-score gradient, bit for bit."""

import pytest
import torch

pytest.importorskip("megatron")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _inputs(n_blocks, topk, seed):
    b, n_index, sq = 1, 2, 256
    gen = torch.Generator(device="cuda").manual_seed(seed)
    block_scores = torch.randn(b, n_index, sq, n_blocks, device="cuda", generator=gen)
    # distinct blocks per row, -1 padded, with always-visible (+inf) blocks
    order = torch.rand(b, n_index, sq, n_blocks, device="cuda", generator=gen).argsort(dim=-1)
    block_indices = order[..., :topk].contiguous()
    block_indices[..., -3:] = torch.where(
        torch.rand(b, n_index, sq, 3, device="cuda", generator=gen) < 0.3, -1, block_indices[..., -3:]
    )
    block_scores.scatter_(-1, block_indices[..., :1].clamp_min(0), torch.inf)
    # rows with every selected block forced carry no signal
    block_scores[:, :, :4] = torch.inf
    slot_mass = torch.rand(b, n_index, sq, topk, device="cuda", generator=gen)
    return block_scores, block_indices, slot_mass


@pytest.mark.parametrize("n_blocks", [32, 96])
@pytest.mark.parametrize("seed", [0, 1])
def test_matches_eager_bitwise(n_blocks, seed):
    from primus.backends.megatron.core.transformer.minimax_m3.indexer_loss import (
        compute_sparse_indexer_loss,
    )
    from primus.backends.megatron.core.transformer.minimax_m3.indexer_loss_fused import (
        fused_sparse_indexer_loss,
    )

    block_scores, block_indices, slot_mass = _inputs(n_blocks, 18, seed)

    def loss_and_grad(fn):
        scores = block_scores.clone().requires_grad_()
        loss = fn(slot_mass, scores, block_indices, 0.01)
        loss.backward()
        return loss.detach(), scores.grad

    ref_loss, ref_grad = loss_and_grad(compute_sparse_indexer_loss)
    loss, grad = loss_and_grad(fused_sparse_indexer_loss)
    assert torch.equal(loss.view(torch.int32), ref_loss.view(torch.int32))
    assert torch.equal(grad.view(torch.int32), ref_grad.view(torch.int32))


def test_first_call_check_picks_the_path(monkeypatch):
    from primus.backends.megatron.core.fusions import bitwise_check
    from primus.backends.megatron.core.transformer.minimax_m3 import (
        indexer_loss_fused as mod,
    )

    block_scores, block_indices, slot_mass = _inputs(32, 18, 0)
    key = (tuple(slot_mass.shape), tuple(block_scores.shape), tuple(block_indices.shape))
    monkeypatch.setattr(bitwise_check, "_results", {})
    mod.sparse_indexer_loss(slot_mass, block_scores, block_indices, 0.01)
    assert bitwise_check._results == {("msa_fused_indexer_loss", key): True}

    calls = []
    monkeypatch.setattr(bitwise_check, "_results", {})
    monkeypatch.setattr(mod, "fused_sparse_indexer_loss", lambda *a: calls.append(1) or a[1].sum())
    mod.sparse_indexer_loss(slot_mass, block_scores, block_indices, 0.01)
    assert bitwise_check._results == {("msa_fused_indexer_loss", key): False}
    assert calls == [1], "only the check ran the mismatching path"
