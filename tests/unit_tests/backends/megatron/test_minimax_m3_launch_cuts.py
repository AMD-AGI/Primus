###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``msa_fused_indexer_loss_scatter`` (the indexer-loss backward writes the block-score gradient
itself): the same bits as the path it replaces, and the fallbacks."""

import pytest
import torch

pytest.importorskip("megatron")
needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.fixture
def clean(monkeypatch):
    from primus.backends.megatron.core.fusions import bitwise_check

    monkeypatch.setattr(bitwise_check, "_results", {})
    yield bitwise_check


def test_switches_default_to_off():
    from primus.backends.megatron.core.models.minimax_m3.minimax_m3_transformer_config import (
        MSATransformerConfig,
    )

    fields = MSATransformerConfig.__dataclass_fields__
    assert fields["msa_fused_indexer_loss_scatter"].default is False


# ------------------------------------------------------- msa_fused_indexer_loss_scatter


def _loss_inputs(n_blocks, topk, seed, sq=256):
    b, n_index = 1, 2
    gen = torch.Generator(device="cuda").manual_seed(seed)
    block_scores = torch.randn(b, n_index, sq, n_blocks, device="cuda", generator=gen)
    # distinct blocks per row (a top-k selection), -1 padded, with always-visible (+inf) blocks
    order = torch.rand(b, n_index, sq, n_blocks, device="cuda", generator=gen).argsort(dim=-1)
    block_indices = order[..., :topk].contiguous()
    block_indices[..., -3:] = torch.where(
        torch.rand(b, n_index, sq, 3, device="cuda", generator=gen) < 0.3, -1, block_indices[..., -3:]
    )
    block_scores.scatter_(-1, block_indices[..., :1].clamp_min(0), torch.inf)
    block_scores[:, :, :4] = torch.inf  # rows with every selected block forced carry no signal
    block_indices[:, :, 4:6] = -1  # rows without any valid slot
    slot_mass = torch.rand(b, n_index, sq, topk, device="cuda", generator=gen)
    slot_mass[:, :, 8:12, ::2] = 0.0  # exact zeros in the target -> +-0 gradients
    return block_scores, block_indices, slot_mass


def _loss_and_grad(fn, block_scores, block_indices, slot_mass, **kw):
    scores = block_scores.clone().requires_grad_()
    loss = fn(slot_mass, scores, block_indices, 0.01, **kw)
    loss.backward()
    return loss.detach(), scores.grad


@needs_gpu
@pytest.mark.parametrize("n_blocks,topk", [(32, 16), (32, 18), (64, 18), (33, 7), (16, 16)])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_folded_scatter_matches_eager_bitwise(require_bitwise, n_blocks, topk, seed):
    from primus.backends.megatron.core.fusions import bitwise_check
    from primus.backends.megatron.core.transformer.minimax_m3.indexer_loss import (
        compute_sparse_indexer_loss,
    )
    from primus.backends.megatron.core.transformer.minimax_m3.indexer_loss_fused import (
        fused_sparse_indexer_loss,
    )

    inputs = _loss_inputs(n_blocks, topk, seed, sq=257)
    ref = _loss_and_grad(compute_sparse_indexer_loss, *inputs)
    for _ in range(5):
        got = _loss_and_grad(fused_sparse_indexer_loss, *inputs, fold_scatter=True)
        ok = bitwise_check.same_bits(got[0], ref[0]) and bitwise_check.same_bits(got[1], ref[1])
        require_bitwise(ok, f"msa_fused_indexer_loss_scatter {n_blocks} {topk} {seed}")


@needs_gpu
def test_wide_rows_keep_scatter_add(clean, monkeypatch):
    from primus.backends.megatron.core.transformer.minimax_m3 import (
        indexer_loss_fused as mod,
    )

    block_scores, block_indices, slot_mass = _loss_inputs(mod._SCATTER_MAX_BLOCKS + 32, 18, 0)
    seen = []
    real = mod.fused_sparse_indexer_loss
    monkeypatch.setattr(
        mod, "fused_sparse_indexer_loss", lambda *a, fold_scatter=False: seen.append(fold_scatter) or real(*a)
    )
    mod.sparse_indexer_loss(slot_mass, block_scores, block_indices, 0.01, fold_scatter=True)
    assert seen[-1] is False
    assert not any(k[0] == "msa_fused_indexer_loss_scatter" for k in clean._results)


@needs_gpu
def test_scores_with_other_rows_than_the_indices_keep_scatter_add(clean):
    from primus.backends.megatron.core.transformer.minimax_m3 import (
        indexer_loss_fused as mod,
    )

    block_scores, block_indices, slot_mass = _loss_inputs(32, 18, 0)
    assert mod._can_fold(block_scores, block_indices)
    assert not mod._can_fold(block_scores[:, :1], block_indices)
    assert not mod._can_fold(block_scores.view(-1, 32), block_indices)
    assert not mod._can_fold(block_scores.double(), block_indices)
    # the autograd Function decides the same way: asked to fold, it still scatters (scores with
    # more rows than the indices: scatter_add_ leaves the extra rows zero)
    wide = torch.cat((block_scores, block_scores[:, :, :4]), dim=2)
    assert not mod._can_fold(wide, block_indices)
    grads = []
    for fold in (True, False):
        scores = wide.clone().requires_grad_()
        kl = mod._SparseIndexerKL.apply(scores, block_indices, slot_mass, fold)
        assert kl.grad_fn.meta[-1] is False
        kl.sum().backward()
        grads.append(scores.grad)
    assert clean.same_bits(grads[0], grads[1])
    assert not grads[0][:, :, -4:].any()


@needs_gpu
def test_first_call_check_picks_the_path(clean, require_bitwise, monkeypatch):
    from primus.backends.megatron.core.transformer.minimax_m3 import (
        indexer_loss_fused as mod,
    )

    block_scores, block_indices, slot_mass = _loss_inputs(32, 18, 0)
    key = (tuple(slot_mass.shape), tuple(block_scores.shape), tuple(block_indices.shape))
    # switch off: only the fused loss is checked
    mod.sparse_indexer_loss(slot_mass, block_scores, block_indices, 0.01)
    assert list(clean._results) == [("msa_fused_indexer_loss", key)]
    mod.sparse_indexer_loss(slot_mass, block_scores, block_indices, 0.01, fold_scatter=True)
    require_bitwise(clean._results.get(("msa_fused_indexer_loss_scatter", key), False), "folded scatter")

    # a folded kernel that differs: the fused loss with scatter_add_ stays
    clean._results.pop(("msa_fused_indexer_loss_scatter", key))
    real = mod._SparseIndexerKL.backward

    def wrong(ctx, grad_kl):
        out = real(ctx, grad_kl)
        return (out[0] + 1.0,) + tuple(out[1:]) if ctx.meta[-1] else out

    monkeypatch.setattr(mod._SparseIndexerKL, "backward", staticmethod(wrong))
    ref = _loss_and_grad(mod.fused_sparse_indexer_loss, block_scores, block_indices, slot_mass)
    got = _loss_and_grad(mod.sparse_indexer_loss, block_scores, block_indices, slot_mass, fold_scatter=True)
    assert clean._results[("msa_fused_indexer_loss_scatter", key)] is False
    assert clean.same_bits(got[1], ref[1])
