###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``msa_shared_rope_tables`` (RoPE tables kept per ``freqs`` tensor) and
``msa_fused_indexer_loss_scatter`` (the indexer-loss backward writes the block-score gradient
itself): the same bits as the paths they replace, and the fallbacks."""

import pytest
import torch

pytest.importorskip("megatron")
needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.fixture
def clean(monkeypatch):
    from primus.backends.megatron.core.fusions import bitwise_check
    from primus.backends.megatron.core.transformer.minimax_m3.rope import RopeTables

    monkeypatch.setattr(bitwise_check, "_results", {})
    RopeTables._shared.clear()
    yield bitwise_check
    RopeTables._shared.clear()


def _freqs(seq=64, rot=8, seed=0):
    gen = torch.Generator().manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return (torch.rand(seq, 1, 1, rot, generator=gen) * 50.0).to(dev)


# ---------------------------------------------------------------- msa_shared_rope_tables


def test_switches_default_to_off():
    from primus.backends.megatron.core.models.minimax_m3.minimax_m3_transformer_config import (
        MSATransformerConfig,
    )

    fields = MSATransformerConfig.__dataclass_fields__
    assert fields["msa_shared_rope_tables"].default is False
    assert fields["msa_fused_indexer_loss_scatter"].default is False


def test_unshared_tables_are_recomputed_per_object(clean):
    from primus.backends.megatron.core.transformer.minimax_m3.rope import RopeTables

    freqs = _freqs()
    a, b = RopeTables(freqs), RopeTables(freqs)
    assert a.cos_sin(torch.bfloat16)[0] is not b.cos_sin(torch.bfloat16)[0]
    assert not RopeTables._shared and not clean._results


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_shared_tables_are_reused_and_bitwise(clean, require_bitwise, dtype):
    from primus.backends.megatron.core.transformer.minimax_m3.rope import RopeTables

    freqs = _freqs()
    want = RopeTables(freqs).cos_sin(dtype)
    first = RopeTables(freqs, shared=True)
    cos0, sin0 = first.cos_sin(dtype)
    for _ in range(3):
        again = RopeTables(freqs, shared=True)
        cos1, sin1 = again.cos_sin(dtype)
        require_bitwise(
            any(k[0] == "msa_shared_rope_tables" and v for k, v in clean._results.items()),
            "msa_shared_rope_tables",
        )
        assert cos1 is cos0 and sin1 is sin0
    assert clean.same_bits(cos0, want[0]) and clean.same_bits(sin0, want[1])
    x = torch.randn(64, 2, 3, 12, device=freqs.device).to(dtype)
    assert clean.same_bits(again.apply(x), RopeTables(freqs).apply(x))
    assert len(clean._results) == 1, "one check per freqs shape and dtype set"


def test_shared_tables_miss_on_another_or_changed_freqs(clean):
    from primus.backends.megatron.core.transformer.minimax_m3.rope import RopeTables

    freqs = _freqs()
    cos0 = RopeTables(freqs, shared=True).cos_sin(torch.float32)[0]
    other = freqs.clone()  # equal values, another tensor: its own tables
    assert RopeTables(other, shared=True).cos_sin(torch.float32)[0] is not cos0
    freqs.mul_(2.0)  # written in place: the kept tables are stale
    fresh = RopeTables(freqs, shared=True).cos_sin(torch.float32)
    assert fresh[0] is not cos0
    assert clean.same_bits(fresh[0], torch.cos(freqs)) and clean.same_bits(fresh[1], torch.sin(freqs))
    # the entry was replaced by the new version's tables
    assert RopeTables(freqs, shared=True).cos_sin(torch.float32)[0] is fresh[0]
    grad_freqs = _freqs().requires_grad_()
    RopeTables(grad_freqs, shared=True).cos_sin(torch.float32)
    assert id(grad_freqs) not in RopeTables._shared


def test_shared_tables_stay_out_of_a_cuda_graph_capture(clean, monkeypatch):
    from primus.backends.megatron.core.transformer.minimax_m3.rope import RopeTables

    freqs = _freqs()
    if not freqs.is_cuda:
        pytest.skip("needs a GPU")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    RopeTables(freqs, shared=True).cos_sin(torch.float32)
    assert not RopeTables._shared


def test_shared_tables_of_a_freed_freqs_are_not_reused(clean):
    from primus.backends.megatron.core.transformer.minimax_m3.rope import RopeTables

    freqs = _freqs()
    RopeTables(freqs, shared=True).cos_sin(torch.float32)
    key = id(freqs)
    del freqs
    assert RopeTables._shared[key][0]() is None
    for seed in range(1, 12):  # more tensors than the dict keeps
        f = _freqs(seed=seed)
        t = RopeTables(f, shared=True).cos_sin(torch.float32)
        assert clean.same_bits(t[0], torch.cos(f))
    assert len(RopeTables._shared) <= RopeTables._SHARED_MAX


def test_shared_tables_keep_recomputing_when_the_check_fails(clean, monkeypatch):
    from primus.backends.megatron.core.transformer.minimax_m3.rope import RopeTables

    freqs = _freqs()
    RopeTables(freqs, shared=True).cos_sin(torch.float32)
    monkeypatch.setattr(clean, "same_bits", lambda a, b: False)
    second = RopeTables(freqs, shared=True)
    monkeypatch.undo()
    cos0 = RopeTables._shared[id(freqs)][2][torch.float32][0]
    assert second.cos_sin(torch.float32)[0] is not cos0
    assert torch.equal(second.cos_sin(torch.float32)[0], cos0)


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
