###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The fused flydsl indexer selection against the PyTorch path it replaces.

The PyTorch path forms the full [b, n_index, S, S] fp32 scores, masks them and
max-pools each block; the fused path never forms them. Both must select the
same blocks, report the same block scores, and send the sparse indexer loss's
gradient back to index q and k identically.
"""

import pytest
import torch

pytest.importorskip("megatron")
pytest.importorskip("flydsl")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName,
    reason="the flydsl MSA kernels target gfx950",
)

BLOCK = 128


class _Selector:
    """The attributes MinimaxM3Indexer's selection reads, without building the module."""

    def __init__(self, topk, local_blocks=1, init_blocks=0):
        from primus.backends.megatron.core.transformer.minimax_m3.indexer import (
            MinimaxM3Indexer,
        )

        self.block_size = BLOCK
        self.topk_blocks = topk
        self.local_blocks = local_blocks
        self.init_blocks = init_blocks
        self._impl = MinimaxM3Indexer

    def _boost_always_visible(self, *args):
        return self._impl._boost_always_visible(self, *args)

    def select_from_block_scores(self, *args):
        return self._impl.select_from_block_scores(self, *args)

    def select_blocks(self, scores):
        return self._impl.select_blocks(self, scores)


def _inputs(S, B, H, D, seed=0, shared=0.0):
    """Random index q and k. ``shared`` > 0 adds a direction common to both --
    like an attention sink -- so a few keys win most blocks for most queries."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    if not shared:
        q = torch.randn(S, B, H, D, device="cuda", dtype=torch.bfloat16, generator=gen)
        k = torch.randn(S, B, 1, D, device="cuda", dtype=torch.bfloat16, generator=gen)
        return q, k
    q = torch.randn(S, B, H, D, device="cuda", generator=gen)
    k = torch.randn(S, B, 1, D, device="cuda", generator=gen)
    u = torch.randn(D, device="cuda", generator=gen)
    u = u / u.norm()
    q = q + shared * D**0.5 * u
    k = k + shared * (k @ u).abs().unsqueeze(-1) * u
    return q.bfloat16(), k.bfloat16()


def _reference_select(selector, q, k):
    scores = q.permute(1, 2, 0, 3).float() @ k.permute(1, 2, 0, 3).float().transpose(-1, -2)
    return selector.select_blocks(scores)


def _snr_db(ref, x):
    ref, x = ref.double(), x.double()
    return (10 * torch.log10(ref.norm() ** 2 / ((ref - x).norm() ** 2 + 1e-12))).item()


@pytest.mark.parametrize("S, B, H, D", [(300, 1, 4, 128), (1000, 2, 4, 128), (2148, 1, 2, 64)])
def test_block_max_matches_pytorch(S, B, H, D):
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl.index_block_max import (
        index_block_max,
    )

    q, k = _inputs(S, B, H, D)
    block_max, argmax = index_block_max(q, k, BLOCK)

    scores = q.permute(1, 2, 0, 3).float() @ k.permute(1, 2, 0, 3).float().transpose(-1, -2)
    pos = torch.arange(S, device="cuda")
    scores = scores.masked_fill(pos.view(1, 1, 1, S) > pos.view(1, 1, S, 1), float("-inf"))
    n_blocks = -(-S // BLOCK)
    blocks = torch.nn.functional.pad(scores, (0, n_blocks * BLOCK - S), value=float("-inf"))
    ref_max, ref_arg = blocks.view(B, H, S, n_blocks, BLOCK).max(dim=-1)
    ref_arg = ref_arg + torch.arange(n_blocks, device="cuda") * BLOCK

    visible = torch.isfinite(ref_max)
    assert torch.equal(torch.isneginf(block_max), ~visible)
    torch.testing.assert_close(block_max[visible], ref_max[visible], rtol=0, atol=1e-3)
    assert (argmax[visible] == ref_arg[visible]).float().mean() > 0.999
    assert (argmax[~visible] == -1).all()


@pytest.mark.parametrize("S, B, H, topk", [(300, 1, 4, 16), (1000, 2, 4, 4), (2148, 1, 2, 16)])
def test_fused_selection_matches_the_pytorch_path(S, B, H, topk):
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl.index_select import (
        fused_select_blocks,
    )

    selector = _Selector(topk)
    q, k = _inputs(S, B, H, 128, seed=1)
    with torch.no_grad():
        idx, scores, plan = fused_select_blocks(q, k, selector.select_from_block_scores, BLOCK)
    assert plan is None, "no backward will follow, so no plan"
    ref_idx, ref_scores = _reference_select(selector, q, k)

    assert idx.shape == ref_idx.shape
    same = (idx.sort(dim=-1).values == ref_idx.sort(dim=-1).values).all(dim=-1)
    assert same.float().mean() > 0.999, "selections may differ only on near-ties"
    finite = torch.isfinite(ref_scores)
    assert torch.equal(torch.isfinite(scores), finite)
    torch.testing.assert_close(scores[finite], ref_scores[finite], rtol=0, atol=1e-3)


@pytest.mark.parametrize("shared", [0.0, 4.0])
def test_fused_gradients_match_the_pytorch_path(shared):
    """Through the sparse indexer loss, the one the flydsl backend trains with.

    ``shared`` skews the winning keys so one key collects hundreds of slots,
    spread over several of dK's plan chunks and every index head, which
    exercises the one-hot accumulation and the cross-chunk reduce."""
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl.index_block_max import (
        index_block_max,
    )
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl.index_select import (
        fused_select_blocks,
    )
    from primus.backends.megatron.core.transformer.minimax_m3.indexer_loss import (
        compute_sparse_indexer_loss,
    )

    S, B, H, topk = 1000, 2, 4, 4
    selector = _Selector(topk)
    q, k = _inputs(S, B, H, 128, seed=2, shared=shared)
    if shared:
        idx, _, plan = fused_select_blocks(q, k, selector.select_from_block_scores, BLOCK)
        winners = index_block_max(q, k, BLOCK)[1].gather(-1, idx.clamp_min(0))[idx >= 0]
        assert torch.bincount(winners.long()).max() > 256
        assert (plan.chunk_ptr[1:] - plan.chunk_ptr[:-1]).max() > 1

    def grads(select):
        qs, ks = q.clone().requires_grad_(True), k.clone().requires_grad_(True)
        idx, scores = select(qs, ks)[:2]
        gen = torch.Generator(device="cuda").manual_seed(3)
        target = torch.softmax(torch.randn(idx.shape, device="cuda", generator=gen), dim=-1).masked_fill(
            idx < 0, 0
        )
        compute_sparse_indexer_loss(target, scores, idx, 1.0).backward()
        return qs.grad.float(), ks.grad.float()

    dq_f, dk_f = grads(lambda a, b: fused_select_blocks(a, b, selector.select_from_block_scores, BLOCK))
    dq_r, dk_r = grads(lambda a, b: _reference_select(selector, a, b))

    assert dq_r.abs().sum() > 0 and dk_r.abs().sum() > 0
    assert _snr_db(dq_r, dq_f) >= 40.0, f"dq SNR {_snr_db(dq_r, dq_f):.1f} dB"
    assert _snr_db(dk_r, dk_f) >= 40.0, f"dk SNR {_snr_db(dk_r, dk_f):.1f} dB"


def _fused_step(q, k, selector):
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl.index_select import (
        fused_select_blocks,
    )
    from primus.backends.megatron.core.transformer.minimax_m3.indexer_loss import (
        compute_sparse_indexer_loss,
    )

    qs, ks = q.clone().requires_grad_(True), k.clone().requires_grad_(True)
    idx, scores, _ = fused_select_blocks(qs, ks, selector.select_from_block_scores, BLOCK)
    target = torch.full(idx.shape, 1.0 / idx.shape[-1], device="cuda").masked_fill(idx < 0, 0)
    compute_sparse_indexer_loss(target, scores, idx, 1.0).backward()
    return qs.grad, ks.grad


@pytest.mark.parametrize("shared", [0.0, 4.0])
def test_fused_backward_is_deterministic(shared):
    """dK is summed per key in a fixed order, not with atomics."""
    selector = _Selector(16)
    q, k = _inputs(2048, 1, 4, 128, seed=4, shared=shared)
    first = _fused_step(q, k, selector)
    second = _fused_step(q, k, selector)
    assert all(torch.equal(a, b) for a, b in zip(first, second))


def test_fused_selection_has_no_host_sync():
    selector = _Selector(16)
    q, k = _inputs(1000, 2, 4, 128, seed=5)
    _fused_step(q, k, selector)  # compile outside the check
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        _fused_step(q, k, selector)
    finally:
        torch.cuda.set_sync_debug_mode("default")
