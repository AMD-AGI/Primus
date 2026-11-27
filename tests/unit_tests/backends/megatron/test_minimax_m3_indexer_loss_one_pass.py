###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``msa_indexer_loss_one_pass``: slot mass + sparse indexer KL in one Triton kernel, the backward
in one more. The sums run in another order: compared with the eager reference
(``slot_mass_from_slot_lse`` + ``compute_sparse_indexer_loss``) under explicit tolerances,
repeated-run identity, the shapes that break kernels (S not a multiple of the block, ``-1`` padded
rows, early queries with fewer than topk blocks and rows whose blocks are all forced, hot blocks,
B > 1, rows wider than the folded scatter), the first-call check and its fallback, and the gating
(gfx1250, ``msa_fused_indexer_loss``).

The first group runs on the CPU: it checks the kernels' arithmetic as written in plain PyTorch
(``emulate`` / ``emulate_backward``, the walk-order formulation and the hand-written backward
chain) against the eager loss and autograd. The second group needs a GPU and runs the kernels."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron")
pytest.importorskip("triton")

BLOCK = 128
COEFF = 1.0e-2
# kernel against the eager reference (FP32 row sums, exp and log in another order / library). The
# loss tolerance is the MSA operator check's limit for this path (1.25 x the starting tree's
# relative error, which is 0 for the loss, + 1e-6), not a looser one.
LOSS_RTOL = 1.0e-6
GRAD_RTOL = 5.0e-5

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _mods():
    from primus.backends.megatron.core.transformer.minimax_m3 import (
        indexer_loss,
        indexer_loss_one_pass,
    )

    return indexer_loss, indexer_loss_one_pass


def _inputs(S, B, G, heads, topk, kind, seed, device, int32_table=False):
    """``slot_lse [S, B, heads, topk]``, ``lse [S, B, heads]``, ``block_scores [B, G, S, NB]`` and
    the indexer-style table ``[B, G, S, topk]`` (causal: block <= the token's own block, the own
    block forced to +inf, -1 padded where a token has fewer than topk blocks)."""
    gen = torch.Generator().manual_seed(seed)
    NB = -(-S // BLOCK)
    own = (torch.arange(S) // BLOCK).view(1, 1, S, 1)
    scores = torch.randn(B, G, S, NB, generator=gen)
    if kind == "hot":  # many queries select the same few blocks
        scores[..., : min(NB, 3)] += 6.0
    if kind == "zipf":  # text-like block popularity
        scores += 3.0 / (1.0 + torch.arange(NB, dtype=torch.float32)).view(1, 1, 1, NB)
    if kind == "ties":  # equal maxima among the free blocks: the minimum / amax tie rules
        scores = (scores * 2).round() / 2
    blocks = torch.arange(NB).view(1, 1, 1, NB)
    scores = scores.masked_fill(blocks > own, -torch.inf)
    scores = torch.where(blocks == own, torch.full_like(scores, torch.inf), scores)
    k = min(topk, NB)
    top, idx = scores.topk(k, dim=-1)
    idx = idx.masked_fill(top == -torch.inf, -1)
    if kind == "padded":  # holes anywhere in a row, not only behind the valid entries
        idx = idx.masked_fill(torch.rand(idx.shape, generator=gen) < 0.2, -1)
    if k < topk:
        idx = torch.nn.functional.pad(idx, (0, topk - k), value=-1)
    # the forward's logsumexp per slot: -inf on empty slots; lse = logsumexp over the slots
    slot = torch.randn(S, B, heads, topk, generator=gen) * 2.0
    valid = (idx >= 0).permute(2, 0, 1, 3).repeat_interleave(heads // G, dim=2)
    slot = slot.masked_fill(~valid, -torch.inf)
    lse = torch.logsumexp(slot, dim=-1)
    table = idx.to(torch.int32) if int32_table else idx
    return slot.to(device), lse.to(device), scores.to(device), table.to(device)


def _walked(slot_lse, table, seed):
    """A walk: every row's slots permuted (the same permutation for the heads of a group, as the
    forward kernel does); returns slot_lse in walk order and ``(walk_table, slots)``."""
    S, B, heads, K = slot_lse.shape
    G = table.shape[1]
    gen = torch.Generator().manual_seed(seed)
    slots = torch.rand(B, G, S, K, generator=gen).argsort(dim=-1).to(table.device)
    walk_table = table.gather(-1, slots).to(torch.int32)
    per_head = slots.permute(2, 0, 1, 3).repeat_interleave(heads // G, dim=2)
    return slot_lse.gather(-1, per_head), (walk_table, slots)


def _eager(slot_lse, lse, scores, table, walk=None):
    """Loss and block-score gradient on the replaced path, as ``attention.msa_core`` runs it."""
    L, _ = _mods()
    leaf = scores.detach().clone().requires_grad_(True)
    mass = L.slot_mass_from_slot_lse(slot_lse, lse, table.shape[1])
    if walk is not None:
        mass = torch.empty_like(mass).scatter_(-1, walk[1], mass)
    loss = L.compute_sparse_indexer_loss(mass, leaf, table.long(), COEFF)
    (grad,) = torch.autograd.grad(loss, leaf)
    return loss.detach(), grad


def _unselected(table, n_blocks):
    """``[B, G, S, NB]`` bool: the blocks a row's table does not list."""
    idx = table.long()
    hit = torch.zeros(tuple(idx.shape[:-1]) + (n_blocks + 1,), dtype=torch.bool, device=idx.device)
    hit.scatter_(-1, torch.where(idx < 0, torch.full_like(idx, n_blocks), idx), True)
    return ~hit[..., :n_blocks]


def _rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


CPU_CASES = [
    # S, B, G, heads, topk, kind
    (512, 1, 4, 16, 16, "random"),
    (900, 2, 2, 8, 8, "hot"),  # S not a multiple of the block, B > 1
    (641, 1, 4, 8, 16, "padded"),  # -1 holes, early queries with fewer than topk blocks
    (384, 1, 2, 4, 18, "ties"),  # topk not a power of two, tied maxima
    (1024, 1, 4, 16, 16, "zipf"),
]


@pytest.mark.parametrize("case", CPU_CASES)
@pytest.mark.parametrize("walk", [False, True])
def test_emulated_arithmetic_matches_eager(case, walk):
    """The kernels' formulation (walk order, row sums) is the eager loss up to rounding."""
    _, O = _mods()
    S, B, G, heads, topk, kind = case
    slot, lse, scores, table = _inputs(S, B, G, heads, topk, kind, 0, "cpu")
    w = None
    tbl = table
    if walk:
        slot, w = _walked(slot, table, 1)
        tbl = w[0]
    ref_loss, ref_grad = _eager(slot, lse, scores, table, w)
    leaf = scores.clone().requires_grad_(True)
    loss = O.emulate(slot, lse, leaf, tbl, COEFF)
    (grad,) = torch.autograd.grad(loss, leaf)
    assert torch.isfinite(ref_loss) and float(ref_loss) > 0
    assert abs(float(loss) - float(ref_loss)) <= LOSS_RTOL * abs(float(ref_loss))
    assert _rel(grad, ref_grad) <= GRAD_RTOL


@pytest.mark.parametrize("case", CPU_CASES)
def test_emulated_backward_chain_matches_autograd(case):
    """``_bwd_kernel``'s chain (masked fills, softmax backward, the minimum's tie split, amax's
    count, the scatter) against autograd of the eager loss."""
    _, O = _mods()
    S, B, G, heads, topk, kind = case
    slot, lse, scores, table = _inputs(S, B, G, heads, topk, kind, 2, "cpu")
    _, ref_grad = _eager(slot, lse, scores, table)
    R = B * G * S
    g_row = torch.full((R,), COEFF / R)  # d(mean * coeff) / d klrow
    grad = O.emulate_backward(slot, lse, scores, table, g_row)
    assert _rel(grad, ref_grad) <= GRAD_RTOL
    # exactly zero on the blocks a row did not select (elsewhere an exact FP32 cancellation of
    # the eager chain need not be one here)
    assert bool((grad[_unselected(table, scores.shape[-1])] == 0).all())


def test_rows_per_program_rule():
    _, O = _mods()
    assert O._fwd_rows(16, 16) == 4  # 16 heads per group, topk 16: the training shape
    assert O._fwd_rows(1, 8) == O._FWD_MAX_ROWS
    assert O._fwd_rows(64, 32) == 1
    for nrepp in (1, 2, 4, 8, 16, 32, 64):
        for kp in (1, 2, 4, 8, 16, 32):
            br = O._fwd_rows(nrepp, kp)
            assert br >= 1 and (br == 1 or br * nrepp * kp <= O._FWD_ELEMS)


def test_not_eligible_off_gpu_or_with_other_dtypes():
    _, O = _mods()
    slot, lse, scores, table = _inputs(256, 1, 2, 4, 8, "random", 0, "cpu")
    assert not O.eligible(slot, lse, scores, table)  # CPU tensors: the caller keeps its path
    assert O.one_pass_indexer_loss(slot, lse, scores, table, None, COEFF, None, BLOCK) is None


@pytest.mark.parametrize("walk", [False, True])
def test_probe_is_a_function_of_the_shape(walk):
    """The check's inputs: the same on every call (every rank, a resumed run), and shaped like the
    forward's: -inf exactly on the dead slots of the order the kernel reads, lse their logsumexp,
    block scores +inf on the own block and -inf past it."""
    _, O = _mods()
    S, B, G, heads, K = 640, 2, 2, 8, 8
    NB = -(-S // BLOCK)
    a = O._probe(B, G, S, NB, K, heads, BLOCK, torch.int64, walk, "cpu")
    b = O._probe(B, G, S, NB, K, heads, BLOCK, torch.int64, walk, "cpu")
    for x, y in zip(a[:4], b[:4]):
        assert torch.equal(x, y)
    slot, lse, scores, indices, w = a
    assert slot.shape == (S, B, heads, K) and lse.shape == (S, B, heads)
    assert scores.shape == (B, G, S, NB) and indices.shape == (B, G, S, K) and indices.dtype == torch.int64
    assert (w is not None) == walk
    order = indices if w is None else w[0]
    dead = (order < 0).permute(2, 0, 1, 3).repeat_interleave(heads // G, dim=2)
    assert bool(torch.isneginf(slot[dead]).all()) and bool(torch.isfinite(slot[~dead]).all())
    assert torch.equal(lse, torch.logsumexp(slot, dim=-1))
    own = (torch.arange(S) // BLOCK).view(S, 1)
    blocks = torch.arange(NB)
    assert bool(torch.isposinf(scores[..., blocks == own]).all())
    assert bool(torch.isneginf(scores[..., blocks > own]).all())
    assert bool(torch.isfinite(scores[..., blocks < own]).all())
    assert bool((indices[..., 0] == torch.arange(S) // BLOCK).all())  # the own block, forced


def test_check_raises_out_of_memory(monkeypatch):
    """Running out of memory in the check stops the run instead of turning the path off."""
    _, O = _mods()
    monkeypatch.setattr(O, "_decided", {})

    def probe():
        raise torch.cuda.OutOfMemoryError("probe")

    with pytest.raises(torch.cuda.OutOfMemoryError):
        O._checked(("oom",), probe, None, COEFF)
    assert O._decided == {}

    def broken():
        raise ValueError("probe")

    assert O._checked(("value",), broken, None, COEFF) is False
    assert O._decided == {("value",): False}


def _fake_attention(gfx1250=True, **config):
    defaults = dict(msa_fused_indexer_loss=True)
    defaults.update(config)
    return SimpleNamespace(config=SimpleNamespace(**defaults), _gfx1250=gfx1250)


def test_on_with_the_fused_loss_on_gfx1250():
    from primus.backends.megatron.core.models.minimax_m3.minimax_m3_transformer_config import (
        MSATransformerConfig,
    )
    from primus.backends.megatron.core.transformer.minimax_m3 import attention

    assert "msa_indexer_loss_one_pass" not in MSATransformerConfig.__dataclass_fields__
    assert "msa_indexer_loss_one_pass" in attention._MSA_GFX1250_PATHS
    A = attention.MinimaxSparseAttention
    assert A._loss_one_pass(_fake_attention()) is True
    # it stands in for the fused loss's Triton path: nothing without that switch
    assert A._loss_one_pass(_fake_attention(msa_fused_indexer_loss=False)) is False
    assert A._loss_one_pass(_fake_attention(gfx1250=False)) is False


# --------------------------------------------------------------------------- GPU: the kernels

GPU_CASES = [
    # S, B, G, heads, topk, kind, int32 table
    (4096, 1, 4, 64, 16, "random", False),  # the training shape (int64 block_indices)
    (4096, 1, 4, 64, 16, "zipf", True),  # ... with the walk's int32 table
    (4100, 1, 4, 64, 16, "random", True),  # S not a multiple of the block or of the row block
    (900, 2, 2, 8, 8, "hot", False),  # B > 1, hot blocks
    (641, 1, 4, 8, 16, "padded", False),  # -1 holes, early queries
    (384, 1, 2, 4, 18, "ties", True),  # topk not a power of two, tied maxima
    (8320, 1, 4, 16, 16, "random", True),  # 65 blocks: the direct (zero-filled) scatter
    (16384, 1, 4, 64, 16, "zipf", True),  # 128 blocks
    (32768, 1, 4, 64, 16, "zipf", True),  # 256 blocks: the long-sequence shape, at the gate's tolerance
]


def _kernel(slot, lse, scores, tbl):
    _, O = _mods()
    leaf = scores.detach().clone().requires_grad_(True)
    loss = O.one_pass_loss(slot, lse, leaf, tbl, COEFF)
    (grad,) = torch.autograd.grad(loss, leaf)
    return loss.detach(), grad


@gpu
@pytest.mark.parametrize("case", GPU_CASES)
@pytest.mark.parametrize("walk", [False, True])
def test_kernels_match_eager_reference(case, walk):
    S, B, G, heads, topk, kind, i32 = case
    slot, lse, scores, table = _inputs(S, B, G, heads, topk, kind, 0, "cuda", i32)
    w, tbl = None, table
    if walk:
        slot, w = _walked(slot, table, 1)
        tbl = w[0]
    ref_loss, ref_grad = _eager(slot, lse, scores, table, w)
    loss, grad = _kernel(slot, lse, scores, tbl)
    assert torch.isfinite(loss) and bool(torch.isfinite(grad).all())
    assert abs(float(loss) - float(ref_loss)) <= LOSS_RTOL * abs(float(ref_loss))
    assert _rel(grad, ref_grad) <= GRAD_RTOL
    assert bool((grad[_unselected(table, scores.shape[-1])] == 0).all())


@gpu
@pytest.mark.parametrize("case", GPU_CASES[:5])
def test_kernels_are_deterministic(case):
    S, B, G, heads, topk, kind, i32 = case
    slot, lse, scores, table = _inputs(S, B, G, heads, topk, kind, 3, "cuda", i32)
    first = _kernel(slot, lse, scores, table)
    for _ in range(4):
        again = _kernel(slot, lse, scores, table)
        assert torch.equal(first[0].view(torch.int32), again[0].view(torch.int32))
        assert torch.equal(first[1].view(torch.int32), again[1].view(torch.int32))


def _reference(seen=None):
    """``attention.msa_core``'s ``exact_loss`` (the eager loss), recording the inputs it is run on."""

    def reference(slot_lse, lse, bs, indices, walk):
        if seen is not None:
            seen.append((slot_lse, indices))
        L, _ = _mods()
        mass = L.slot_mass_from_slot_lse(slot_lse, lse, indices.shape[1])
        if walk is not None:
            mass = torch.empty_like(mass).scatter_(-1, walk[1], mass)
        return L.compute_sparse_indexer_loss(mass, bs, indices.long(), COEFF)

    return reference


@gpu
@pytest.mark.parametrize("walk", [False, True])
def test_check_takes_and_rejects_the_path(monkeypatch, walk):
    _, O = _mods()
    slot, lse, scores, table = _inputs(1024, 1, 4, 16, 16, "random", 4, "cuda")
    w = None
    if walk:
        slot, w = _walked(slot, table, 1)
    seen = []
    reference = _reference(seen)

    monkeypatch.setattr(O, "_decided", {})
    leaf = scores.clone().requires_grad_(True)
    loss = O.one_pass_indexer_loss(slot, lse, leaf, table, w, COEFF, reference, BLOCK)
    assert loss is not None and len(O._decided) == 1 and all(O._decided.values())
    # decided on the probe inputs of the shape, not on the call's
    probe = O._probe(1, 4, 1024, scores.shape[-1], 16, 16, BLOCK, table.dtype, walk, scores.device)
    assert len(seen) == 1 and torch.equal(seen[0][0], probe[0]) and torch.equal(seen[0][1], probe[3])
    ref = reference(slot, lse, scores, table, w)
    assert abs(float(loss) - float(ref)) <= O._LOSS_RTOL * abs(float(ref))
    # other values of the same shape reuse the decision: nothing is run but the path
    seen.clear()
    zeros = torch.zeros_like(scores).requires_grad_(True)
    assert O.one_pass_indexer_loss(slot, lse, zeros, table, w, COEFF, reference, BLOCK) is not None
    assert not seen
    # a path outside the tolerance is rejected once and for all for that shape
    monkeypatch.setattr(O, "_decided", {})
    monkeypatch.setattr(O, "_LOSS_RTOL", -1.0)
    assert O.one_pass_indexer_loss(slot, lse, leaf, table, w, COEFF, reference, BLOCK) is None
    assert list(O._decided.values()) == [False]
    seen.clear()
    assert O.one_pass_indexer_loss(slot, lse, leaf, table, w, COEFF, reference, BLOCK) is None
    assert not seen  # decided: the reference is not run again
    # an error in the path (not a GPU fault or OOM) keeps the replaced path too
    monkeypatch.setattr(O, "_decided", {})
    monkeypatch.setattr(O, "one_pass_loss", lambda *a: (_ for _ in ()).throw(ValueError("boom")))
    assert O.one_pass_indexer_loss(slot, lse, leaf, table, w, COEFF, reference, BLOCK) is None
