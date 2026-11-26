###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``msa_aligned_block_walk`` (sums in another order): the permuted block table of
``flydsl/block_walk.py`` and the MSA forward / backward that walk it, against the PyTorch
reference of the permutation and an FP32 reference of the attention, with explicit tolerances."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron")
pytest.importorskip("flydsl")
pytest.importorskip("triton")


def _gcn_arch() -> str:
    if not torch.cuda.is_available():
        return ""
    return torch.cuda.get_device_properties(0).gcnArchName


pytestmark = pytest.mark.skipif(
    "gfx1250" not in _gcn_arch(), reason="the aligned walk serves the gfx1250 wave32 MSA kernels"
)

BLOCK, D = 128, 128


def _table(B, Hkv, S, topk, kind, seed=0):
    """Causal, left-packed, -1 padded selections, own block in slot 0. ``random``: independent
    per token; ``local``: neighbours pick nearly the same blocks in different orders; ``hot``: a
    few blocks most tokens select; ``hostile``: -1 holes, rolled rows, a repeated block."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    n_blocks = -(-S // BLOCK)
    qb = torch.arange(S, device="cuda") // BLOCK
    if kind == "random":
        scores = torch.randn(B, Hkv, S, n_blocks, device="cuda", generator=gen)
    else:
        per_block = torch.randn(B, Hkv, n_blocks, n_blocks, device="cuda", generator=gen)
        scores = per_block[:, :, qb] + 0.3 * torch.randn(B, Hkv, S, n_blocks, device="cuda", generator=gen)
        if kind == "hot":
            scores[..., : max(1, n_blocks // 8)] += 6.0
    future = torch.arange(n_blocks, device="cuda")[None, :] > qb[:, None]
    scores = scores.masked_fill(future, float("-inf"))
    scores.scatter_(-1, qb.view(1, 1, S, 1).expand(B, Hkv, S, 1), float("inf"))
    k = min(topk, n_blocks)
    vals, idx = scores.topk(k, dim=-1)
    idx = idx.masked_fill(vals == float("-inf"), -1)
    if k < topk:
        idx = torch.nn.functional.pad(idx, (0, topk - k), value=-1)
    table = idx.to(torch.int32).contiguous()
    if kind == "hostile":
        holes = torch.rand(B, Hkv, S, topk, device="cuda", generator=gen) < 0.3
        holes[..., 0] = False
        mixed = table.masked_fill(holes, -1)
        mixed[:, :, 1::2] = mixed[:, :, 1::2].roll(1, dims=-1)
        mixed[:, :, 2::4] = mixed[:, :, 2::4].roll(3, dims=-1)
        mixed[:, :, 3::8, -1] = mixed[:, :, 3::8, 0]  # one block twice in a row
        # only for rows whose topk slots are all read (the kernels read the first min(visible, topk)
        # slots): earlier rows stay left-packed, so every token keeps a visible key
        full = (torch.arange(S, device="cuda") // BLOCK + 1 >= topk).view(1, 1, S, 1)
        table = torch.where(full, mixed, table).contiguous()
    return table


def _staged_fraction(table, group=4):
    """Share of a group's (token, position) tiles that no earlier token of the group has at the
    same position: what the kernels' shared staging still has to copy."""
    B, H, S, K = table.shape
    g = table.reshape(B * H, S // group, group, K)
    first = torch.ones_like(g, dtype=torch.bool)
    for i in range(1, group):
        for j in range(i):
            first[:, :, i] &= g[:, :, i] != g[:, :, j]
    valid = g >= 0
    return (first & valid).sum().item() / max(1, valid.sum().item())


TABLES = [
    # S, B, Hkv, topk, kind
    (4096, 1, 4, 16, "random"),
    (4096, 1, 4, 16, "local"),
    (4096, 1, 4, 16, "hot"),
    (4096, 1, 4, 16, "hostile"),
    (8192, 1, 2, 16, "local"),
    (1024, 2, 2, 4, "local"),  # B > 1
    (640, 3, 1, 8, "hostile"),
    (384, 1, 2, 16, "local"),  # fewer blocks than topk: -1 padded rows
    (4100, 1, 2, 16, "random"),  # S not a multiple of the block
    (1000, 2, 2, 12, "hot"),  # topk not a power of two, S not a multiple of the block, B > 1
    (2048, 1, 1, 32, "hot"),  # the widest row the kernel takes
]


@pytest.mark.parametrize("S, B, Hkv, topk, kind", TABLES)
def test_kernel_is_the_reference_permutation(S, B, Hkv, topk, kind):
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl import block_walk

    table = _table(B, Hkv, S, topk, kind)
    assert block_walk.eligible(table, BLOCK)
    want_walk, want_slots = block_walk._reference(table)
    for _ in range(5):  # integer kernel: every run must give the same permutation
        walk, slots = block_walk._kernel(table)
        assert walk.dtype == torch.int32 and slots.dtype == torch.int64
        assert torch.equal(walk, want_walk) and torch.equal(slots, want_slots)
    # each row holds exactly the table row's entries; slots maps positions back to slots
    assert torch.equal(table.gather(-1, slots), walk)
    assert torch.equal(slots.sort(-1).values, torch.arange(topk, device="cuda").expand_as(slots))
    # the first token of a group keeps its row
    assert torch.equal(walk[:, :, 0::4], table[:, :, 0::4])
    if kind != "hostile":
        # valid entries stay in front of the -1 padding (the kernels read the first slots only)
        valid = walk >= 0
        assert not (valid[..., 1:] & ~valid[..., :-1]).any()
    # never more tiles to stage than in score order, and far fewer when neighbours agree
    assert _staged_fraction(walk) <= _staged_fraction(table) + 1e-9
    if kind == "local" and topk >= 16 and S >= 4096:
        # neighbours with the same blocks in different score orders: most repeats become shared
        assert _staged_fraction(walk) < 0.7 * _staged_fraction(table)


def test_ineligible_shapes_keep_the_table():
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl import block_walk

    table = _table(1, 2, 512, 8, "local")
    assert block_walk.aligned_block_walk(table[:, :, :510].contiguous(), BLOCK) is None  # S % 4
    assert block_walk.aligned_block_walk(table.to(torch.int64), BLOCK) is None  # not the int32 table
    assert block_walk.aligned_block_walk(table.cpu(), BLOCK) is None
    assert block_walk.aligned_block_walk(table, 126) is None  # block size the workgroup does not divide
    assert block_walk.aligned_block_walk(table[..., :1].contiguous(), BLOCK) is None  # nothing to permute


def test_dispatcher_checks_once_per_shape_and_falls_back(monkeypatch):
    from primus.backends.megatron.core.fusions import bitwise_check
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl import block_walk

    saved = dict(bitwise_check._results)
    bitwise_check._results.clear()
    try:
        table = _table(1, 2, 512, 8, "local")
        want = block_walk._reference(table)
        for _ in range(2):
            got = block_walk.aligned_block_walk(table, BLOCK)
            assert got is not None and torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])
        assert [name for name, _ in bitwise_check._results] == ["msa_aligned_block_walk"]
        assert all(bitwise_check._results.values())
        # a kernel that does not return the reference's permutation is not used
        bitwise_check._results.clear()
        monkeypatch.setattr(block_walk, "_is_permutation", lambda *a: False)
        assert block_walk.aligned_block_walk(table, BLOCK) is None
        assert list(bitwise_check._results.values()) == [False]
    finally:
        bitwise_check._results.clear()
        bitwise_check._results.update(saved)


def _fp32_reference(q, k, v, dout, table):
    """Dense FP32 attention over the selected blocks: o, the per-slot attention mass (group mean),
    dq, dk, dv. ``q`` ``[S, B, Hq, D]``, ``table`` ``[B, Hkv, S, topk]``."""
    S, B, Hq, _ = q.shape
    Hkv = k.shape[2]
    R = Hq // Hkv
    n_blocks = -(-S // BLOCK)
    pos = torch.arange(S, device="cuda")
    keep = torch.zeros(B, Hkv, S, n_blocks + 1, dtype=torch.bool, device="cuda")
    keep.scatter_(-1, torch.where(table < 0, n_blocks, table).long(), True)
    keep = keep[..., :n_blocks][..., pos // BLOCK] & (pos.view(1, 1, 1, S) <= pos.view(1, 1, S, 1))
    qf = q.float().requires_grad_(True)
    kf = k.float().requires_grad_(True)
    vf = v.float().requires_grad_(True)
    qh = qf.permute(1, 2, 0, 3).reshape(B, Hkv, R, S, D)
    kh = kf.permute(1, 2, 0, 3)
    vh = vf.permute(1, 2, 0, 3)
    logits = torch.einsum("bgrsd,bgkd->bgrsk", qh, kh) * D**-0.5
    logits = logits.masked_fill(~keep.unsqueeze(2), float("-inf"))
    p = torch.softmax(logits, dim=-1)
    o = torch.einsum("bgrsk,bgkd->bgrsd", p, vh).reshape(B, Hq, S, D).permute(2, 0, 1, 3)
    (o * dout.float()).sum().backward()
    # mass of slot s of token t: its block's keys
    pb = torch.nn.functional.pad(p.detach(), (0, n_blocks * BLOCK - S)).view(B, Hkv, R, S, n_blocks, BLOCK)
    pb = pb.sum(-1).mean(2)  # [B, Hkv, S, n_blocks]
    mass = pb.gather(-1, table.clamp_min(0).long()).masked_fill(table < 0, 0.0)
    return o.detach(), mass, qf.grad, kf.grad, vf.grad


def _run(q, k, v, dout, table, walk):
    """(o, slot mass in the indexer's slot order, dq, dk, dv) through the autograd wrapper."""
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_function import (
        msa_attention,
    )
    from primus.backends.megatron.core.transformer.minimax_m3.indexer_loss import (
        slot_mass_from_slot_lse,
    )

    leaves = [t.detach().clone().requires_grad_(True) for t in (q, k, v)]
    kwargs = {} if walk is None else {"walk_table": walk[0]}
    o, lse, slot_lse = msa_attention(
        *leaves, table, D**-0.5, BLOCK, return_slot_lse=True, shared_tiles=True, **kwargs
    )
    mass = slot_mass_from_slot_lse(slot_lse, lse, table.shape[1])
    if walk is not None:
        mass = torch.empty_like(mass).scatter_(-1, walk[1], mass)
    o.backward(dout)
    return [o.detach(), mass] + [t.grad for t in leaves]


def _rel(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


E2E = [
    (1024, 1, 2, 4, "local"),
    (1024, 2, 1, 8, "hot"),  # B > 1, hot blocks
    (768, 1, 2, 16, "local"),  # early queries only: fewer blocks than topk, -1 padded rows
    (900, 1, 2, 4, "random"),  # S not a multiple of the block
    (1280, 1, 1, 4, "hostile"),
]


@pytest.mark.parametrize("S, B, Hkv, topk, kind", E2E)
def test_attention_with_the_aligned_walk_matches_fp32(S, B, Hkv, topk, kind):
    """Relaxed numerics: every output and gradient within an explicit tolerance of the FP32
    reference and no further from it than 1.25 x the score-order walk (plus 1e-6); two runs give
    the same bits."""
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl import block_walk

    gen = torch.Generator(device="cuda").manual_seed(11)
    Hq = 16 * Hkv
    q = torch.randn(S, B, Hq, D, device="cuda", dtype=torch.bfloat16, generator=gen)
    k = torch.randn(S, B, Hkv, D, device="cuda", dtype=torch.bfloat16, generator=gen)
    v = torch.randn(S, B, Hkv, D, device="cuda", dtype=torch.bfloat16, generator=gen)
    dout = torch.randn(S, B, Hq, D, device="cuda", dtype=torch.bfloat16, generator=gen) * 0.05
    table = _table(B, Hkv, S, topk, kind, seed=5)
    if kind == "hostile":
        # a repeated block would be counted twice by both walks but once by the dense reference
        table[:, :, 3::8, -1] = -1
    walk = block_walk._kernel(table)
    assert not torch.equal(walk[0], table)

    ref = _fp32_reference(q, k, v, dout, table)
    base = _run(q, k, v, dout, table, None)
    got = _run(q, k, v, dout, table, walk)
    again = _run(q, k, v, dout, table, walk)
    names = ["o", "slot mass", "dq", "dk", "dv"]
    tol = {"o": 5e-3, "slot mass": 1e-5, "dq": 6e-3, "dk": 6e-3, "dv": 5e-3}
    for name, r, b, g, a in zip(names, ref, base, got, again):
        assert torch.equal(g, a), f"{name}: two runs differ"
        assert torch.isfinite(g.float()).all()
        err, err_base = _rel(g, r), _rel(b, r)
        assert err <= tol[name], f"{name}: relative error {err:.3e} vs FP32"
        assert err <= 1.25 * err_base + 1e-6, f"{name}: {err:.3e} vs score-order walk {err_base:.3e}"


def _fake_attention(**config):
    defaults = dict(msa_shared_tile_staging=True)
    defaults.update(config)
    return SimpleNamespace(config=SimpleNamespace(**defaults), _gfx1250=True, block_size=BLOCK)


def test_the_shared_staging_decides_the_walk():
    from primus.backends.megatron.core.models.minimax_m3.minimax_m3_transformer_config import (
        MSATransformerConfig,
    )
    from primus.backends.megatron.core.transformer.minimax_m3.attention import (
        MinimaxSparseAttention,
    )

    assert "msa_aligned_block_walk" not in MSATransformerConfig.__dataclass_fields__
    table = _table(1, 2, 512, 8, "local")
    walk = MinimaxSparseAttention._aligned_walk(_fake_attention(), table.to(torch.int64), None)
    assert walk is not None and torch.equal(table.gather(-1, walk[1]), walk[0])
    plan = SimpleNamespace(table=table)
    walk2 = MinimaxSparseAttention._aligned_walk(_fake_attention(), table.to(torch.int64), plan)
    assert torch.equal(walk2[0], walk[0])
    # nothing shares a tile: no reason to permute
    off = _fake_attention(msa_shared_tile_staging=False)
    assert MinimaxSparseAttention._aligned_walk(off, table.to(torch.int64), None) is None
    other_gpu = _fake_attention()
    other_gpu._gfx1250 = False
    assert MinimaxSparseAttention._aligned_walk(other_gpu, table.to(torch.int64), None) is None
