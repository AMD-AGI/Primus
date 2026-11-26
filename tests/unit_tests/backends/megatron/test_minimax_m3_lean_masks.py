###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``msa_lean_masks``: the gfx1250 MSA forward, dQ and dK/dV kernels and the indexer's scoring
kernel with the causal compare skipped on fully visible steps, and the indexer's dK kernel with
its zero operand half read from a pad, against the same kernels without it, bit for bit."""

import pytest
import torch

pytest.importorskip("megatron")
pytest.importorskip("flydsl")


def _gcn_arch() -> str:
    if not torch.cuda.is_available():
        return ""
    return torch.cuda.get_device_properties(0).gcnArchName


pytestmark = pytest.mark.skipif(
    "gfx1250" not in _gcn_arch(), reason="the mask skip is in the gfx1250 wave32 MSA kernels"
)

BLOCK, D = 128, 128


def _table(B, Hkv, S, topk, gen, kind):
    """Causal, left-packed, -1 padded selections with the token's own block in slot 0, as the
    indexer makes them. ``random``: independent per token; ``local``: neighbours pick nearly the
    same blocks; ``hot``: a few blocks that most tokens select."""
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
    return idx.to(torch.int32).contiguous()


def _hostile_table(B, Hkv, S, topk, seed=3):
    """Tables no indexer makes: -1 holes inside a row, the token's own block at any slot or twice,
    and rows rolled so that neighbours disagree. Still causal. The skip must be decided per step
    from the block actually read, not from the slot."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    table = _table(B, Hkv, S, topk, gen, "local")
    holes = torch.rand(B, Hkv, S, topk, device="cuda", generator=gen) < 0.3
    holes[..., 0] = False
    mixed = table.masked_fill(holes, -1)
    mixed[:, :, 1::2] = mixed[:, :, 1::2].roll(1, dims=-1)
    mixed[:, :, 2::4] = mixed[:, :, 2::4].roll(3, dims=-1)
    mixed[:, :, 3::8, -1] = mixed[:, :, 3::8, 0]  # the own (diagonal) block a second time
    full = (torch.arange(S, device="cuda") // BLOCK + 1 >= topk).view(1, 1, S, 1)
    return torch.where(full, mixed, table).contiguous()


def _inputs(S, B, Hkv, topk, kind, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    Hq = 16 * Hkv
    q = torch.randn(S, B, Hq, D, device="cuda", dtype=torch.bfloat16, generator=gen)
    k = torch.randn(S, B, Hkv, D, device="cuda", dtype=torch.bfloat16, generator=gen)
    v = torch.randn(S, B, Hkv, D, device="cuda", dtype=torch.bfloat16, generator=gen)
    dout = torch.randn(S, B, Hq, D, device="cuda", dtype=torch.bfloat16, generator=gen)
    table = _hostile_table(B, Hkv, S, topk) if kind == "hostile" else _table(B, Hkv, S, topk, gen, kind)
    return q, k, v, dout, table


def _bits(t):
    return t.view({2: torch.int16, 4: torch.int32}[t.element_size()])


def _same(got, want):
    return len(got) == len(want) and all(
        a.shape == b.shape and a.dtype == b.dtype and torch.equal(_bits(a), _bits(b))
        for a, b in zip(got, want)
    )


@pytest.fixture
def force_checks(monkeypatch):
    """Run the kernels under test whatever the self-check says: these tests compare the kernels
    themselves; the dispatchers would run the kernels without the skip after a failed check."""
    from primus.backends.megatron.core.fusions import bitwise_check

    monkeypatch.setattr(bitwise_check, "passed", lambda *args, **kwargs: True)


CASES = [
    # S, B, Hkv, topk, table kind
    (4096, 1, 4, 16, "random"),
    (4096, 1, 4, 16, "local"),
    (4096, 1, 4, 16, "hot"),
    (4096, 1, 4, 16, "hostile"),
    (8192, 1, 2, 16, "random"),  # more blocks than topk for most tokens
    (1024, 2, 2, 4, "local"),  # B > 1
    (640, 3, 1, 8, "hostile"),
    (384, 1, 2, 16, "local"),  # fewer blocks than topk: -1 padded rows
    (4100, 1, 2, 16, "random"),  # partial last block (keys past S), shared layout
    (1000, 2, 2, 16, "hot"),  # S not a multiple of the block, B > 1
    (641, 1, 2, 16, "random"),  # odd S: one token per workgroup
    (4098, 1, 2, 16, "hostile"),  # forward one token per workgroup, dQ two
]


@pytest.mark.parametrize("S, B, Hkv, topk, kind", CASES)
@pytest.mark.parametrize("slot_lse", [False, True])
@pytest.mark.parametrize("shared", [False, True])
def test_forward_kernel_bitwise(require_bitwise, force_checks, S, B, Hkv, topk, kind, slot_lse, shared):
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_token_fwd_w32 import (
        msa_token_fwd_wave32,
    )

    q, k, v, _, table = _inputs(S, B, Hkv, topk, kind)
    want = msa_token_fwd_wave32(q, k, v, table, None, BLOCK, slot_lse, shared_tiles=shared)
    ok = True
    for _ in range(5):  # repeats: the result must not depend on scheduling
        got = msa_token_fwd_wave32(
            q, k, v, table, None, BLOCK, slot_lse, shared_tiles=shared, step_mask_skip=True
        )
        ok = ok and _same(got, want)
    require_bitwise(ok, f"msa_lean_masks forward {S, B, Hkv, topk, kind, slot_lse, shared}")


@pytest.mark.parametrize("S, B, Hkv, topk, kind", CASES)
@pytest.mark.parametrize("shared", [False, True])
def test_backward_kernels_bitwise(require_bitwise, force_checks, S, B, Hkv, topk, kind, shared):
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_token_bwd import (
        msa_token_bwd,
    )
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_token_fwd_w32 import (
        msa_token_fwd_wave32,
    )

    q, k, v, dout, table = _inputs(S, B, Hkv, topk, kind, seed=1)
    out, lse = msa_token_fwd_wave32(q, k, v, table, None, BLOCK, False)
    want = msa_token_bwd(dout, q, k, v, out, lse, table, None, BLOCK, shared_tiles=shared)
    ok = True
    for _ in range(5):
        got = msa_token_bwd(
            dout, q, k, v, out, lse, table, None, BLOCK, shared_tiles=shared, step_mask_skip=True
        )
        ok = ok and _same(got, want)  # dq (dQ kernel), dk and dv (dK/dV kernel)
    require_bitwise(ok, f"msa_lean_masks dQ, dK/dV {S, B, Hkv, topk, kind, shared}")


@pytest.mark.parametrize(
    "S, B, H", [(4096, 1, 4), (8192, 1, 4), (4100, 2, 4), (1000, 3, 2), (384, 1, 4), (130, 2, 1), (63, 1, 4)]
)
def test_index_block_max_bitwise(require_bitwise, force_checks, S, B, H):
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl.index_block_max import (
        index_block_max,
    )

    gen = torch.Generator(device="cuda").manual_seed(S)
    q = torch.randn(S, B, H, D, device="cuda", dtype=torch.bfloat16, generator=gen)
    k = torch.randn(S, B, D, device="cuda", dtype=torch.bfloat16, generator=gen)
    # ties and a hot key: equal scores must keep the same (first) argmax
    k[S // 3] = k[S // 3 + 1]
    k[S // 2] = 4 * k[S // 2]
    want = index_block_max(q, k, BLOCK)
    ok = True
    for _ in range(5):
        got = index_block_max(q, k, BLOCK, mask_skip=True)
        ok = ok and _same(got, want)
    require_bitwise(ok, f"msa_lean_masks index_block_max {S, B, H}")


def _index_dk_inputs(S, B, H, topk, kind, seed=0, nonfinite=False):
    """A selection, its plan, winning keys inside the selected blocks and slot gradients (zero on
    dead slots, as the indexer's backward passes them)."""
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_token_bwd import (
        build_block_plan,
    )

    gen = torch.Generator(device="cuda").manual_seed(seed)
    table = _hostile_table(B, H, S, topk) if kind == "hostile" else _table(B, H, S, topk, gen, kind)
    plan = build_block_plan(table, BLOCK)
    tok = torch.arange(S, device="cuda").view(1, 1, S, 1)
    off = torch.randint(0, BLOCK, table.shape, device="cuda", generator=gen)
    if kind == "hot":
        off = off % 3  # a few keys win most slots: long per-key sums
    keys = (table.clamp_min(0) * BLOCK + off).clamp_max(S - 1)
    keys = torch.minimum(keys, tok.expand_as(keys)).to(torch.int32)
    g = torch.randn(table.shape, device="cuda", generator=gen) * 1e-3
    dead = (table < 0) | (torch.rand(table.shape, device="cuda", generator=gen) < 0.2)
    g = g.masked_fill(dead, 0.0).contiguous()
    q = torch.randn(S, B, H, D, device="cuda", dtype=torch.bfloat16, generator=gen)
    if nonfinite:
        # the zero operand half must be exact zeros whatever Q holds
        q[S // 3, :, :, 5] = float("inf")
        q[S // 2, :, :, 7] = float("-inf")
        q[S // 5, :, :, 9] = float("nan")
    return plan, g, keys.contiguous(), q


@pytest.mark.parametrize(
    "S, B, H, topk, kind",
    [
        (4096, 1, 4, 16, "random"),
        (4096, 1, 4, 16, "hot"),
        (4096, 1, 4, 16, "hostile"),
        (8192, 1, 4, 16, "local"),
        (1000, 2, 2, 8, "hot"),  # S not a multiple of the block, B > 1
        (384, 1, 4, 16, "local"),  # fewer blocks than topk: -1 padded rows
        (640, 3, 1, 4, "random"),
    ],
)
@pytest.mark.parametrize("nonfinite", [False, True])
def test_index_dk_bitwise(require_bitwise, force_checks, S, B, H, topk, kind, nonfinite):
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl.index_block_max import (
        index_dk,
    )

    plan, g, keys, q = _index_dk_inputs(S, B, H, topk, kind, seed=S + topk, nonfinite=nonfinite)
    want = index_dk(plan, g, keys, q, BLOCK)
    ok = True
    for _ in range(5):
        got = index_dk(plan, g, keys, q, BLOCK, lean=True)
        ok = ok and bitwise_same(got, want)
    require_bitwise(ok, f"msa_lean_masks index_dk {S, B, H, topk, kind, nonfinite}")


def bitwise_same(a, b):
    """Equal bits, any NaN matching any NaN (the non-finite probes)."""
    from primus.backends.megatron.core.fusions import bitwise_check

    return bitwise_check.same_bits(a, b)


def test_dispatchers_check_once_per_shape_and_fall_back(monkeypatch):
    from primus.backends.megatron.core.fusions import bitwise_check
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl import (
        index_block_max as ibm,
    )
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl import (
        msa_token_fwd_w32,
    )
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_token_bwd import (
        msa_token_bwd,
    )

    saved = dict(bitwise_check._results)
    bitwise_check._results.clear()
    try:
        q, k, v, dout, table = _inputs(512, 1, 2, 4, "local", seed=2)
        qi, ki = q[:, :, :4].contiguous(), k[:, :, 0].contiguous()
        dk_in = _index_dk_inputs(512, 1, 4, 4, "local", seed=2)
        want = msa_token_fwd_w32.msa_token_fwd_wave32(q, k, v, table, None, BLOCK, False)
        want_g = msa_token_bwd(dout, q, k, v, want[0], want[1], table, None, BLOCK)
        want_i = ibm.index_block_max(qi, ki, BLOCK)
        want_dk = ibm.index_dk(*dk_in, BLOCK)
        assert bitwise_check._results == {}  # the switch is off: nothing checked
        for _ in range(2):
            got = msa_token_fwd_w32.msa_token_fwd_wave32(
                q, k, v, table, None, BLOCK, False, step_mask_skip=True
            )
            got_g = msa_token_bwd(dout, q, k, v, want[0], want[1], table, None, BLOCK, step_mask_skip=True)
            got_i = ibm.index_block_max(qi, ki, BLOCK, mask_skip=True)
            got_dk = ibm.index_dk(*dk_in, BLOCK, lean=True)
            assert _same(got, want) and _same(got_g, want_g) and _same(got_i, want_i)
            assert _same([got_dk], [want_dk])
        names = sorted(name for name, _ in bitwise_check._results)
        assert names == [
            "msa_lean_masks dK/dV",
            "msa_lean_masks dQ",
            "msa_lean_masks forward",
            "msa_lean_masks index_block_max",
            "msa_lean_masks index_dk",
        ]
        if "gfx1250" in _gcn_arch():
            assert all(bitwise_check._results.values())

        # a failed check keeps the kernels without the skip
        bitwise_check._results.clear()
        monkeypatch.setattr(bitwise_check, "same_bits", lambda a, b: False)
        got = msa_token_fwd_w32.msa_token_fwd_wave32(q, k, v, table, None, BLOCK, False, step_mask_skip=True)
        got_g = msa_token_bwd(dout, q, k, v, want[0], want[1], table, None, BLOCK, step_mask_skip=True)
        got_i = ibm.index_block_max(qi, ki, BLOCK, mask_skip=True)
        got_dk = ibm.index_dk(*dk_in, BLOCK, lean=True)
        assert _same(got, want) and _same(got_g, want_g) and _same(got_i, want_i)
        assert _same([got_dk], [want_dk])
        assert list(bitwise_check._results.values()) == [False, False, False, False, False]
    finally:
        bitwise_check._results.clear()
        bitwise_check._results.update(saved)


def test_no_switch_and_outside_the_relaxed_group():
    """Bit-identical: no config switch, and msa_exact_numerics leaves it on."""
    from primus.backends.megatron.core.models.minimax_m3.minimax_m3_transformer_config import (
        MSATransformerConfig,
    )
    from primus.backends.megatron.core.transformer.minimax_m3 import attention

    assert "msa_lean_masks" not in MSATransformerConfig.__dataclass_fields__
    assert "msa_lean_masks" not in attention._MSA_RELAXED
