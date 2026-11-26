###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``msa_index_dk_scatter`` (sums in another order): the indexer backward's dK chunk kernel as a
per-entry scatter into the winning key row, against an fp64 dense reference with explicit
tolerances, against the one-hot WMMA kernel it replaces (at most 1.25 x its error + 1e-6), and
for determinism; plus the gating (gfx1250) and the dispatcher's first-call decision.

The tests of the kernel itself need gfx1250 (they skip elsewhere); the gating tests and the
plain-PyTorch emulation of the scatter order run on any machine."""

import os

import pytest
import torch

pytest.importorskip("megatron")
pytest.importorskip("flydsl")

BLOCK, D = 128, 128
# relative error (Frobenius) of the bf16 dk against the fp64 reference: bf16 rounding is ~2e-3
DK_TOL = 6e-3


def _gcn_arch() -> str:
    if not torch.cuda.is_available():
        return ""
    return torch.cuda.get_device_properties(0).gcnArchName


needs_gfx1250 = pytest.mark.skipif(
    "gfx1250" not in _gcn_arch(), reason="msa_index_dk_scatter is a gfx1250 wave32 kernel"
)


def _table(B, H, S, topk, gen, kind, device):
    """Causal, left-packed, -1 padded selections with the token's own block in slot 0, as the
    indexer makes them (early tokens have fewer than topk blocks). ``random``: independent per
    token; ``local``: neighbours pick nearly the same blocks; ``hot``: a few blocks that most
    tokens select; ``zipf``: block popularity ~ 1 / rank (a text-like selection)."""
    n_blocks = -(-S // BLOCK)
    qb = torch.arange(S, device=device) // BLOCK
    if kind == "random":
        scores = torch.randn(B, H, S, n_blocks, device=device, generator=gen)
    elif kind == "zipf":
        pop = torch.log(1.0 / torch.arange(1, n_blocks + 1, device=device).float())
        pop = pop[torch.randperm(n_blocks, device=device, generator=gen)]
        gumbel = -torch.log(-torch.log(torch.rand(B, H, S, n_blocks, device=device, generator=gen)))
        scores = 1.5 * pop.view(1, 1, 1, n_blocks) + gumbel
    else:
        per_block = torch.randn(B, H, n_blocks, n_blocks, device=device, generator=gen)
        scores = per_block[:, :, qb] + 0.3 * torch.randn(B, H, S, n_blocks, device=device, generator=gen)
        if kind == "hot":
            scores[..., : max(1, n_blocks // 8)] += 6.0
    future = torch.arange(n_blocks, device=device)[None, :] > qb[:, None]
    scores = scores.masked_fill(future, float("-inf"))
    scores.scatter_(-1, qb.view(1, 1, S, 1).expand(B, H, S, 1), float("inf"))
    k = min(topk, n_blocks)
    vals, idx = scores.topk(k, dim=-1)
    idx = idx.masked_fill(vals == float("-inf"), -1)
    if k < topk:
        idx = torch.nn.functional.pad(idx, (0, topk - k), value=-1)
    return idx.to(torch.int32).contiguous()


def _inputs(S, B, H, topk, kind, seed=0, device="cuda", dead_keys="minus_one"):
    """A selection, winning keys inside the selected blocks, slot gradients (zero on dead slots)
    and index queries. ``dead_keys``: what the key of a dead slot is -- ``-1`` (what the indexer's
    backward passes for an invalid slot) or a valid key (a forced or masked slot)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    table = _table(B, H, S, topk, gen, kind, device)
    tok = torch.arange(S, device=device).view(1, 1, S, 1)
    off = torch.randint(0, BLOCK, table.shape, device=device, generator=gen)
    if kind == "hot":
        off = off % 3  # a few keys win most slots: long sums into one key row
    keys = (table.clamp_min(0) * BLOCK + off).clamp_max(S - 1)
    keys = torch.minimum(keys, tok.expand_as(keys)).to(torch.int32)
    g = torch.randn(table.shape, device=device, generator=gen) * 1e-3
    dead = (table < 0) | (torch.rand(table.shape, device=device, generator=gen) < 0.2)
    g = g.masked_fill(dead, 0.0).contiguous()
    if dead_keys == "minus_one":
        keys = keys.masked_fill(table < 0, -1)
    q = torch.randn(S, B, H, D, device=device, dtype=torch.bfloat16, generator=gen)
    return table, g, keys.contiguous(), q


def _dense_reference(table, g, keys, q):
    """``dk[key, b] += g * q[t, b, h]`` over every slot that carries a gradient, in fp64, with a
    sorted ``index_put_`` (accumulate): ``[S, B, D]`` float64."""
    S, B, H, Dq = q.shape
    live = (g != 0) & (keys >= 0)
    b, h, t, s = live.nonzero(as_tuple=True)
    key = keys[b, h, t, s].long()
    order = torch.argsort(key * B + b, stable=True)
    b, h, t, s, key = b[order], h[order], t[order], s[order], key[order]
    contrib = g[b, h, t, s].double().unsqueeze(-1) * q[t, b, h].double()
    dk = torch.zeros(S, B, Dq, dtype=torch.float64, device=q.device)
    dk.index_put_((key, b), contrib, accumulate=True)
    return dk


def _rel_err(x, ref):
    return float((x.double() - ref).norm() / ref.norm().clamp_min(1e-300))


def _emulate_scatter(table, g, keys, q, chunk=64):
    """The scatter kernel's arithmetic in plain PyTorch on any device: per (batch, head, block)
    plan row the entries in token order, cut into chunks; inside a chunk every entry adds its
    fp32 ``g * q_row`` into its winning key row, one after the other; the partials are then summed
    head by head and chunk by chunk (the reduce kernel's order) and rounded to bf16."""
    S, B, H, Dq = q.shape
    n_blocks = -(-S // BLOCK)
    out = torch.zeros(n_blocks * BLOCK, B, Dq, dtype=torch.float32)
    qf = q.float().cpu()
    table, g, keys = table.cpu(), g.cpu(), keys.cpu()
    for b in range(B):
        for blk in range(n_blocks):
            acc = torch.zeros(BLOCK, Dq, dtype=torch.float32)
            for h in range(H):
                t_idx, s_idx = (table[b, h] == blk).nonzero(as_tuple=True)  # token order
                for c0 in range(0, max(len(t_idx), 1), chunk):
                    part = torch.zeros(BLOCK, Dq, dtype=torch.float32)
                    for t, s in zip(t_idx[c0 : c0 + chunk].tolist(), s_idx[c0 : c0 + chunk].tolist()):
                        key = int(keys[b, h, t, s])
                        if blk * BLOCK <= key < (blk + 1) * BLOCK:
                            part[key - blk * BLOCK] += g[b, h, t, s] * qf[t, b, h]
                    acc = acc + part
            out[blk * BLOCK : (blk + 1) * BLOCK, b] = acc
    return out[:S].to(torch.bfloat16)


# ---------------------------------------------------------------- any machine (CPU)


def test_on_gfx1250_without_a_switch(monkeypatch):
    from primus.backends.megatron.core.models.minimax_m3.minimax_m3_transformer_config import (
        MSATransformerConfig,
    )
    from primus.backends.megatron.core.transformer.minimax_m3 import attention
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl import index_select

    assert "msa_index_dk_scatter" not in MSATransformerConfig.__dataclass_fields__
    assert "msa_index_dk_scatter" in attention._MSA_GFX1250_PATHS
    monkeypatch.setattr(index_select, "_on_gfx1250", lambda: True)
    assert index_select.index_dk_scatter_enabled() is True
    monkeypatch.setattr(index_select, "_on_gfx1250", lambda: False)
    assert index_select.index_dk_scatter_enabled() is False


def test_the_configs_carry_no_switch():
    import yaml

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    model = yaml.safe_load(open(os.path.join(root, "primus/configs/models/megatron/minimax_m3.yaml")))
    assert "msa_index_dk_scatter" not in model and model["msa_bwd_store_ds"] is True
    text = open(os.path.join(root, "examples/megatron/configs/MI455X/minimax_m3-BF16-pretrain.yaml")).read()
    assert "msa_index_dk_scatter" not in text


def test_supported_shapes():
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl.index_block_max import (
        index_dk_scatter_supported,
    )

    assert index_dk_scatter_supported(128, 128)  # the model's index dim: 4 elements per lane, 64 KB
    assert not index_dk_scatter_supported(64, 128)  # 2 per lane: never run on the device
    assert not index_dk_scatter_supported(256, 128)  # 8 per lane: never run on the device
    assert not index_dk_scatter_supported(48, 128)  # not a multiple of 32
    assert not index_dk_scatter_supported(32, 128)  # one element per lane does not lower
    assert not index_dk_scatter_supported(512, 128)  # 16 per lane; 256 KB
    assert not index_dk_scatter_supported(256, 256)  # accumulator above 160 KB


@pytest.mark.parametrize(
    "S, B, H, topk, kind, dead_keys",
    [
        (384, 1, 2, 4, "random", "minus_one"),
        (300, 2, 2, 8, "hot", "valid"),  # S not a multiple of the block, B > 1, -1 padded rows
        (260, 1, 4, 4, "zipf", "minus_one"),
    ],
)
def test_scatter_order_emulation_matches_dense_reference(S, B, H, topk, kind, dead_keys):
    """The algorithm (not the kernel): entry-order scatter per chunk + ordered reduce + bf16."""
    table, g, keys, q = _inputs(S, B, H, topk, kind, seed=S, device="cpu", dead_keys=dead_keys)
    ref = _dense_reference(table, g, keys, q)
    got = _emulate_scatter(table, g, keys, q)
    assert got.shape == (S, B, D)
    assert _rel_err(got, ref) < DK_TOL
    # rows no entry won stay exactly zero
    assert torch.equal(
        got.float()[ref.abs().sum(-1) == 0], torch.zeros_like(got.float()[ref.abs().sum(-1) == 0])
    )


# ---------------------------------------------------------------- gfx1250 (GPU)


@pytest.fixture
def forced_scatter(monkeypatch):
    """Run the scatter kernel itself, not the dispatcher's decision."""
    import primus.backends.megatron.core.transformer.minimax_m3.flydsl.index_block_max as ibm

    monkeypatch.setattr(ibm, "_dk_scatter_checked", lambda *a, **k: True)
    return ibm


def _plan(table):
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_token_bwd import (
        build_block_plan,
    )

    return build_block_plan(table, BLOCK)


SHAPES = [
    (4096, 1, 4, 16, "random", "minus_one"),
    (4096, 1, 4, 16, "hot", "valid"),  # hot blocks and hot keys: long sums into one key row
    (4096, 1, 4, 16, "zipf", "minus_one"),
    (4096, 2, 4, 16, "local", "minus_one"),  # B > 1
    (4100, 1, 4, 16, "random", "minus_one"),  # S not a multiple of the block
    (900, 2, 2, 8, "hot", "valid"),
    (641, 1, 4, 16, "local", "minus_one"),  # fewer blocks than topk: -1 padded rows
    (384, 3, 1, 4, "random", "valid"),
    (8192, 1, 4, 16, "zipf", "minus_one"),
    (32768, 1, 4, 16, "zipf", "minus_one"),  # 256 blocks: the long-sequence shape (gate's rule below)
]


@needs_gfx1250
@pytest.mark.parametrize("S, B, H, topk, kind, dead_keys", SHAPES)
def test_scatter_against_fp64_reference_and_reference_kernel(forced_scatter, S, B, H, topk, kind, dead_keys):
    ibm = forced_scatter
    table, g, keys, q = _inputs(S, B, H, topk, kind, seed=S + topk, dead_keys=dead_keys)
    plan = _plan(table)
    ref = _dense_reference(table, g, keys, q)
    old = ibm.index_dk(plan, g, keys, q, BLOCK)
    new = ibm.index_dk(plan, g, keys, q, BLOCK, scatter=True)
    assert new.shape == old.shape and new.dtype == torch.bfloat16
    assert bool(torch.isfinite(new.float()).all())
    e_old, e_new = _rel_err(old, ref), _rel_err(new, ref)
    assert e_new < DK_TOL, (e_new, e_old)
    assert e_new <= 1.25 * e_old + 1e-6, (e_new, e_old)
    # the two kernels differ by rounding only: far below one bf16 step of the largest value
    assert float((new.float() - old.float()).abs().max()) <= 2.0**-6 * float(old.float().abs().max())


@needs_gfx1250
@pytest.mark.parametrize("S, B, H, topk, kind, dead_keys", SHAPES[:7])
def test_scatter_is_deterministic(forced_scatter, S, B, H, topk, kind, dead_keys):
    ibm = forced_scatter
    table, g, keys, q = _inputs(S, B, H, topk, kind, seed=3 * S + topk, dead_keys=dead_keys)
    plan = _plan(table)
    first = ibm.index_dk(plan, g, keys, q, BLOCK, scatter=True)
    for _ in range(5):
        again = ibm.index_dk(plan, g, keys, q, BLOCK, scatter=True)
        assert torch.equal(first.view(torch.int16), again.view(torch.int16))


@needs_gfx1250
def test_dead_entries_with_nonfinite_q_do_not_reach_the_result(forced_scatter):
    """Slots whose key is -1 (invalid) are skipped by a select, not added as ``0 * q``."""
    ibm = forced_scatter
    S, B, H, topk = 1024, 1, 4, 16
    table, g, keys, q = _inputs(S, B, H, topk, "local", seed=5)
    # a token that only owns invalid or zero-gradient slots in head 0: make its row non-finite
    t = 3  # early token: slots 1.. are -1 padded
    keys[0, 0, t, :] = -1
    g[0, 0, t, :] = 0.0
    plan = _plan(table)
    want = ibm.index_dk(plan, g, keys, q, BLOCK, scatter=True)
    q2 = q.clone()
    q2[t, 0, 0, :] = float("inf")
    got = ibm.index_dk(plan, g, keys, q2, BLOCK, scatter=True)
    assert torch.equal(want.view(torch.int16), got.view(torch.int16))


@needs_gfx1250
def test_dispatcher_decides_once_and_falls_back():
    import primus.backends.megatron.core.transformer.minimax_m3.flydsl.index_block_max as ibm

    table, g, keys, q = _inputs(1280, 1, 4, 16, "random", seed=11)
    plan = _plan(table)
    old = ibm.index_dk(plan, g, keys, q, BLOCK)
    saved, tol = dict(ibm._DK_SCATTER_OK), ibm._DK_SCATTER_TOL
    try:
        ibm._DK_SCATTER_OK.clear()
        new = ibm.index_dk(plan, g, keys, q, BLOCK, scatter=True)
        assert list(ibm._DK_SCATTER_OK.values()) == [True], "the first-call comparison did not pass"
        assert bool(torch.isfinite(new.float()).all())
        # a failed comparison keeps the reference kernel: the same bits as without the switch
        ibm._DK_SCATTER_OK.clear()
        ibm._DK_SCATTER_TOL = -1.0
        kept = ibm.index_dk(plan, g, keys, q, BLOCK, scatter=True)
        assert list(ibm._DK_SCATTER_OK.values()) == [False]
        assert torch.equal(kept.view(torch.int16), old.view(torch.int16))
        # nothing to compare (all gradients zero): undecided, reference kernel, nothing cached
        ibm._DK_SCATTER_OK.clear()
        ibm._DK_SCATTER_TOL = tol
        zero = ibm.index_dk(plan, torch.zeros_like(g), keys, q, BLOCK, scatter=True)
        assert ibm._DK_SCATTER_OK == {} and not bool(zero.float().abs().any())
    finally:
        ibm._DK_SCATTER_TOL = tol
        ibm._DK_SCATTER_OK.clear()
        ibm._DK_SCATTER_OK.update(saved)


@needs_gfx1250
def test_indexer_backward_uses_it(monkeypatch):
    """Through ``_FusedSelect``: the path reaches ``index_dk``."""
    import primus.backends.megatron.core.transformer.minimax_m3.flydsl.index_select as sel

    seen = []
    real = sel.index_dk

    def spy(*a, **k):
        seen.append(bool(k.get("scatter", False)))
        return real(*a, **k)

    monkeypatch.setattr(sel, "index_dk", spy)
    S, B, H = 1024, 1, 4
    gen = torch.Generator(device="cuda").manual_seed(1)
    q = torch.randn(S, B, H, D, device="cuda", dtype=torch.bfloat16, generator=gen).requires_grad_(True)
    k = torch.randn(S, B, 1, D, device="cuda", dtype=torch.bfloat16, generator=gen).requires_grad_(True)

    def select(raw, b):
        vals, idx = raw.topk(4, dim=-1)
        return idx.masked_fill(vals == float("-inf"), -1), raw

    bi, scores, _plan_ = sel.fused_select_blocks(q, k, select, BLOCK)
    picked = scores.gather(-1, bi.clamp_min(0)).masked_fill(bi < 0, 0.0)
    picked.sum().backward()
    assert seen and seen[-1] is True
