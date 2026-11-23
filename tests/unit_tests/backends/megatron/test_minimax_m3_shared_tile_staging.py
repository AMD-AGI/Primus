###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``msa_shared_tile_staging``: the gfx1250 MSA forward and dQ kernels with several adjacent
tokens per workgroup against the one-token-per-workgroup kernels, bit for bit."""

import pytest
import torch

pytest.importorskip("megatron")
pytest.importorskip("flydsl")


def _gcn_arch() -> str:
    if not torch.cuda.is_available():
        return ""
    return torch.cuda.get_device_properties(0).gcnArchName


pytestmark = pytest.mark.skipif(
    "gfx1250" not in _gcn_arch(), reason="shared tile staging is in the gfx1250 wave32 MSA kernels"
)

BLOCK, D = 128, 128


def _table(B, Hkv, S, topk, gen, kind):
    """Causal, left-packed, -1 padded selections. ``random``: independent per token (few shared
    tiles); ``local``: neighbouring tokens pick nearly the same blocks, as trained indexers do
    (many shared tiles); ``same``: one selection per query block (every tile shared)."""
    n_blocks = -(-S // BLOCK)
    qb = torch.arange(S, device="cuda") // BLOCK
    if kind == "random":
        scores = torch.randn(B, Hkv, S, n_blocks, device="cuda", generator=gen)
    else:
        per_block = torch.randn(B, Hkv, n_blocks, n_blocks, device="cuda", generator=gen)
        scores = per_block[:, :, qb]
        if kind == "local":
            scores = scores + 0.3 * torch.randn(B, Hkv, S, n_blocks, device="cuda", generator=gen)
    future = torch.arange(n_blocks, device="cuda")[None, :] > qb[:, None]
    scores = scores.masked_fill(future, float("-inf"))
    scores.scatter_(-1, qb.view(1, 1, S, 1).expand(B, Hkv, S, 1), float("inf"))
    k = min(topk, n_blocks)
    vals, idx = scores.topk(k, dim=-1)
    idx = idx.masked_fill(vals == float("-inf"), -1)
    if k < topk:
        idx = torch.nn.functional.pad(idx, (0, topk - k), value=-1)
    return idx.to(torch.int32).contiguous()


def _inputs(S, B, Hkv, topk, kind, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    Hq = 16 * Hkv
    q = torch.randn(S, B, Hq, D, device="cuda", dtype=torch.bfloat16, generator=gen)
    k = torch.randn(S, B, Hkv, D, device="cuda", dtype=torch.bfloat16, generator=gen)
    v = torch.randn(S, B, Hkv, D, device="cuda", dtype=torch.bfloat16, generator=gen)
    dout = torch.randn(S, B, Hq, D, device="cuda", dtype=torch.bfloat16, generator=gen)
    return q, k, v, dout, _table(B, Hkv, S, topk, gen, kind)


def _bits(t):
    return t.view({2: torch.int16, 4: torch.int32}[t.element_size()])


def _same(got, want):
    return len(got) == len(want) and all(
        a.shape == b.shape and a.dtype == b.dtype and torch.equal(_bits(a), _bits(b)) for a, b in zip(got, want)
    )


@pytest.fixture
def force_shared(monkeypatch):
    """Run the shared-staging kernels whatever the self-check says: these tests compare the
    kernels themselves; the dispatchers would run the one-token kernels after a failed check."""
    from primus.backends.megatron.core.fusions import bitwise_check

    monkeypatch.setattr(bitwise_check, "passed", lambda *args, **kwargs: True)


CASES = [
    # S, B, Hkv, topk, table kind
    (4096, 1, 4, 16, "local"),
    (4096, 1, 4, 16, "random"),
    (4096, 1, 4, 16, "same"),
    (1024, 2, 2, 4, "local"),
    (8192, 1, 2, 16, "local"),
    (640, 3, 1, 8, "random"),
    (384, 1, 2, 16, "same"),  # fewer blocks than topk: -1 padded rows
]


@pytest.mark.parametrize("S, B, Hkv, topk, kind", CASES)
@pytest.mark.parametrize("slot_lse", [False, True])
def test_forward_kernel_bitwise(require_bitwise, force_shared, S, B, Hkv, topk, kind, slot_lse):
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_token_fwd_w32 import (
        fwd_tokens_per_wg,
        msa_token_fwd_wave32,
    )

    assert fwd_tokens_per_wg(S, BLOCK, "lpt") == 4
    q, k, v, _, table = _inputs(S, B, Hkv, topk, kind)
    want = msa_token_fwd_wave32(q, k, v, table, None, BLOCK, slot_lse, shared_tiles=False)
    ok = True
    for _ in range(20):  # repeats: the workgroup's waves race on nothing
        got = msa_token_fwd_wave32(q, k, v, table, None, BLOCK, slot_lse, shared_tiles=True)
        ok = ok and _same(got, want)
    require_bitwise(ok, f"msa_shared_tile_staging forward {S, B, Hkv, topk, kind, slot_lse}")


@pytest.mark.parametrize("S, B, Hkv, topk, kind", CASES)
def test_backward_kernel_bitwise(require_bitwise, force_shared, S, B, Hkv, topk, kind):
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_token_bwd import msa_token_bwd
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_token_bwd_w32 import (
        dq_tokens_per_wg,
    )
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_token_fwd_w32 import (
        msa_token_fwd_wave32,
    )

    assert dq_tokens_per_wg(S, BLOCK) == 2
    q, k, v, dout, table = _inputs(S, B, Hkv, topk, kind, seed=1)
    out, lse = msa_token_fwd_wave32(q, k, v, table, None, BLOCK, False)
    want = msa_token_bwd(dout, q, k, v, out, lse, table, None, BLOCK, shared_tiles=False)
    ok = True
    for _ in range(20):
        got = msa_token_bwd(dout, q, k, v, out, lse, table, None, BLOCK, shared_tiles=True)
        ok = ok and _same(got, want)  # dq, and dk / dv, which read the dQ kernel's delta
    require_bitwise(ok, f"msa_shared_tile_staging dQ {S, B, Hkv, topk, kind}")


def _hostile_table(B, Hkv, S, topk, seed=3):
    """Tables no indexer makes: -1 in the middle and at the start of a row, and the same blocks at
    different slots of neighbouring tokens (so the waves of a workgroup want a tile at different
    steps). Still causal, each row keeps its own block. Table contents must not change how many
    steps (barriers) a wave runs."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    table = _table(B, Hkv, S, topk, gen, "local")  # slot 0 is the token's own block
    holes = torch.rand(B, Hkv, S, topk, device="cuda", generator=gen) < 0.3
    holes[..., 0] = False
    mixed = table.masked_fill(holes, -1)
    mixed[:, :, 1::2] = mixed[:, :, 1::2].roll(1, dims=-1)
    mixed[:, :, 2::4] = mixed[:, :, 2::4].roll(3, dims=-1)
    # only for rows whose topk slots are all read (the kernels read the first min(visible, topk)
    # slots): earlier rows stay left-packed, so every token keeps a visible key
    full = (torch.arange(S, device="cuda") // BLOCK + 1 >= topk).view(1, 1, S, 1)
    return torch.where(full, mixed, table).contiguous()


def _run_both(q, k, v, dout, table, slot_lse=False, repeats=20):
    """(ok, forward outputs): the shared-staging entries against the one-token kernels, bit for bit."""
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_token_bwd import msa_token_bwd
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_token_fwd_w32 import (
        msa_token_fwd_wave32,
    )

    want = msa_token_fwd_wave32(q, k, v, table, None, BLOCK, slot_lse, shared_tiles=False)
    out, lse = want[0], want[1]
    want_g = msa_token_bwd(dout, q, k, v, out, lse, table, None, BLOCK, shared_tiles=False)
    ok = True
    for _ in range(repeats):
        got = msa_token_fwd_wave32(q, k, v, table, None, BLOCK, slot_lse, shared_tiles=True)
        got_g = msa_token_bwd(dout, q, k, v, out, lse, table, None, BLOCK, shared_tiles=True)
        ok = ok and _same(got, want) and _same(got_g, want_g)
    return ok


@pytest.mark.parametrize("S, B, Hkv, topk", [(4096, 1, 4, 16), (2048, 2, 2, 8), (1024, 1, 1, 4)])
@pytest.mark.parametrize("slot_lse", [False, True])
def test_hostile_tables_bitwise(require_bitwise, force_shared, S, B, Hkv, topk, slot_lse):
    q, k, v, dout, _ = _inputs(S, B, Hkv, topk, "random", seed=4)
    table = _hostile_table(B, Hkv, S, topk)
    ok = _run_both(q, k, v, dout, table, slot_lse)
    require_bitwise(ok, f"msa_shared_tile_staging hostile table {S, B, Hkv, topk, slot_lse}")


# S the group does not divide. S % 4 == 2: the forward keeps one token per workgroup, dQ shares
# with 2, and the last query block is partial (keys past S are zero-filled). S odd: both keep one
# token per workgroup. S % 4 == 0 but not a multiple of the block: both share, partial last block.
@pytest.mark.parametrize(
    "S, fwd_tpw, dq_tpw", [(4098, 1, 2), (642, 1, 2), (641, 1, 1), (4097, 1, 1), (644, 4, 2), (4100, 4, 2)]
)
@pytest.mark.parametrize("kind", ["local", "random"])
def test_sequence_the_group_does_not_divide(require_bitwise, force_shared, S, fwd_tpw, dq_tpw, kind):
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl import msa_token_bwd as bwd
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl import msa_token_bwd_w32, msa_token_fwd_w32

    assert msa_token_fwd_w32.fwd_tokens_per_wg(S, BLOCK, "lpt") == fwd_tpw
    assert msa_token_bwd_w32.dq_tokens_per_wg(S, BLOCK) == dq_tpw
    B, Hkv, topk = 1, 2, 16
    q, k, v, dout, table = _inputs(S, B, Hkv, topk, kind, seed=5)
    fwd_before = {key for key in msa_token_fwd_w32._CACHE if key[-1] > 1}
    dq_before = {key for key in bwd._DQ_CACHE if key[-1] > 1}
    ok = _run_both(q, k, v, dout, table, repeats=5)
    require_bitwise(ok, f"msa_shared_tile_staging S={S} {kind}")
    # the fallback launches the one-token kernel and builds nothing else
    scale = float(D**-0.5)
    if fwd_tpw == 1:
        assert {key for key in msa_token_fwd_w32._CACHE if key[-1] > 1} == fwd_before
    else:
        assert (Hkv, topk, BLOCK, scale, "lpt", False, fwd_tpw) in msa_token_fwd_w32._CACHE
    if dq_tpw == 1:
        assert {key for key in bwd._DQ_CACHE if key[-1] > 1} == dq_before
    else:
        assert (Hkv, topk, BLOCK, scale, dq_tpw) in bwd._DQ_CACHE


def test_lockstep_kernels_refuse_a_sequence_they_do_not_divide():
    """The barrier count is workgroup-uniform only when the group divides S: nothing may launch
    the lockstep layout otherwise. The guard sits at the launch, behind the eligibility helpers."""
    from primus.backends.megatron.core.fusions import bitwise_check
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl import msa_token_bwd as bwd
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl import msa_token_bwd_w32, msa_token_fwd_w32

    q, k, v, dout, table = _inputs(642, 1, 2, 4, "local", seed=6)
    out, lse = msa_token_fwd_w32.msa_token_fwd_wave32(q, k, v, table, None, BLOCK, False)
    saved = dict(bitwise_check._results)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(bitwise_check, "passed", lambda *args, **kwargs: True)
        mp.setattr(msa_token_fwd_w32, "fwd_tokens_per_wg", lambda S, block, remap: 4)  # a broken helper
        with pytest.raises(ValueError, match="msa_shared_tile_staging forward"):
            msa_token_fwd_w32.msa_token_fwd_wave32(q, k, v, table, None, BLOCK, False, shared_tiles=True)
        mp.setattr(msa_token_bwd_w32, "dq_tokens_per_wg", lambda S, block: 4)
        with pytest.raises(ValueError, match="msa_shared_tile_staging dQ"):
            bwd.msa_token_bwd(dout, q, k, v, out, lse, table, None, BLOCK, shared_tiles=True)
    assert bitwise_check._results == saved


def _real_tables():
    """Block tables dumped from a training step (``PRIMUS_MSA_BLOCK_TABLE_DUMP``: a ``torch.save``d
    list of ``[B, Hkv, S, topk]`` integer tensors, one per MSA call)."""
    import os

    path = os.environ.get("PRIMUS_MSA_BLOCK_TABLE_DUMP", "")
    if not path or not os.path.exists(path):
        return None
    return [t.to(torch.int32) for t in torch.load(path, map_location="cpu")]


def test_block_tables_of_a_training_step_bitwise(require_bitwise, force_shared):
    tables = _real_tables()
    if tables is None:
        pytest.skip("set PRIMUS_MSA_BLOCK_TABLE_DUMP to a dump of the block tables of a training step")
    for i, table in enumerate(tables):
        table = table.cuda().contiguous()
        B, Hkv, S, topk = table.shape
        q, k, v, dout, _ = _inputs(S, B, Hkv, topk, "random", seed=7 + i)
        for slot_lse in (False, True):
            ok = _run_both(q, k, v, dout, table, slot_lse)
            require_bitwise(ok, f"msa_shared_tile_staging training-step table {i} {tuple(table.shape)} {slot_lse}")


def test_odd_sequence_keeps_one_token_per_workgroup():
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_token_bwd_w32 import (
        dq_tokens_per_wg,
    )
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_token_fwd_w32 import (
        fwd_tokens_per_wg,
    )

    assert fwd_tokens_per_wg(4097, BLOCK, "lpt") == 1 and fwd_tokens_per_wg(4098, BLOCK, "lpt") == 1
    assert fwd_tokens_per_wg(4096, BLOCK, "none") == 1
    assert dq_tokens_per_wg(4097, BLOCK) == 1 and dq_tokens_per_wg(4098, BLOCK) == 2


def test_dispatchers_check_once_per_shape_and_fall_back(monkeypatch):
    from primus.backends.megatron.core.fusions import bitwise_check
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl import msa_token_fwd_w32
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_token_bwd import msa_token_bwd

    saved = dict(bitwise_check._results)
    bitwise_check._results.clear()
    try:
        q, k, v, dout, table = _inputs(512, 1, 2, 4, "local", seed=2)
        want = msa_token_fwd_w32.msa_token_fwd_wave32(q, k, v, table, None, BLOCK, False)
        assert bitwise_check._results == {}  # the switch is off: nothing checked
        for _ in range(2):
            got = msa_token_fwd_w32.msa_token_fwd_wave32(q, k, v, table, None, BLOCK, False, shared_tiles=True)
            assert _same(got, want)
        want_g = msa_token_bwd(dout, q, k, v, want[0], want[1], table, None, BLOCK)
        got_g = msa_token_bwd(dout, q, k, v, want[0], want[1], table, None, BLOCK, shared_tiles=True)
        assert _same(got_g, want_g)
        names = sorted(name for name, _ in bitwise_check._results)
        assert names == ["msa_shared_tile_staging dQ", "msa_shared_tile_staging forward"]

        # a failed check keeps the one-token kernels: the shared builder must not run again
        bitwise_check._results.clear()
        monkeypatch.setattr(bitwise_check, "same_bits", lambda a, b: False)
        got = msa_token_fwd_w32.msa_token_fwd_wave32(q, k, v, table, None, BLOCK, False, shared_tiles=True)
        assert _same(got, want)
        assert list(bitwise_check._results.values()) == [False]
        got_g = msa_token_bwd(dout, q, k, v, want[0], want[1], table, None, BLOCK, shared_tiles=True)
        assert _same(got_g, want_g)
        assert list(bitwise_check._results.values()) == [False, False]
    finally:
        bitwise_check._results.clear()
        bitwise_check._results.update(saved)


def test_switch_defaults_to_off():
    from primus.backends.megatron.core.models.minimax_m3.minimax_m3_transformer_config import (
        MSATransformerConfig,
    )

    assert MSATransformerConfig.__dataclass_fields__["msa_shared_tile_staging"].default is False
