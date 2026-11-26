###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``msa_bounded_plan_chunks`` (sums in another order): the BlockPlan cut into chunks of a bounded
size. The rule and its gating on CPU; on gfx1250 the plan's structure (both builders), dK / dV and
the index dK on it against FP32 / FP64 references and the reference chunking with explicit
tolerances, determinism, and the shapes that break kernels (S not a multiple of the block, -1
padded rows, early queries, hot blocks, B > 1)."""


import pytest
import torch

pytest.importorskip("megatron")
pytest.importorskip("flydsl")
pytest.importorskip("triton")

BLOCK = 128


def _gcn_arch() -> str:
    if not torch.cuda.is_available():
        return ""
    return torch.cuda.get_device_properties(0).gcnArchName


needs_gfx1250 = pytest.mark.skipif(
    "gfx1250" not in _gcn_arch(), reason="the BlockPlan and its kernels are the gfx1250 FlyDSL ones"
)


# ---------------------------------------------------------------- the rule and its gating (CPU)


def test_chunk_size_rule():
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl import (
        msa_token_bwd as bwd,
    )

    lo, hi, target = bwd.BOUNDED_CHUNK_MIN, bwd.BOUNDED_CHUNK_MAX, bwd.BOUNDED_CHUNK_TARGET
    count = bwd.BOUNDED_CHUNK_COUNT
    assert 64 <= lo <= hi and lo % 4 == 0 and hi % 4 == 0
    previous = 0
    for S in (128, 900, 4096, 8192, 16384, 32768, 65536):
        for B in (1, 2, 8):
            for topk in (4, 16):
                entries = B * 4 * S * topk
                size = bwd.plan_chunk_size(entries, bounded=True)
                assert size >= lo
                assert size == max(min(hi, max(lo, -(-entries // target))), -(-entries // count))
                # the chunk count, and with it the FP32 partial workspace, stays bounded
                assert -(-entries // size) <= count
                # the reference rule is untouched
                assert bwd.plan_chunk_size(entries) == max(64, -(-entries // 2048))
        size = bwd.plan_chunk_size(4 * S * 16, bounded=True)
        assert size >= previous, "the size never shrinks with S"
        previous = size
    # the two shapes the reference rule serves worst
    assert bwd.plan_chunk_size(4 * 4096 * 16) == 128 and bwd.plan_chunk_size(4 * 32768 * 16) == 1024
    assert bwd.plan_chunk_size(4 * 4096 * 16, bounded=True) == lo
    assert bwd.plan_chunk_size(4 * 32768 * 16, bounded=True) == hi
    assert bwd.plan_chunk_size(2 * 4 * 131072 * 16, bounded=True) == 2 * 4 * 131072 * 16 // count


def test_on_gfx1250_without_a_switch(monkeypatch):
    from primus.backends.megatron.core.models.minimax_m3.minimax_m3_transformer_config import (
        MSATransformerConfig,
    )
    from primus.backends.megatron.core.transformer.minimax_m3 import attention
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl import index_select

    assert "msa_bounded_plan_chunks" not in MSATransformerConfig.__dataclass_fields__
    assert "msa_bounded_plan_chunks" in attention._MSA_GFX1250_PATHS
    monkeypatch.setattr(index_select, "_on_gfx1250", lambda: True)
    assert index_select.bounded_plan_chunks_enabled()
    # other GPUs keep the reference chunking
    monkeypatch.setattr(index_select, "_on_gfx1250", lambda: False)
    assert not index_select.bounded_plan_chunks_enabled()


@needs_gfx1250
def test_plan_past_the_workspace_limit_keeps_the_reference_rule():
    """A table whose bounded plan would need 2 GiB of dK / dV partials or more (8192 rows and more)
    is chunked by the reference rule: decided from the row count and the capacity alone."""
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl import (
        msa_token_bwd as bwd,
    )

    n_rows, capacity = 8192, 8192 * 384 * 4
    ptr = (torch.arange(n_rows + 1, device="cuda", dtype=torch.int32) * 100).contiguous()
    _, chunk_ptr, n_chunks = bwd.plan_chunks(ptr, capacity, bounded=True)
    _, ref_ptr, ref_chunks = bwd.plan_chunks(ptr, capacity)
    assert n_chunks == ref_chunks and torch.equal(chunk_ptr, ref_ptr)
    assert n_chunks * bwd.BOUNDED_CHUNK_WS_BYTES < 2**31
    # one row fewer: the bounded rule again
    _, _, n_small = bwd.plan_chunks(ptr[:-1].contiguous(), capacity, bounded=True)
    assert n_small == n_rows - 1 + -(-capacity // bwd.plan_chunk_size(capacity, bounded=True))


# ---------------------------------------------------------------- gfx1250 (GPU)

PLAN_CASES = [
    # S, B, Hkv, topk, kind
    (4096, 1, 4, 16, "random"),
    (4096, 1, 4, 16, "hot"),  # hot blocks: rows of several chunks
    (4100, 1, 4, 16, "random"),  # S not a multiple of the block
    (900, 2, 2, 8, "hot"),  # B > 1
    (641, 1, 4, 16, "local"),  # fewer blocks than topk: -1 padded rows
    (384, 3, 1, 4, "random"),
    (1280, 1, 1, 4, "hostile"),  # -1 holes, a repeated block
    (16384, 1, 4, 16, "local"),
]


def _check_chunks(plan, size):
    """Every row is covered exactly once by its chunks, in order, each at most ``size`` entries;
    rows without entries keep one empty chunk; chunks past the live ones are empty."""
    ptr, chunk_ptr, chunks = plan.ptr.cpu().long(), plan.chunk_ptr.cpu().long(), plan.chunks.cpu().long()
    n_rows = ptr.numel() - 1
    assert chunk_ptr[0] == 0 and plan.n_chunks == chunks.shape[0] >= int(chunk_ptr[-1])
    per_row = chunk_ptr[1:] - chunk_ptr[:-1]
    want = ((ptr[1:] - ptr[:-1] + size - 1) // size).clamp_min(1)
    assert torch.equal(per_row, want)
    live = chunks[: int(chunk_ptr[-1])]
    rows = torch.repeat_interleave(torch.arange(n_rows), per_row)
    assert torch.equal(live[:, 0], rows)
    first = torch.zeros(live.shape[0], dtype=torch.bool)
    first[chunk_ptr[:-1]] = True
    last = torch.zeros(live.shape[0], dtype=torch.bool)
    last[chunk_ptr[1:] - 1] = True
    assert torch.equal(live[first, 1], ptr[:-1]) and torch.equal(live[last, 2], ptr[1:])
    assert torch.equal(live[1:, 1][~first[1:]], live[:-1, 2][~first[1:]]), "chunks of a row are contiguous"
    length = live[:, 2] - live[:, 1]
    assert int(length.min()) >= 0 and int(length.max()) <= size
    assert bool((length[~last] == size).all()), "only a row's last chunk is short"
    spare = chunks[int(chunk_ptr[-1]) :]
    assert torch.equal(spare[:, 1], spare[:, 2])


@needs_gfx1250
@pytest.mark.parametrize("S, B, Hkv, topk, kind", PLAN_CASES)
def test_plan_same_entries_other_chunks(S, B, Hkv, topk, kind):
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl import (
        msa_token_bwd as bwd,
    )

    from .test_minimax_m3_dq_from_ds import _table

    table = _table(B, Hkv, S, topk, kind, seed=S + topk)
    ref = bwd.build_block_plan(table, BLOCK)
    new = bwd.build_block_plan(table, BLOCK, bounded_chunks=True)
    again = bwd.build_block_plan(table, BLOCK, bounded_chunks=True)
    assert new.n_blocks == ref.n_blocks
    assert torch.equal(new.ptr, ref.ptr)
    live_entries = int(ref.ptr[-1])  # the entry buffer is sized for the capacity; its tail is unused
    assert torch.equal(new.entries[:live_entries], ref.entries[:live_entries])
    entries = B * Hkv * S * topk
    _check_chunks(ref, bwd.plan_chunk_size(entries))
    _check_chunks(new, bwd.plan_chunk_size(entries, bounded=True))
    assert new.n_chunks == again.n_chunks and torch.equal(new.chunk_ptr, again.chunk_ptr)
    live = int(new.chunk_ptr[-1])
    assert torch.equal(new.chunks[:live], again.chunks[:live])


@needs_gfx1250
@pytest.mark.parametrize("S, B, H", [(1000, 1, 4), (4096, 1, 4), (2000, 2, 4), (640, 3, 2)])
def test_fused_selection_plan_matches_build_block_plan(S, B, H):
    """The fused selection (``select_plan``, at most 64 blocks) builds the same bounded plan as
    ``build_block_plan`` does from its table."""
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl import (
        msa_token_bwd as bwd,
    )
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl import select_plan
    from primus.backends.megatron.core.transformer.minimax_m3.indexer import (
        _forced_blocks,
    )

    from .test_minimax_m3_select_plan import _block_scores

    raw, argmax = _block_scores(B, H, S, seed=S)
    n_blocks = raw.shape[-1]
    topk = min(16, n_blocks)
    assert select_plan.supported(B, H, S, n_blocks, topk)
    forced = _forced_blocks(S, n_blocks, BLOCK, 1, 1, raw.device)
    boosted = select_plan.boost_forced(raw, forced)
    topk_scores, topk_idx = boosted.topk(topk, dim=-1)
    for bounded in (False, True):
        bi, _, _, _, _, plan = select_plan.select_post_and_plan(
            topk_scores, topk_idx, argmax, boosted, BLOCK, True, bounded
        )
        ref = bwd.build_block_plan(bi, BLOCK, bounded_chunks=bounded)
        assert plan.n_blocks == ref.n_blocks and plan.n_chunks == ref.n_chunks
        assert torch.equal(plan.ptr, ref.ptr)
        live = int(ref.ptr[-1])
        assert torch.equal(plan.entries[:live], ref.entries[:live])
        assert torch.equal(plan.chunk_ptr, ref.chunk_ptr)
        live_chunks = int(ref.chunk_ptr[-1])
        assert torch.equal(plan.chunks[:live_chunks], ref.chunks[:live_chunks])
        assert torch.equal(plan.chunks[live_chunks:, 1], plan.chunks[live_chunks:, 2])
        _check_chunks(plan, bwd.plan_chunk_size(B * H * S * topk, bounded=bounded))


BWD_CASES = [
    # S, B, Hkv, topk, kind, dq from dS
    (1024, 1, 2, 4, "local", False),
    (1024, 2, 1, 8, "hot", True),  # B > 1, hot blocks that many queries select
    (768, 1, 2, 16, "local", True),  # early queries only: fewer blocks than topk, -1 padded rows
    (900, 1, 2, 4, "random", False),  # S not a multiple of the block
    (641, 1, 1, 4, "random", False),  # odd S
    (1280, 1, 1, 4, "hostile", True),
    (4096, 1, 4, 16, "hot", True),  # the training shape: rows of up to 16 bounded chunks
]


def _backward(q, k, v, dout, table, bounded, **kwargs):
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_token_bwd import (
        build_block_plan,
        msa_token_bwd,
    )
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_token_fwd_w32 import (
        msa_token_fwd_wave32,
    )

    plan = build_block_plan(table, BLOCK, bounded_chunks=bounded)
    out, lse = msa_token_fwd_wave32(q, k, v, table, None, BLOCK, False, shared_tiles=True)
    return msa_token_bwd(dout, q, k, v, out, lse, table, None, BLOCK, plan=plan, shared_tiles=True, **kwargs)


@needs_gfx1250
@pytest.mark.parametrize("S, B, Hkv, topk, kind, from_ds", BWD_CASES)
def test_dk_dv_match_fp32_and_the_reference_chunking(S, B, Hkv, topk, kind, from_ds):
    """Relaxed numerics: dk and dv within an explicit tolerance of the FP32 reference, no further
    from it than 1.25 x the reference chunking (plus 1e-6), within 1e-3 of the reference chunking,
    bit-identical between two runs; dq does not depend on the chunking at all."""
    from .test_minimax_m3_dq_from_ds import _fp32_reference, _inputs, _rel

    q, k, v, dout, table = _inputs(S, B, Hkv, topk, kind)
    ref = _fp32_reference(q, k, v, dout, table)[2:]
    kwargs = {"dq_from_ds": "force"} if from_ds else {}
    base = _backward(q, k, v, dout, table, False, **kwargs)
    got = _backward(q, k, v, dout, table, True, **kwargs)
    again = _backward(q, k, v, dout, table, True, **kwargs)
    assert torch.equal(got[0], base[0]), "dq changed with the chunking"
    tol = {"dk": 6e-3, "dv": 5e-3}
    for name, r, b, g, a in zip(["dk", "dv"], ref[1:], base[1:], got[1:], again[1:]):
        assert torch.equal(g, a), f"{name}: two runs differ"
        assert torch.isfinite(g.float()).all()
        err, err_base = _rel(g, r), _rel(b, r)
        assert err <= tol[name], f"{name}: relative error {err:.3e} vs FP32"
        assert err <= 1.25 * err_base + 1e-6, f"{name}: {err:.3e} vs reference chunking {err_base:.3e}"
        assert _rel(g, b) <= 1e-3, f"{name}: {_rel(g, b):.3e} from the reference chunking"


INDEX_CASES = [
    (4096, 1, 4, 16, "random", "minus_one", False),
    (4096, 1, 4, 16, "hot", "valid", True),  # hot blocks and hot keys
    (4096, 2, 4, 16, "local", "minus_one", True),  # B > 1
    (4100, 1, 4, 16, "random", "minus_one", True),  # S not a multiple of the block
    (641, 1, 4, 16, "local", "minus_one", False),  # fewer blocks than topk: -1 padded rows
    (384, 3, 1, 4, "random", "valid", False),
]


@needs_gfx1250
@pytest.mark.parametrize("S, B, H, topk, kind, dead_keys, scatter", INDEX_CASES)
def test_index_dk_matches_fp64_and_the_reference_chunking(
    monkeypatch, S, B, H, topk, kind, dead_keys, scatter
):
    """d index k on the bounded plan (reference kernel and ``msa_index_dk_scatter``'s): no further
    from the FP64 reference than 1.25 x the reference chunking (plus 1e-6), within 1e-3 of it, and
    bit-identical between two runs."""
    import primus.backends.megatron.core.transformer.minimax_m3.flydsl.index_block_max as ibm
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_token_bwd import (
        build_block_plan,
    )

    from .test_minimax_m3_dq_from_ds import _rel
    from .test_minimax_m3_index_dk_scatter import _dense_reference, _inputs

    table, g, keys, q = _inputs(S, B, H, topk, kind, seed=S + topk, dead_keys=dead_keys)
    if scatter:
        if not ibm.index_dk_scatter_supported(q.shape[-1], BLOCK):
            pytest.skip(f"msa_index_dk_scatter does not serve index dim {q.shape[-1]}")
        monkeypatch.setattr(ibm, "_dk_scatter_checked", lambda *a, **k: True)
    ref = _dense_reference(table, g, keys, q)
    kwargs = {"scatter": True} if scatter else {}
    base = ibm.index_dk(build_block_plan(table, BLOCK), g, keys, q, BLOCK, **kwargs)
    got = ibm.index_dk(build_block_plan(table, BLOCK, bounded_chunks=True), g, keys, q, BLOCK, **kwargs)
    again = ibm.index_dk(build_block_plan(table, BLOCK, bounded_chunks=True), g, keys, q, BLOCK, **kwargs)
    assert got.shape == base.shape and got.dtype == base.dtype
    assert torch.equal(got, again), "two runs differ"
    assert torch.isfinite(got.float()).all()
    r = ref.reshape(got.shape)
    err, err_base = _rel(got, r), _rel(base, r)
    assert err <= 1.25 * err_base + 1e-6, f"{err:.3e} vs reference chunking {err_base:.3e}"
    assert _rel(got, base) <= 1e-3, f"{_rel(got, base):.3e} from the reference chunking"
