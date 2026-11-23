###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``msa_ds_chunk_store``: the dK/dV kernel's dS stores of ``msa_dq_from_ds`` through one buffer
descriptor per workgroup. An MSA operator kernel option that keeps every bit: tested as a
relaxed-class option (FP32 reference with explicit tolerances, repeated-run identity, the shapes
that break kernels) and, more strictly, bit for bit against ``msa_dq_from_ds`` without it."""

import importlib.util
import os
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron")
pytest.importorskip("flydsl")
pytest.importorskip("triton")

_spec = importlib.util.spec_from_file_location(
    "_dq_from_ds_helpers", os.path.join(os.path.dirname(__file__), "test_minimax_m3_dq_from_ds.py")
)
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)

pytestmark = pytest.mark.skipif(
    "gfx1250" not in H._gcn_arch(), reason="options of the gfx1250 wave32 MSA kernels"
)

BLOCK = H.BLOCK

CASES = [
    # S, B, Hkv, topk, kind, aligned walk
    (4096, 1, 4, 16, "local", True),  # the training shape
    (4096, 1, 4, 16, "random", False),
    (1024, 2, 1, 8, "hot", False),  # B > 1, hot blocks that many queries select
    (1152, 2, 2, 16, "hot", True),  # B > 1 and several KV heads: slices of the dS buffer
    (768, 1, 2, 16, "local", False),  # early queries only: fewer blocks than topk, -1 padded rows
    (900, 1, 2, 4, "random", False),  # S not a multiple of the block
    (641, 1, 1, 4, "random", False),  # odd S: one token per workgroup (no owner to carry)
    (1280, 1, 1, 4, "hostile", False),  # -1 holes, rolled rows
    (4100, 1, 2, 16, "random", True),  # partial last block
]


def _bits_equal(got, want):
    return len(got) == len(want) and all(
        (
            a.shape == b.shape
            and a.dtype == b.dtype
            and torch.equal(a.view(torch.int16), b.view(torch.int16))
            if a.element_size() == 2
            else torch.equal(a.view(torch.int32), b.view(torch.int32))
        )
        for a, b in zip(got, want)
    )


def _walk(table, aligned):
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl import block_walk

    return block_walk._kernel(table) if aligned else None


@pytest.mark.parametrize("S, B, Hkv, topk, kind, aligned", CASES)
def test_backward_ds_chunk_store(S, B, Hkv, topk, kind, aligned):
    """dq / dk / dv with the per-workgroup dS descriptor: bit-identical to ``msa_dq_from_ds``
    without it (five runs), and within the tolerances of that path against the FP32 reference."""
    q, k, v, dout, table = H._inputs(S, B, Hkv, topk, kind)
    walk = _walk(table, aligned)
    ref = H._fp32_reference(q, k, v, dout, table)[2:]
    base = H._backward(q, k, v, dout, table, walk)
    want = H._backward(q, k, v, dout, table, walk, dq_from_ds="force")
    for _ in range(5):
        got = H._backward(q, k, v, dout, table, walk, dq_from_ds="force", ds_chunk_store=True)
        assert _bits_equal(got, want), "chunk store changes bits"
    tol = {"dq": 6e-3, "dk": 6e-3, "dv": 5e-3}
    for name, r, b, g in zip(["dq", "dk", "dv"], ref, base, got):
        assert torch.isfinite(g.float()).all()
        err, err_base = H._rel(g, r), H._rel(b, r)
        assert err <= tol[name], f"{name}: relative error {err:.3e} vs FP32"
        assert err <= 1.25 * err_base + 1e-6, f"{name}: {err:.3e} vs reference kernels {err_base:.3e}"


def test_ds_chunk_store_does_not_read_or_leave_stale_records():
    """The dS buffer is uninitialised memory: the result must not depend on what it held, with
    -1 padded rows, holes and a partial last block (records the dK/dV kernel never writes)."""
    for S, B, Hkv, topk, kind in ((900, 2, 2, 8, "hostile"), (768, 1, 2, 16, "local")):
        q, k, v, dout, table = H._inputs(S, B, Hkv, topk, kind)
        want = H._backward(q, k, v, dout, table, None, dq_from_ds="force", ds_chunk_store=True)
        for fill in (float("nan"), 1e30):
            junk = torch.full((B * Hkv * S * topk, BLOCK, 16), fill, dtype=torch.bfloat16, device="cuda")
            del junk  # back to the allocator: the next dS buffer of this size recycles it
            got = H._backward(q, k, v, dout, table, None, dq_from_ds="force", ds_chunk_store=True)
            assert _bits_equal(got, want)


def test_ds_chunk_store_keeps_per_record_stores_for_large_slices(monkeypatch):
    """A (batch, KV head) slice above the 32-bit limit builds the per-record kernel: decided from
    S, topk and the block size before any launch."""
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl import msa_token_bwd

    q, k, v, dout, table = H._inputs(1024, 1, 2, 4, "local")
    want = H._backward(q, k, v, dout, table, None, dq_from_ds="force")
    monkeypatch.setattr(msa_token_bwd, "_DS_SLICE_MAX_BYTES", 1024 * 4 * BLOCK * 16 * 2 - 1)
    before = set(msa_token_bwd._DKDV_CACHE)
    got = H._backward(q, k, v, dout, table, None, dq_from_ds="force", ds_chunk_store=True)
    assert _bits_equal(got, want)
    assert not any("ds_chunk_store" in key for key in set(msa_token_bwd._DKDV_CACHE) - before)
    monkeypatch.setattr(msa_token_bwd, "_DS_SLICE_MAX_BYTES", 1024 * 4 * BLOCK * 16 * 2)
    got = H._backward(q, k, v, dout, table, None, dq_from_ds="force", ds_chunk_store=True)
    assert _bits_equal(got, want)
    assert any("ds_chunk_store" in key for key in msa_token_bwd._DKDV_CACHE)


def _fake_attention(**config):
    defaults = dict(msa_dq_from_ds=True, msa_ds_chunk_store=True, msa_exact_numerics=False)
    defaults.update(config)
    return SimpleNamespace(config=SimpleNamespace(**defaults), _gfx1250=True, block_size=BLOCK)


def test_switch_defaults_off_and_exact_numerics_turns_it_off():
    from primus.backends.megatron.core.models.minimax_m3.minimax_m3_transformer_config import (
        MSATransformerConfig,
    )
    from primus.backends.megatron.core.transformer.minimax_m3 import attention
    from primus.backends.megatron.core.transformer.minimax_m3.attention import MinimaxSparseAttention

    assert MSATransformerConfig.__dataclass_fields__["msa_ds_chunk_store"].default is False
    assert "msa_ds_chunk_store" in attention._MSA_RELAXED
    A = MinimaxSparseAttention
    assert A._dq_from_ds(_fake_attention(), None) == {"dq_from_ds": True, "ds_chunk_store": True}
    # msa_exact_numerics reproduces the reference kernels: off, whatever the switches say
    assert A._dq_from_ds(_fake_attention(msa_exact_numerics=True), None) == {}
    assert A._dq_from_ds(_fake_attention(msa_ds_chunk_store=False), None) == {"dq_from_ds": True}
    # part of the dq_from_ds path: nothing without it
    assert A._dq_from_ds(_fake_attention(msa_dq_from_ds=False), None) == {}
    other_gpu = _fake_attention()
    other_gpu._gfx1250 = False
    assert A._dq_from_ds(other_gpu, None) == {}
