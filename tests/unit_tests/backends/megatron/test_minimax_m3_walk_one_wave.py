###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``msa_walk_one_wave``: the aligned block walk's Triton kernel launched with one wave per
program (no workgroup barrier in the compiled kernel). An integer kernel: the walk table and the
slots must be the PyTorch reference's, element for element, on every run and at every rows-per-
program value, so everything downstream keeps its bits."""

import importlib.util
import os
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron")
pytest.importorskip("triton")

BLOCK = 128


def _helpers():
    """The table generator of the aligned-walk tests (GPU tables)."""
    spec = importlib.util.spec_from_file_location(
        "_walk_helpers", os.path.join(os.path.dirname(__file__), "test_minimax_m3_aligned_block_walk.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _gfx1250() -> bool:
    return torch.cuda.is_available() and "gfx1250" in torch.cuda.get_device_properties(0).gcnArchName


gpu = pytest.mark.skipif(not _gfx1250(), reason="the aligned walk serves the gfx1250 wave32 MSA kernels")

TABLES = [
    # S, B, Hkv, topk, kind
    (4096, 1, 4, 16, "local"),  # the training shape
    (4096, 1, 4, 16, "random"),
    (4100, 1, 4, 16, "hot"),  # S not a multiple of the block
    (900, 2, 2, 8, "hot"),  # B > 1, hot blocks
    (644, 1, 1, 16, "hostile"),  # -1 holes, early queries with fewer than topk blocks
    (1280, 1, 2, 18, "local"),  # topk not a power of two
    (8192, 1, 4, 16, "local"),
]


@gpu
@pytest.mark.parametrize("S, B, Hkv, topk, kind", TABLES)
@pytest.mark.parametrize("rows", [1, 2, 4])
def test_one_wave_kernel_is_the_reference_permutation(S, B, Hkv, topk, kind, rows):
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl import block_walk

    table = _helpers()._table(B, Hkv, S, topk, kind)
    want_walk, want_slots = block_walk._reference(table)
    base_walk, base_slots = block_walk._kernel(table)
    for _ in range(5):  # integer kernel: every run must give the same permutation
        walk, slots = block_walk._kernel(table, rows=rows, warps=1)
        assert walk.dtype == torch.int32 and slots.dtype == torch.int64
        assert torch.equal(walk, want_walk) and torch.equal(slots, want_slots)
        assert torch.equal(walk, base_walk) and torch.equal(slots, base_slots)


@gpu
def test_dispatcher_checks_and_falls_back(monkeypatch):
    from primus.backends.megatron.core.fusions import bitwise_check
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl import block_walk

    table = _helpers()._table(1, 2, 1024, 8, "local")
    monkeypatch.setattr(bitwise_check, "_results", {})
    want = block_walk.aligned_block_walk(table, BLOCK)
    got = block_walk.aligned_block_walk(table, BLOCK, one_wave=True)
    assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])
    names = [name for name, _ in bitwise_check._results]
    assert names == ["msa_aligned_block_walk", "msa_walk_one_wave"]
    assert all(bitwise_check._results.values())
    # a failed check keeps the four-wave kernel (itself checked), not the score-order walk
    monkeypatch.setattr(bitwise_check, "_results", {})
    real = bitwise_check.passed
    monkeypatch.setattr(
        bitwise_check,
        "passed",
        lambda name, run, ref, key=None: (
            False if name == "msa_walk_one_wave" else real(name, run, ref, key=key)
        ),
    )
    launched = []
    kernel = block_walk._kernel
    monkeypatch.setattr(
        block_walk,
        "_kernel",
        lambda t, rows=16, warps=4: launched.append((rows, warps)) or kernel(t, rows, warps),
    )
    got = block_walk.aligned_block_walk(table, BLOCK, one_wave=True)
    assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])
    assert launched[-1] == (16, 4)


def test_the_walk_asks_for_one_wave(monkeypatch):
    from primus.backends.megatron.core.models.minimax_m3.minimax_m3_transformer_config import (
        MSATransformerConfig,
    )
    from primus.backends.megatron.core.transformer.minimax_m3 import attention
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl import block_walk

    assert "msa_walk_one_wave" not in MSATransformerConfig.__dataclass_fields__
    assert "msa_walk_one_wave" in attention._MSA_GFX1250_PATHS
    assert block_walk._ONE_WAVE_ROWS in (1, 2, 4)
    seen = []
    monkeypatch.setattr(
        block_walk, "aligned_block_walk", lambda table, bs, one_wave=False: seen.append(one_wave)
    )

    def fake(**config):
        defaults = dict(msa_shared_tile_staging=True)
        defaults.update(config)
        return SimpleNamespace(config=SimpleNamespace(**defaults), _gfx1250=True, block_size=BLOCK)

    idx = torch.zeros(1, 1, 8, 4, dtype=torch.int64)
    A = attention.MinimaxSparseAttention
    A._aligned_walk(fake(), idx, None)
    assert seen == [True]
    # no walk without the shared staging, so no walk kernel either
    assert A._aligned_walk(fake(msa_shared_tile_staging=False), idx, None) is None
    assert seen == [True]
