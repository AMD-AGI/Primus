###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``msa_owner_carry``: the gfx1250 MSA forward with each step's tile owner carried in the loop
state. An MSA operator kernel option that keeps every bit: tested as a relaxed-class option (FP32
reference with explicit tolerances, repeated-run identity, the shapes that break kernels) and,
more strictly, bit for bit against the kernel without it."""

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


@pytest.fixture
def force_checks(monkeypatch):
    """Run the kernel under test whatever the self-check says (the dispatcher would run the
    kernel without the option after a failed check, and the test would compare it with itself)."""
    from primus.backends.megatron.core.fusions import bitwise_check

    monkeypatch.setattr(bitwise_check, "passed", lambda *args, **kwargs: True)


def _walk(table, aligned):
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl import block_walk

    return block_walk._kernel(table) if aligned else None


@pytest.mark.parametrize("S, B, Hkv, topk, kind, aligned", CASES)
@pytest.mark.parametrize("slot_lse", [False, True])
def test_forward_owner_carry(force_checks, S, B, Hkv, topk, kind, aligned, slot_lse):
    """o within tolerance of the FP32 reference and no further from it than the kernel without
    the carry; every output bit equal to that kernel's, five times."""
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_token_fwd_w32 import (
        msa_token_fwd_wave32,
    )

    q, k, v, dout, table = H._inputs(S, B, Hkv, topk, kind)
    walk = _walk(table, aligned)
    steps = table if walk is None else walk[0]
    kw = dict(shared_tiles=True, step_mask_skip=True)
    want = msa_token_fwd_wave32(q, k, v, steps, None, BLOCK, slot_lse, **kw)
    ref_o = H._fp32_reference(q, k, v, dout, table)[0]
    for _ in range(5):  # repeats: the result must not depend on scheduling
        got = msa_token_fwd_wave32(q, k, v, steps, None, BLOCK, slot_lse, owner_carry=True, **kw)
        assert _bits_equal(got, want), "owner carry changes bits"
    err, err_base = H._rel(got[0], ref_o), H._rel(want[0], ref_o)
    assert err <= 6e-3 and err <= 1.25 * err_base + 1e-6, f"o: {err:.3e} vs FP32 (without: {err_base:.3e})"


def test_owner_carry_dispatcher_checks_once_and_falls_back(monkeypatch):
    """The dispatcher takes the carry only after a bit-identical first call, per shape."""
    from primus.backends.megatron.core.fusions import bitwise_check
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl import (
        msa_token_fwd_w32,
    )

    q, k, v, _, table = H._inputs(1024, 1, 2, 4, "local")
    calls = []
    real = bitwise_check.passed

    def spy(name, run, reference, key=None):
        if name == "msa_owner_carry forward":
            calls.append(key)
            return False  # a failed check: the kernel without the carry is kept
        return real(name, run, reference, key=key)

    monkeypatch.setattr(bitwise_check, "passed", spy)
    before = set(msa_token_fwd_w32._CACHE)
    want = msa_token_fwd_w32.msa_token_fwd_wave32(q, k, v, table, None, BLOCK, False, shared_tiles=True)
    got = msa_token_fwd_w32.msa_token_fwd_wave32(
        q, k, v, table, None, BLOCK, False, shared_tiles=True, owner_carry=True
    )
    assert len(calls) == 1 and calls[0][0] == 1024
    assert _bits_equal(got, want)
    assert not any("owner_carry" in key for key in set(msa_token_fwd_w32._CACHE) - before)
    # without shared tiles there is no owner: not even checked
    msa_token_fwd_w32.msa_token_fwd_wave32(q, k, v, table, None, BLOCK, False, owner_carry=True)
    assert len(calls) == 1


def _fake_attention(**config):
    defaults = dict(msa_shared_tile_staging=True, msa_exact_numerics=False)
    defaults.update(config)
    return SimpleNamespace(config=SimpleNamespace(**defaults), _gfx1250=True, block_size=BLOCK)


def test_on_by_default_and_exact_numerics_turns_it_off():
    from primus.backends.megatron.core.models.minimax_m3.minimax_m3_transformer_config import (
        MSATransformerConfig,
    )
    from primus.backends.megatron.core.transformer.minimax_m3 import attention
    from primus.backends.megatron.core.transformer.minimax_m3.attention import (
        MinimaxSparseAttention,
    )

    assert "msa_owner_carry" not in MSATransformerConfig.__dataclass_fields__
    assert "msa_owner_carry" in attention._MSA_RELAXED
    A = MinimaxSparseAttention
    assert A._owner_carry(_fake_attention())["owner_carry"] is True
    # msa_exact_numerics reproduces the reference kernels
    assert A._owner_carry(_fake_attention(msa_exact_numerics=True)) == {}
    # no tile owner to carry without the shared staging
    assert A._owner_carry(_fake_attention(msa_shared_tile_staging=False)) == {}
    other_gpu = _fake_attention()
    other_gpu._gfx1250 = False
    assert A._owner_carry(other_gpu) == {}
