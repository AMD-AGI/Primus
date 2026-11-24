###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``msa_dq_early_issue``: the gfx1250 slim dQ kernel (``msa_dq_from_ds``) with the next step's
copies issued before the wait for the current step's. An MSA operator kernel option that keeps
every bit: tested as a relaxed-class option (FP32 reference with explicit tolerances, repeated-run
identity, the shapes that break kernels) and, more strictly, bit for bit against the kernel
without it (``require_bitwise``)."""

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
    (1152, 2, 2, 16, "hot", True),  # B > 1 and several KV heads
    (768, 1, 2, 16, "local", False),  # early queries only: fewer blocks than topk, -1 padded rows
    (900, 1, 2, 4, "random", False),  # S not a multiple of the block
    (641, 1, 1, 4, "random", False),  # odd S: one token per workgroup (no lockstep, option unused)
    (1280, 1, 1, 4, "hostile", False),  # -1 holes, rolled rows
    (4100, 1, 2, 16, "random", True),  # partial last block
    (4096, 1, 1, 16, "zipf", True),  # a text-like selection: few hot blocks, a long tail
]


def _inputs(S, B, Hkv, topk, kind):
    if kind != "zipf":
        return H._inputs(S, B, Hkv, topk, kind)
    # block popularity ~ 1 / rank among the blocks a query may see, plus its own block
    q, k, v, dout, _ = H._inputs(S, B, Hkv, topk, "local")
    g = torch.Generator().manual_seed(7)
    nb = -(-S // BLOCK)
    table = torch.full((B, Hkv, S, topk), -1, dtype=torch.int32)
    weight = 1.0 / torch.arange(1, nb + 1, dtype=torch.float64)
    for qb in range(nb):
        rows = slice(qb * BLOCK, min((qb + 1) * BLOCK, S))
        n = rows.stop - rows.start
        take = min(topk, qb + 1)
        for b in range(B):
            for h in range(Hkv):
                if take > 1:
                    w = weight[:qb].expand(n, qb)
                    picks = torch.multinomial(w, take - 1, replacement=False, generator=g)
                    table[b, h, rows, 1:take] = picks.to(torch.int32)
                table[b, h, rows, 0] = qb
    return q, k, v, dout, table.to(q.device).contiguous()


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
def test_backward_dq_early_issue(force_checks, require_bitwise, S, B, Hkv, topk, kind, aligned):
    """dq / dk / dv with the early issue: bit-identical to ``msa_dq_from_ds`` without it (five
    runs), and within the tolerances of that path against the FP32 reference."""
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl import msa_token_bwd

    q, k, v, dout, table = _inputs(S, B, Hkv, topk, kind)
    walk = _walk(table, aligned)
    ref = H._fp32_reference(q, k, v, dout, table)[2:]
    base = H._backward(q, k, v, dout, table, walk)
    kw = dict(dq_from_ds="force", ds_chunk_store=True)
    want = H._backward(q, k, v, dout, table, walk, **kw)
    for _ in range(5):  # repeats: the result must not depend on scheduling
        got = H._backward(q, k, v, dout, table, walk, dq_early_issue=True, **kw)
        require_bitwise(_bits_equal(got, want), "msa_dq_early_issue")
    built = any("early_issue" in key for key in msa_token_bwd._DQ_CACHE)
    assert built or S % 4, "the early-issue kernel was never built"
    tol = {"dq": 6e-3, "dk": 6e-3, "dv": 5e-3}
    for name, r, b, g in zip(["dq", "dk", "dv"], ref, base, got):
        assert torch.isfinite(g.float()).all()
        err, err_base = H._rel(g, r), H._rel(b, r)
        assert err <= tol[name], f"{name}: relative error {err:.3e} vs FP32"
        assert err <= 1.25 * err_base + 1e-6, f"{name}: {err:.3e} vs reference kernels {err_base:.3e}"


def test_dq_early_issue_repeats_with_every_wave_copying(force_checks, require_bitwise):
    """The kernel waits for a step's copies by count (all but the two issued last), which relies
    on a wave's copies completing in issue order. Forty runs on a table without shared tiles
    (every wave owns and copies its K tile, each from another block) must give the same bits."""
    q, k, v, dout, table = H._inputs(4096, 1, 4, 16, "random")
    kw = dict(dq_from_ds="force", ds_chunk_store=True)
    want = H._backward(q, k, v, dout, table, None, **kw)
    for _ in range(40):
        got = H._backward(q, k, v, dout, table, None, dq_early_issue=True, **kw)
        require_bitwise(_bits_equal(got, want), "msa_dq_early_issue (repeat)")


def test_dq_early_issue_dispatcher_checks_once_and_falls_back(monkeypatch):
    """The dispatcher takes the option only after a bit-identical first call, per shape."""
    from primus.backends.megatron.core.fusions import bitwise_check
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl import msa_token_bwd

    q, k, v, dout, table = H._inputs(1024, 1, 2, 4, "local")
    calls = []
    real = bitwise_check.passed

    def spy(name, run, reference, key=None):
        if name == "msa_dq_early_issue dQ":
            calls.append(key)
            return False  # a failed check: the kernel without the option is kept
        return real(name, run, reference, key=key)

    monkeypatch.setattr(bitwise_check, "passed", spy)
    kw = dict(dq_from_ds="force", ds_chunk_store=True)
    want = H._backward(q, k, v, dout, table, None, **kw)
    before = set(msa_token_bwd._DQ_CACHE)
    got = H._backward(q, k, v, dout, table, None, dq_early_issue=True, **kw)
    assert len(calls) == 1 and calls[0][0] == 1024
    assert _bits_equal(got, want)
    assert not any("early_issue" in key for key in set(msa_token_bwd._DQ_CACHE) - before)
    # one token per workgroup (odd S): no lockstep, not even checked
    q, k, v, dout, table = H._inputs(641, 1, 1, 4, "random")
    H._backward(q, k, v, dout, table, None, dq_early_issue=True, **kw)
    assert len(calls) == 1


def test_dq_early_issue_self_check_passes(require_bitwise):
    """The real self-check, on the training shape with an aligned walk."""
    from primus.backends.megatron.core.fusions import bitwise_check

    q, k, v, dout, table = H._inputs(4096, 1, 4, 16, "local")
    walk = _walk(table, True)
    H._backward(q, k, v, dout, table, walk, dq_from_ds="force", ds_chunk_store=True, dq_early_issue=True)
    decided = [ok for (name, _), ok in bitwise_check._results.items() if name == "msa_dq_early_issue dQ"]
    assert decided, "the self-check did not run"
    require_bitwise(all(decided), "msa_dq_early_issue")


def _fake_attention(**config):
    defaults = dict(
        msa_dq_from_ds=True, msa_ds_chunk_store=True, msa_dq_early_issue=True, msa_exact_numerics=False
    )
    defaults.update(config)
    return SimpleNamespace(config=SimpleNamespace(**defaults), _gfx1250=True, block_size=BLOCK)


def test_switch_defaults_off_and_exact_numerics_turns_it_off():
    from primus.backends.megatron.core.models.minimax_m3.minimax_m3_transformer_config import (
        MSATransformerConfig,
    )
    from primus.backends.megatron.core.transformer.minimax_m3 import attention
    from primus.backends.megatron.core.transformer.minimax_m3.attention import MinimaxSparseAttention

    assert MSATransformerConfig.__dataclass_fields__["msa_dq_early_issue"].default is False
    assert "msa_dq_early_issue" in attention._MSA_RELAXED
    A = MinimaxSparseAttention
    on = {"dq_from_ds": True, "ds_chunk_store": True, "dq_early_issue": True}
    assert A._dq_from_ds(_fake_attention(), None) == on
    # msa_exact_numerics reproduces the reference kernels: off, whatever the switches say
    assert A._dq_from_ds(_fake_attention(msa_exact_numerics=True), None) == {}
    off = {"dq_from_ds": True, "ds_chunk_store": True}
    assert A._dq_from_ds(_fake_attention(msa_dq_early_issue=False), None) == off
    # part of the dq_from_ds path: nothing without it
    assert A._dq_from_ds(_fake_attention(msa_dq_from_ds=False), None) == {}
    other_gpu = _fake_attention()
    other_gpu._gfx1250 = False
    assert A._dq_from_ds(other_gpu, None) == {}
