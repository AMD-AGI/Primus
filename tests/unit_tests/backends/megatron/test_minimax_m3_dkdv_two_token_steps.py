###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``msa_dkdv_two_token_steps``: the gfx1250 dK/dV kernel with two tokens per step instead of
four. An MSA operator kernel option that keeps every bit: tested as a relaxed-class option (FP32
reference with explicit tolerances, repeated-run identity, the shapes that break kernels) and,
more strictly, bit for bit against the four-token kernel (``require_bitwise``), on the reference
dQ path and on the ``msa_dq_from_ds`` path (whose dS records the kernel also writes)."""

import importlib.util
import os
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron")
pytest.importorskip("flydsl")
pytest.importorskip("triton")

_here = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location(
    "_dq_from_ds_helpers", os.path.join(_here, "test_minimax_m3_dq_from_ds.py")
)
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)
_spec = importlib.util.spec_from_file_location(
    "_dq_early_issue_helpers", os.path.join(_here, "test_minimax_m3_dq_early_issue.py")
)
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)

pytestmark = pytest.mark.skipif(
    "gfx1250" not in H._gcn_arch(), reason="options of the gfx1250 wave32 MSA kernels"
)

BLOCK = H.BLOCK
DS = dict(dq_from_ds="force", ds_chunk_store=True, dq_early_issue=True)

CASES = [
    # S, B, Hkv, topk, kind, aligned walk, keyword arguments of the path under the option
    (4096, 1, 4, 16, "local", True, DS),  # the training shape and path
    (4096, 1, 4, 16, "random", False, {}),  # the reference dQ path
    (1024, 2, 1, 8, "hot", False, DS),  # B > 1, hot blocks that many queries select
    (1152, 2, 2, 16, "hot", True, {}),  # B > 1 and several KV heads
    (768, 1, 2, 16, "local", False, DS),  # early queries only: fewer blocks than topk, -1 padded rows
    (900, 1, 2, 4, "random", False, DS),  # S not a multiple of the block
    (641, 1, 1, 4, "random", False, {}),  # odd S: not a multiple of the tile either
    (1280, 1, 1, 4, "hostile", False, DS),  # -1 holes, rolled rows
    (4100, 1, 2, 16, "random", True, DS),  # partial last block
    (4096, 1, 1, 16, "zipf", True, DS),  # a text-like selection: few hot blocks, a long tail
    (130, 3, 1, 4, "local", False, {}),  # chunks of one, two and three entries (odd step tails)
]


@pytest.mark.parametrize("S, B, Hkv, topk, kind, aligned, path", CASES)
def test_backward_two_token_steps(E_force, require_bitwise, S, B, Hkv, topk, kind, aligned, path):
    """dq / dk / dv with two tokens per step: bit-identical to the four-token kernel on the same
    path (three runs), and within that path's tolerances against the FP32 reference."""
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl import msa_token_bwd

    q, k, v, dout, table = E._inputs(S, B, Hkv, topk, kind)
    walk = E._walk(table, aligned)
    ref = H._fp32_reference(q, k, v, dout, table)[2:]
    base = H._backward(q, k, v, dout, table, walk)
    want = H._backward(q, k, v, dout, table, walk, **path)
    for _ in range(3):  # repeats: the result must not depend on scheduling
        got = H._backward(q, k, v, dout, table, walk, dkdv_two_token_steps=True, **path)
        require_bitwise(E._bits_equal(got, want), "msa_dkdv_two_token_steps")
    assert any("two_token_steps" in key for key in msa_token_bwd._DKDV_CACHE), "variant never built"
    tol = {"dq": 6e-3, "dk": 6e-3, "dv": 5e-3}
    for name, r, b, g in zip(["dq", "dk", "dv"], ref, base, got):
        assert torch.isfinite(g.float()).all()
        err, err_base = H._rel(g, r), H._rel(b, r)
        assert err <= tol[name], f"{name}: relative error {err:.3e} vs FP32"
        assert err <= 1.25 * err_base + 1e-6, f"{name}: {err:.3e} vs reference kernels {err_base:.3e}"


@pytest.fixture
def E_force(monkeypatch):
    """Run the kernel under test whatever the self-check says (after a failed check the
    dispatcher runs the four-token kernel, and the test would compare it with itself)."""
    from primus.backends.megatron.core.fusions import bitwise_check

    monkeypatch.setattr(bitwise_check, "passed", lambda *args, **kwargs: True)


@pytest.mark.parametrize("bounded", [False, True])
def test_two_token_steps_under_both_chunk_rules(E_force, require_bitwise, bounded):
    """Chunks of the reference rule and of ``msa_bounded_plan_chunks`` (other trip counts)."""
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl import msa_token_bwd

    from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_token_fwd_w32 import (
        msa_token_fwd_wave32,
    )

    q, k, v, dout, table = H._inputs(4096, 2, 2, 16, "hot")
    plan = msa_token_bwd.build_block_plan(table, BLOCK, bounded)
    out, lse = msa_token_fwd_wave32(q, k, v, table, None, BLOCK, False, shared_tiles=True)

    def backward(**kwargs):
        return msa_token_bwd.msa_token_bwd(
            dout, q, k, v, out, lse, table, None, BLOCK, plan=plan, shared_tiles=True, **kwargs
        )

    want = backward()
    for _ in range(3):
        got = backward(dkdv_two_token_steps=True)
        require_bitwise(E._bits_equal(got, want), "msa_dkdv_two_token_steps (chunk rule)")


def test_two_token_steps_dispatcher_checks_once_and_falls_back(monkeypatch):
    """The dispatcher takes the option only after a bit-identical first call, per shape, and
    asks nothing where the option is not eligible."""
    from primus.backends.megatron.core.fusions import bitwise_check
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl import msa_token_bwd

    q, k, v, dout, table = H._inputs(1024, 1, 2, 4, "local")
    calls = []
    real = bitwise_check.passed

    def spy(name, run, reference, key=None):
        if name == "msa_dkdv_two_token_steps dK/dV":
            calls.append(key)
            return False  # a failed check: the four-token kernel is kept
        return real(name, run, reference, key=key)

    monkeypatch.setattr(bitwise_check, "passed", spy)
    want = H._backward(q, k, v, dout, table, None)
    before = set(msa_token_bwd._DKDV_CACHE)
    got = H._backward(q, k, v, dout, table, None, dkdv_two_token_steps=True)
    assert len(calls) == 1 and calls[0][0] == 1024
    assert E._bits_equal(got, want)
    assert not any("two_token_steps" in key for key in set(msa_token_bwd._DKDV_CACHE) - before)
    # eligibility is the kernel family and the block size, whatever S and B are
    ok = msa_token_bwd.dkdv_two_token_steps_eligible
    assert ok(128, True) and not ok(128, False) and not ok(64, True) and not ok(256, True)


def test_two_token_steps_self_check_passes(require_bitwise):
    """The real self-check, on the training shape and path (dS records included) and on the
    reference dQ path with B > 1."""
    from primus.backends.megatron.core.fusions import bitwise_check

    q, k, v, dout, table = H._inputs(4096, 1, 4, 16, "local")
    H._backward(q, k, v, dout, table, E._walk(table, True), dkdv_two_token_steps=True, **DS)
    q, k, v, dout, table = H._inputs(1152, 2, 2, 16, "hot")
    H._backward(q, k, v, dout, table, None, dkdv_two_token_steps=True)
    decided = {
        key: ok
        for (name, key), ok in bitwise_check._results.items()
        if name == "msa_dkdv_two_token_steps dK/dV"
    }
    assert any(key[0] == 4096 and key[-2] for key in decided), "no check on the dS path"
    assert any(key[0] == 1152 and not key[-2] for key in decided), "no check on the reference dQ path"
    require_bitwise(all(decided.values()), "msa_dkdv_two_token_steps")


def _fake_attention(**config):
    defaults = dict(msa_dkdv_two_token_steps=True, msa_exact_numerics=False)
    defaults.update(config)
    return SimpleNamespace(config=SimpleNamespace(**defaults), _gfx1250=True, block_size=BLOCK)


def test_switch_defaults_off_and_exact_numerics_turns_it_off():
    from primus.backends.megatron.core.models.minimax_m3.minimax_m3_transformer_config import (
        MSATransformerConfig,
    )
    from primus.backends.megatron.core.transformer.minimax_m3 import attention
    from primus.backends.megatron.core.transformer.minimax_m3.attention import MinimaxSparseAttention

    assert MSATransformerConfig.__dataclass_fields__["msa_dkdv_two_token_steps"].default is False
    assert "msa_dkdv_two_token_steps" in attention._MSA_RELAXED
    A = MinimaxSparseAttention
    assert A._dkdv_two_token_steps(_fake_attention()) == {"dkdv_two_token_steps": True}
    # msa_exact_numerics reproduces the reference kernels: off, whatever the switch says
    assert A._dkdv_two_token_steps(_fake_attention(msa_exact_numerics=True)) == {}
    assert A._dkdv_two_token_steps(_fake_attention(msa_dkdv_two_token_steps=False)) == {}
    other_gpu = _fake_attention()
    other_gpu._gfx1250 = False
    assert A._dkdv_two_token_steps(other_gpu) == {}
