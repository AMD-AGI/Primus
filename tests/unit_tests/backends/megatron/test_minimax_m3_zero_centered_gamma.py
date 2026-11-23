###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``msa_fused_zero_centered_gamma``: the q / k and indexer norm + RoPE kernels form a
zero-centered norm's BF16 ``1 + weight`` themselves. Same bits as upstream's split, Turbo
RMSNorms and RoPE, forward and backward, at MiniMax-M3.1-Flash's shapes, at MBS 2 and at a
longer sequence; and the cases that keep the separate ``weight + 1``."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transformer_engine.pytorch")
pytest.importorskip("primus_turbo")

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

D, ROT = 128, 64


@pytest.fixture
def fresh_checks(monkeypatch):
    from primus.backends.megatron.core.fusions import bitwise_check
    from primus.backends.megatron.core.transformer.minimax_m3 import qk_norm_rope

    if qk_norm_rope._turbo_cfg() is None:
        pytest.skip("this Primus-Turbo lacks the RMSNorm launch helpers the kernels follow")
    monkeypatch.setattr(bitwise_check, "_results", {})
    return bitwise_check


def _weights(kind):
    """Zero-centered weights whose BF16 ``1 + w`` rounds: tiny, mixed-sign, -1 (gamma 0), large."""
    gen = torch.Generator().manual_seed(7)
    w = torch.randn(D, generator=gen) * 0.1
    if kind == "edges":
        w[:8] = torch.tensor([0.0, -0.0, 1e-4, -1e-4, -1.0, 3.0e-3, 255.5, -2.0])
        w[8:16] = torch.tensor([2.0**-8, 2.0**-9, -(2.0**-9), 1.5 * 2.0**-8, 0.00390625, 1e-8, -1e-8, 0.5])
    return w


def _norm(zero_centered=True, kind="randn", dtype=torch.bfloat16):
    from primus.backends.megatron.core.extensions.primus_turbo import PrimusTurboRMSNorm

    norm = PrimusTurboRMSNorm(D, eps=1e-6, zero_centered_gamma=zero_centered, params_dtype=dtype).cuda()
    with torch.no_grad():
        norm.weight.copy_(_weights(kind).cuda() + (0.0 if zero_centered else 1.0))
    return norm


def _rope(seq):
    from primus.backends.megatron.core.transformer.minimax_m3.rope import RopeTables

    inv_freq = 1.0 / (1e7 ** (torch.arange(0, ROT, 2, device="cuda", dtype=torch.float32) / ROT))
    freqs = torch.outer(torch.arange(seq, device="cuda", dtype=torch.float32), inv_freq)
    return RopeTables(torch.cat((freqs, freqs), dim=-1).view(seq, 1, 1, ROT))


def _attn(q_norm, k_norm, switch, ng=4, hpg=16):
    return SimpleNamespace(
        q_layernorm=q_norm,
        k_layernorm=k_norm,
        num_attention_heads_per_partition=ng * hpg,
        num_query_groups_per_partition=ng,
        hidden_size_per_attention_head=D,
        config=SimpleNamespace(msa_fused_zero_centered_gamma=switch),
    )


def _run(fn, x, dys, params):
    x = x.detach().clone().requires_grad_()
    for p in params:
        p.grad = None
    outs = fn(x)
    torch.autograd.backward(outs, dys)
    return [o.detach() for o in outs] + [x.grad] + [p.grad.clone() for p in params]


def _assert_bitwise(got, ref):
    assert len(got) == len(ref)
    for i, (g, r) in enumerate(zip(got, ref)):
        assert g.shape == r.shape and g.dtype == r.dtype, i
        assert torch.equal(g.view(torch.int16), r.view(torch.int16)), i


def test_switch_defaults_to_off():
    from primus.backends.megatron.core.models.minimax_m3.minimax_m3_transformer_config import (
        MSATransformerConfig,
    )

    assert MSATransformerConfig.__dataclass_fields__["msa_fused_zero_centered_gamma"].default is False


@needs_gpu
def test_gamma_argument():
    """In-kernel only for a zero-centered BF16 weight with the switch on; the eager ``weight + 1``
    (or the raw weight) otherwise."""
    from primus.backends.megatron.core.transformer.minimax_m3 import qk_norm_rope

    zc, plain, fp32 = _norm(True), _norm(False), _norm(True, dtype=torch.float32)
    g, flag = qk_norm_rope._gamma(zc, zc.weight, True)
    assert g is zc.weight and flag is True
    g, flag = qk_norm_rope._gamma(zc, zc.weight, False)
    assert flag is False and torch.equal(g, zc.weight + 1)
    g, flag = qk_norm_rope._gamma(plain, plain.weight, True)
    assert g is plain.weight and flag is False
    g, flag = qk_norm_rope._gamma(fp32, fp32.weight, True)
    assert flag is False and torch.equal(g, fp32.weight + 1)


@needs_gpu
@pytest.mark.parametrize("seq, batch", [(4096, 1), (4096, 2), (8192, 1), (1000, 3)])
@pytest.mark.parametrize("kind", ["randn", "edges"])
def test_qk_kernels_with_in_kernel_gamma_match_upstream(seq, batch, kind, fresh_checks, require_bitwise):
    """``_QKNormRope`` itself (not its dispatcher) with the raw weights and ``zc=True``."""
    from primus.backends.megatron.core.transformer.minimax_m3 import qk_norm_rope

    torch.manual_seed(0)
    ng, hpg = 4, 16
    attn = _attn(_norm(True, kind), _norm(True, kind), True, ng, hpg)
    rope = _rope(seq)
    params = [attn.q_layernorm.weight, attn.k_layernorm.weight]
    if not (qk_norm_rope._multi_row(D, seq * batch * ng * hpg) and qk_norm_rope._multi_row(D, seq * batch * ng)):
        pytest.skip("Turbo launches one of these row counts single-row; the fused path does not apply")
    ok = qk_norm_rope._qk_self_check(attn, seq, batch, rope, True)
    assert all(p.grad is None for p in params)
    assert [k for (n, k) in fresh_checks._results if n == "msa_fused_qk_norm_rope"] == [
        (seq, batch, ng, hpg, "in-kernel gamma")
    ]
    require_bitwise(ok, "msa_fused_qk_norm_rope with msa_fused_zero_centered_gamma")

    mixed = torch.randn(seq, batch, ng, (hpg + 2) * D, device="cuda").to(torch.bfloat16)
    dys = [
        torch.randn(seq, batch, ng * hpg, D, device="cuda").to(torch.bfloat16),
        torch.randn(seq, batch, ng, D, device="cuda").to(torch.bfloat16),
        torch.randn(seq, batch, ng, D, device="cuda").to(torch.bfloat16),
    ]

    def upstream(m):
        q, k, v = torch.split(m, [hpg * D, D, D], dim=3)
        q = q.reshape(seq, batch, -1, D)
        return rope.apply(attn.q_layernorm(q)), rope.apply(attn.k_layernorm(k)), v

    def kernels(m):
        cos_, sin_ = rope.cos_sin(m.dtype)
        return qk_norm_rope._QKNormRope.apply(m, params[0], params[1], cos_, sin_, hpg, attn.q_layernorm.eps, True)

    ref = _run(upstream, mixed, dys, params)
    _assert_bitwise(_run(kernels, mixed, dys, params), ref)
    # the dispatcher takes the same path with the switch on
    _assert_bitwise(_run(lambda m: qk_norm_rope.qk_norm_rope(attn, m, rope), mixed, dys, params), ref)


@needs_gpu
@pytest.mark.parametrize("seq, heads, batch", [(4096, 4, 1), (4096, 1, 1), (4096, 4, 2), (8192, 4, 1)])
@pytest.mark.parametrize("kind", ["randn", "edges"])
def test_norm_rope_kernels_with_in_kernel_gamma_match_upstream(seq, heads, batch, kind, fresh_checks, require_bitwise):
    from primus.backends.megatron.core.transformer.minimax_m3 import qk_norm_rope

    torch.manual_seed(0)
    norm, rope = _norm(True, kind), _rope(seq)
    x = torch.randn(seq, batch, heads, D, device="cuda").to(torch.bfloat16)
    dy = torch.randn_like(x)
    if not qk_norm_rope._multi_row(D, x.numel() // D):
        pytest.skip("Turbo launches this row count single-row; the fused path does not apply")
    ok = qk_norm_rope.norm_rope_ok(x, norm, rope, False, True)
    assert [k for (n, k) in fresh_checks._results if n == "msa_fused_norm_rope"] == [
        (seq, batch, heads, D, "in-kernel gamma")
    ]
    require_bitwise(ok, "msa_fused_norm_rope with msa_fused_zero_centered_gamma")

    def kernels(t):
        cos_, sin_ = rope.cos_sin(t.dtype)
        return (qk_norm_rope._NormRope.apply(t, norm.weight, cos_, sin_, norm.eps, True),)

    ref = _run(lambda t: (rope.apply(norm(t)),), x, [dy], [norm.weight])
    _assert_bitwise(_run(kernels, x, [dy], [norm.weight]), ref)
    _assert_bitwise(_run(lambda t: (qk_norm_rope.norm_rope(t, norm, rope, True),), x, [dy], [norm.weight]), ref)


@needs_gpu
def test_switch_off_keeps_the_separate_add_and_the_old_check_key(fresh_checks, require_bitwise):
    from primus.backends.megatron.core.transformer.minimax_m3 import qk_norm_rope

    norm, rope = _norm(True), _rope(4096)
    x = torch.randn(4096, 1, 4, D, device="cuda").to(torch.bfloat16)
    if not qk_norm_rope._multi_row(D, x.numel() // D):
        pytest.skip("Turbo launches this row count single-row; the fused path does not apply")
    require_bitwise(qk_norm_rope.norm_rope_ok(x, norm, rope, False), "msa_fused_norm_rope")
    assert list(fresh_checks._results) == [("msa_fused_norm_rope", (4096, 1, 4, D))]


@needs_gpu
def test_not_zero_centered_or_fp32_weight_is_unaffected_by_the_switch(fresh_checks, require_bitwise):
    from primus.backends.megatron.core.transformer.minimax_m3 import qk_norm_rope

    rope = _rope(4096)
    x = torch.randn(4096, 1, 4, D, device="cuda").to(torch.bfloat16)
    if not qk_norm_rope._multi_row(D, x.numel() // D):
        pytest.skip("Turbo launches this row count single-row; the fused path does not apply")
    norm = _norm(False)
    require_bitwise(qk_norm_rope.norm_rope_ok(x, norm, rope, False, True), "msa_fused_norm_rope")
    assert list(fresh_checks._results) == [("msa_fused_norm_rope", (4096, 1, 4, D))]
    dy = torch.randn_like(x)
    ref = _run(lambda t: (rope.apply(norm(t)),), x, [dy], [norm.weight])
    _assert_bitwise(_run(lambda t: (qk_norm_rope.norm_rope(t, norm, rope, True),), x, [dy], [norm.weight]), ref)


@needs_gpu
def test_in_kernel_self_check_fails_on_one_ulp(monkeypatch, fresh_checks):
    """A kernel whose in-register gamma is one BF16 ulp off must lose its self-check."""
    from primus.backends.megatron.core.transformer.minimax_m3 import qk_norm_rope

    norm, rope = _norm(True), _rope(4096)
    x = torch.randn(4096, 1, 4, D, device="cuda").to(torch.bfloat16)
    if not qk_norm_rope._multi_row(D, x.numel() // D):
        pytest.skip("Turbo launches this row count single-row; the fused path does not apply")
    real = qk_norm_rope._norm_rope_fwd

    def off_by_one(*args, **kwargs):
        out, rstd = real(*args, **kwargs)
        out.view(torch.int16)[0, 0] ^= 1
        return out, rstd

    monkeypatch.setattr(qk_norm_rope, "_norm_rope_fwd", off_by_one)
    assert not qk_norm_rope.norm_rope_ok(x, norm, rope, False, True)
    assert fresh_checks._results == {("msa_fused_norm_rope", (4096, 1, 4, D, "in-kernel gamma")): False}
