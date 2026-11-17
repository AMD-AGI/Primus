###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""qk_norm_rope must reproduce upstream's qkv split, Turbo q/k RMSNorms and RoPE bit for bit,
forward and backward, at MiniMax-M3.1-Flash's shapes (64 heads, 4 query groups, head dim 128,
rotary dim 64, 4 index heads)."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transformer_engine.pytorch")
pytest.importorskip("primus_turbo")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

D, ROT, SEQ = 128, 64, 4096


@pytest.fixture(autouse=True)
def fresh_checks(monkeypatch):
    from primus.backends.megatron.core.fusions import bitwise_check
    from primus.backends.megatron.core.transformer.minimax_m3 import qk_norm_rope

    if qk_norm_rope._turbo_cfg() is None:
        pytest.skip("this Primus-Turbo lacks the RMSNorm launch helpers the kernels follow")
    monkeypatch.setattr(bitwise_check, "_results", {})


def _norm(zero_centered):
    from primus.backends.megatron.core.extensions.primus_turbo import PrimusTurboRMSNorm

    norm = PrimusTurboRMSNorm(D, eps=1e-6, zero_centered_gamma=zero_centered, params_dtype=torch.bfloat16)
    norm = norm.cuda()
    with torch.no_grad():
        norm.weight.copy_(torch.randn(D, device="cuda") * 0.1 + (0.0 if zero_centered else 1.0))
    return norm


def _rope():
    from primus.backends.megatron.core.transformer.minimax_m3.rope import RopeTables

    inv_freq = 1.0 / (1e7 ** (torch.arange(0, ROT, 2, device="cuda", dtype=torch.float32) / ROT))
    freqs = torch.outer(torch.arange(SEQ, device="cuda", dtype=torch.float32), inv_freq)
    return RopeTables(torch.cat((freqs, freqs), dim=-1).view(SEQ, 1, 1, ROT))


def _attn(q_norm, k_norm, ng=4, hpg=16):
    return SimpleNamespace(
        q_layernorm=q_norm,
        k_layernorm=k_norm,
        num_attention_heads_per_partition=ng * hpg,
        num_query_groups_per_partition=ng,
        hidden_size_per_attention_head=D,
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


def _decisions(bitwise_check):
    return [(name, ok) for (name, _), ok in bitwise_check._results.items()]


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("zero_centered", [True, False])
def test_qk_norm_rope_matches_upstream_bitwise(zero_centered, batch, require_bitwise):
    from primus.backends.megatron.core.transformer.minimax_m3 import qk_norm_rope

    torch.manual_seed(0)
    ng, hpg = 4, 16
    attn = _attn(_norm(zero_centered), _norm(zero_centered), ng, hpg)
    rope = _rope()
    params = [attn.q_layernorm.weight, attn.k_layernorm.weight]
    for p in params:
        p.grad = None
    used = (
        qk_norm_rope._multi_row(D, SEQ * batch * ng * hpg)
        and qk_norm_rope._multi_row(D, SEQ * batch * ng)
        and qk_norm_rope._qk_self_check(attn, SEQ, batch, rope)
    )
    assert all(p.grad is None for p in params)
    require_bitwise(used, "msa_fused_qk_norm_rope")

    mixed = torch.randn(SEQ, batch, ng, (hpg + 2) * D, device="cuda").to(torch.bfloat16)
    dys = [
        torch.randn(SEQ, batch, ng * hpg, D, device="cuda").to(torch.bfloat16),
        torch.randn(SEQ, batch, ng, D, device="cuda").to(torch.bfloat16),
        torch.randn(SEQ, batch, ng, D, device="cuda").to(torch.bfloat16),
    ]

    def upstream(m):
        q, k, v = torch.split(m, [hpg * D, D, D], dim=3)
        q = q.reshape(SEQ, batch, -1, D)
        return rope.apply(attn.q_layernorm(q)), rope.apply(attn.k_layernorm(k)), v

    ref = _run(upstream, mixed, dys, params)
    got = _run(lambda m: qk_norm_rope.qk_norm_rope(attn, m, rope), mixed, dys, params)
    _assert_bitwise(got, ref)


@pytest.mark.parametrize("heads, batch", [(4, 1), (1, 1), (4, 2)])
@pytest.mark.parametrize("zero_centered", [True, False])
def test_norm_rope_matches_upstream_bitwise(heads, batch, zero_centered, require_bitwise):
    from primus.backends.megatron.core.transformer.minimax_m3 import qk_norm_rope

    torch.manual_seed(0)
    norm, rope = _norm(zero_centered), _rope()
    x = torch.randn(SEQ, batch, heads, D, device="cuda").to(torch.bfloat16)
    dy = torch.randn_like(x)
    require_bitwise(qk_norm_rope.norm_rope_ok(x, norm, rope, False), "msa_fused_norm_rope")

    ref = _run(lambda t: (rope.apply(norm(t)),), x, [dy], [norm.weight])
    got = _run(lambda t: (qk_norm_rope.norm_rope(t, norm, rope),), x, [dy], [norm.weight])
    _assert_bitwise(got, ref)


def test_each_row_count_is_checked():
    """index_q and index_k have different row counts, so Turbo may launch them differently."""
    from primus.backends.megatron.core.fusions import bitwise_check
    from primus.backends.megatron.core.transformer.minimax_m3 import qk_norm_rope

    if not (qk_norm_rope._multi_row(D, SEQ * 4) and qk_norm_rope._multi_row(D, SEQ)):
        pytest.skip("Turbo launches one of these row counts single-row; the fused path does not apply")
    norm, rope = _norm(True), _rope()
    for heads in (4, 1, 4):
        qk_norm_rope.norm_rope_ok(
            torch.randn(SEQ, 1, heads, D, device="cuda").to(torch.bfloat16), norm, rope, False
        )
    assert sorted(key for name, key in bitwise_check._results) == [(SEQ, 1, 1, D), (SEQ, 1, 4, D)]


def test_self_check_moves_gamma_off_its_init_value(monkeypatch):
    """At init a zero-centered gamma is exactly 1, which hides how the kernels apply it."""
    from primus.backends.megatron.core.fusions import bitwise_check
    from primus.backends.megatron.core.transformer.minimax_m3 import qk_norm_rope

    if not (qk_norm_rope._multi_row(D, SEQ) and qk_norm_rope._multi_row(D, SEQ * 64)):
        pytest.skip("Turbo launches one of these row counts single-row; the fused path does not apply")
    weights = []
    real = bitwise_check.outputs_and_grads

    def spy(fn, *inputs):
        weights.append(inputs[-1])
        return real(fn, *inputs)

    monkeypatch.setattr(bitwise_check, "outputs_and_grads", spy)
    norm, rope = _norm(True), _rope()
    with torch.no_grad():
        norm.weight.zero_()
    qk_norm_rope.norm_rope_ok(torch.randn(SEQ, 1, 1, D, device="cuda").to(torch.bfloat16), norm, rope, False)
    qk_norm_rope._qk_self_check(_attn(norm, norm), SEQ, 1, rope)
    assert len(weights) == 4
    assert all(w.shape == norm.weight.shape and w.dtype == norm.weight.dtype for w in weights)
    assert all(w.count_nonzero() == w.numel() for w in weights)


def test_norm_with_hooks_is_not_eligible(require_bitwise):
    from primus.backends.megatron.core.transformer.minimax_m3 import qk_norm_rope

    norm, rope = _norm(True), _rope()
    x = torch.randn(SEQ, 1, 1, D, device="cuda").to(torch.bfloat16)
    require_bitwise(qk_norm_rope.norm_rope_ok(x, norm, rope, False), "msa_fused_norm_rope")
    handle = norm.register_forward_pre_hook(lambda module, args: None)
    assert not qk_norm_rope.norm_rope_ok(x, norm, rope, False)
    handle.remove()


def test_not_eligible_when_the_self_check_fails(monkeypatch):
    from primus.backends.megatron.core.transformer.minimax_m3 import qk_norm_rope

    real = qk_norm_rope._NormRope

    def off_by_one_ulp(*args):
        return (real.apply(*args).view(torch.int16) ^ 1).view(torch.bfloat16)

    monkeypatch.setattr(qk_norm_rope, "_NormRope", SimpleNamespace(apply=off_by_one_ulp))
    norm, rope = _norm(True), _rope()
    x = torch.randn(SEQ, 1, 1, D, device="cuda").to(torch.bfloat16)
    assert not qk_norm_rope.norm_rope_ok(x, norm, rope, False)


def test_qk_self_check_fails_on_one_ulp(monkeypatch):
    from primus.backends.megatron.core.fusions import bitwise_check
    from primus.backends.megatron.core.transformer.minimax_m3 import qk_norm_rope

    real = qk_norm_rope._QKNormRope

    def off_by_one_ulp(*args):
        q, k, v = real.apply(*args)
        return (q.view(torch.int16) ^ 1).view(torch.bfloat16), k, v

    monkeypatch.setattr(qk_norm_rope, "_QKNormRope", SimpleNamespace(apply=off_by_one_ulp))
    norm = _norm(True)
    assert not qk_norm_rope._qk_self_check(_attn(norm, norm), SEQ, 1, _rope())
    assert _decisions(bitwise_check) == [("msa_fused_qk_norm_rope", False)]


def test_decided_checks_build_no_inputs(monkeypatch, require_bitwise):
    from primus.backends.megatron.core.fusions import bitwise_check
    from primus.backends.megatron.core.transformer.minimax_m3 import qk_norm_rope

    norm, rope = _norm(True), _rope()
    x = torch.randn(SEQ, 1, 4, D, device="cuda").to(torch.bfloat16)
    attn = _attn(norm, norm)
    require_bitwise(qk_norm_rope.norm_rope_ok(x, norm, rope, False), "msa_fused_norm_rope")
    require_bitwise(qk_norm_rope._qk_self_check(attn, SEQ, 1, rope), "msa_fused_qk_norm_rope")

    def no_inputs(*args, **kwargs):
        raise AssertionError("a decided check built its inputs again")

    monkeypatch.setattr(bitwise_check, "randn", no_inputs)
    assert qk_norm_rope.norm_rope_ok(x, norm, rope, False)
    assert qk_norm_rope._qk_self_check(attn, SEQ, 1, rope)


def test_not_eligible_without_turbo_config_helpers(monkeypatch, require_bitwise):
    from primus.backends.megatron.core.transformer.minimax_m3 import qk_norm_rope

    norm, rope = _norm(True), _rope()
    x = torch.randn(SEQ, 1, 1, D, device="cuda").to(torch.bfloat16)
    require_bitwise(qk_norm_rope.norm_rope_ok(x, norm, rope, False), "msa_fused_norm_rope")
    monkeypatch.setattr(qk_norm_rope, "_turbo_cfg", lambda: None)
    assert not qk_norm_rope.norm_rope_ok(x, norm, rope, False)
    changed = SimpleNamespace(
        _pick_config=lambda d, n: (d, 4), _pick_bwd_config=lambda d, n: ("multi", d), _finalize_dgamma=None
    )
    monkeypatch.setattr(qk_norm_rope, "_turbo_cfg", lambda: changed)
    assert not qk_norm_rope.norm_rope_ok(x, norm, rope, False)
