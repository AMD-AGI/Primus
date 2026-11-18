###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CPU equivalence of the zaya_fuse_moe path's compiled functions (SwiGLU, router,
gate/unpermute) with the same functions uncompiled, forward and backward, fp32 and bf16."""

import pytest
import torch

import primus.backends.megatron.core.models.zaya1.zaya1_modules as zm
from primus.backends.megatron.core.utils import maybe_compile


def _cfg(**overrides):
    class Config:
        pass

    cfg = Config()
    cfg.hidden_size = 32
    cfg.num_layers = 4
    cfg.num_attention_heads = 4
    cfg.num_query_groups = 2
    cfg.kv_channels = 8
    cfg.ffn_hidden_size = 16
    cfg.moe_ffn_hidden_size = 16
    cfg.num_moe_experts = 4
    cfg.moe_router_topk = 1
    cfg.layernorm_epsilon = 1e-5
    cfg.zaya_mlp_expansion = 8
    cfg.zaya_use_mod = True
    cfg.zaya_use_eda = True
    cfg.zaya_high_prec = True
    cfg.scale_residual_merge = True
    cfg.clamp_temp = False
    cfg.cca_time0 = 2
    cfg.cca_time1 = 2
    cfg.partial_rotary_factor = 0.5
    cfg.rotary_base = 10000.0
    cfg.zaya_balance_lr = 1e-3
    cfg.zaya_balance_beta1 = 0.9
    cfg.zaya_balance_beta2 = 0.999
    cfg.zaya_balance_wd = 0.0
    cfg.zaya_layers = None
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


_FUSED = (
    "_fused_swiglu",
    "_fused_eda_norm",
    "_fused_route_select",
    "_fused_unpermute_fwd",
    "_fused_unpermute_bwd",
    "_fused_unpermute_bwd_noprob",
)


@pytest.fixture
def compiled(monkeypatch):
    """Fresh compiled helpers for this test, failing it if any of them silently fell back to eager."""
    monkeypatch.setattr(zm, "_fused_swiglu", maybe_compile(zm._swiglu, dynamic=True))
    monkeypatch.setattr(zm, "_fused_eda_norm", maybe_compile(zm._eda_norm))
    monkeypatch.setattr(zm, "_fused_route_select", maybe_compile(zm._route_select))
    monkeypatch.setattr(zm, "_fused_unpermute_fwd", maybe_compile(zm._unpermute_scale_fwd))
    monkeypatch.setattr(zm, "_fused_unpermute_bwd", maybe_compile(zm._unpermute_scale_bwd))
    monkeypatch.setattr(zm, "_fused_unpermute_bwd_noprob", maybe_compile(zm._unpermute_scale_bwd_noprob))
    yield monkeypatch
    for name in _FUSED:
        fn = getattr(zm, name)
        if hasattr(fn, "compiled_ok"):
            assert fn.compiled_ok(), f"{name} fell back to eager"


def _eager_twin(module, build, **overrides):
    """``build(cfg)`` with zaya_fuse_moe off, holding ``module``'s weights and buffers."""
    twin = build(_cfg(zaya_fuse_moe=False, **overrides)).to(next(module.parameters()).dtype)
    twin.load_state_dict(module.state_dict())
    return twin


def _run(module, *inputs):
    inputs = [x.detach().clone().requires_grad_(x.is_floating_point()) for x in inputs]
    out = module(*inputs)
    outs = out if isinstance(out, tuple) else (out,)
    loss = sum(
        (o.float() * torch.linspace(0.5, 1.5, o.numel()).view_as(o)).sum()
        for o in outs
        if o.is_floating_point()
    )
    loss.backward()
    params = module.named_parameters() if hasattr(module, "named_parameters") else []
    grads = {n: p.grad.detach().clone() for n, p in params if p.grad is not None}
    return outs, [x.grad for x in inputs], grads


def _tol(dtype):
    return dict(atol=1e-5, rtol=1e-5) if dtype == torch.float32 else dict(atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_swiglu_expert_matches_eager(compiled, dtype):
    torch.manual_seed(0)
    cfg = _cfg(zaya_fuse_moe=True)

    def build(c):
        return zm.SwiGLUExpert(c.hidden_size, c.moe_ffn_hidden_size, c)

    expert = build(cfg).to(dtype)
    ref = _eager_twin(expert, build)
    for n in (1, 5, 13):  # varying token counts share one dynamic graph
        x = torch.randn(n, cfg.hidden_size, dtype=dtype)
        o_e, g_e, p_e = _run(ref, x)
        o_f, g_f, p_f = _run(expert, x)
        torch.testing.assert_close(o_f, o_e, **_tol(dtype))
        torch.testing.assert_close(g_f, g_e, **_tol(dtype))
        for k in p_e:
            torch.testing.assert_close(p_f[k], p_e[k], **_tol(dtype))
        expert.zero_grad(set_to_none=True)
        ref.zero_grad(set_to_none=True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("with_prev", [False, True])
@pytest.mark.parametrize("sort_dispatch", [True, False])
def test_moe_layer_matches_eager(compiled, dtype, with_prev, sort_dispatch):
    torch.manual_seed(1)
    cfg = _cfg(zaya_fuse_moe=True, zaya_sort_dispatch=sort_dispatch)
    moe = zm.ZayaMoE(cfg).to(dtype)
    with torch.no_grad():
        moe.router.router_states_scale.uniform_(0.2, 1.8)
        for m in moe.router.router_mlp:
            if hasattr(m, "weight"):
                m.weight.mul_(20.0)  # spread the routing over all experts and MoD
    ref = _eager_twin(moe, zm.ZayaMoE, zaya_sort_dispatch=sort_dispatch)
    seq, batch = 7, 3
    hidden = torch.randn(seq, batch, cfg.hidden_size, dtype=dtype)
    prev = torch.randn(seq, batch, cfg.zaya_mlp_expansion, dtype=dtype) if with_prev else None

    def call(mod):
        return lambda h, *p: mod(h, p[0] if p else None)

    args = (hidden, prev) if with_prev else (hidden,)
    o_e, g_e, p_e = _run(call(ref), *args)
    o_f, g_f, p_f = _run(call(moe), *args)
    # PID buffers moved identically (same counts).
    torch.testing.assert_close(moe.router.balancing_biases, ref.router.balancing_biases)
    # Same routing decisions (eval: no further bias update), spread over experts.
    moe.eval()
    ref.eval()
    with torch.no_grad():
        _, idx_e, _ = ref.router(hidden, prev)
        _, idx_f, _ = moe.router(hidden, prev)
    assert torch.equal(idx_e, idx_f)
    assert len(torch.unique(idx_e)) >= 3
    tol = _tol(dtype)
    for a, b in zip(o_f, o_e):
        torch.testing.assert_close(a, b, **tol)
    for a, b in zip(g_f, g_e):
        torch.testing.assert_close(a, b, **tol)
    p_f = dict(moe.named_parameters())
    p_e = dict(ref.named_parameters())
    for k, pe in p_e.items():
        if pe.grad is None:
            assert p_f[k].grad is None, k
            continue
        torch.testing.assert_close(p_f[k].grad, pe.grad, **tol)


@pytest.mark.parametrize("fused", [False, True])
def test_unpermute_scale_without_prob_grad(compiled, fused):
    torch.manual_seed(2)
    n, h = 9, 4
    perm = torch.randperm(n)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(n)
    x = torch.randn(n, h, requires_grad=True)
    p = torch.rand(n, 1)
    out = zm._UnpermuteScale.apply(x, perm, inv, p, fused, [n])
    out.sum().backward()
    torch.testing.assert_close(out, x.detach()[inv] * p)
    torch.testing.assert_close(x.grad, p[perm].expand(n, h))
