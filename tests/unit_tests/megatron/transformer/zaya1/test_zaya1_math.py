###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CPU checks that the ZAYA1 mixer uses the implemented affine, not the paper affine."""

import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from primus.backends.megatron.core.models.zaya1.zaya1_modules import (
    CCA,
    ResidualScaling,
    ZayaMoE,
    ZayaRouter,
    ZayaStack,
    layer_kinds,
)


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


def test_residual_is_implemented_affine_not_paper_affine():
    hidden = 16
    mod = ResidualScaling(hidden, has_residual=True)
    with torch.no_grad():
        mod.hidden_states_scale.uniform_(0.5, 1.5)
        mod.hidden_states_bias.uniform_(-0.3, 0.3)
        mod.residual_scale.uniform_(0.5, 1.5)
        mod.residual_bias.uniform_(-0.3, 0.3)
    residual = torch.randn(3, 2, hidden)
    hidden_states = torch.randn(3, 2, hidden)

    got_residual, got_hidden = mod(residual, hidden_states)
    ref_hidden = (hidden_states.float() + mod.hidden_states_bias.float()) * mod.hidden_states_scale.float()
    ref_residual = (residual.float() + mod.residual_bias.float()) * mod.residual_scale.float()
    paper_hidden = mod.hidden_states_scale.float() * hidden_states.float() + mod.hidden_states_bias.float()

    torch.testing.assert_close(got_hidden, ref_hidden)
    torch.testing.assert_close(got_residual, ref_residual)
    assert not torch.allclose(got_hidden, paper_hidden)


def test_layer0_has_no_residual_affine():
    mod = ResidualScaling(8, has_residual=False)
    assert not hasattr(mod, "residual_scale")
    hidden = torch.randn(2, 1, 8)
    residual, out = mod(None, hidden)
    assert residual is None
    torch.testing.assert_close(out, (hidden.float() + mod.hidden_states_bias) * mod.hidden_states_scale)


def test_cca_shapes_value_shift_and_raw_temperature():
    cfg = _cfg()
    cca = CCA(cfg)
    seq, batch = 5, 2
    hidden = torch.randn(seq, batch, cfg.hidden_size)
    query, key, value = cca.project(hidden, position_ids=None)

    assert query.shape == (seq, batch, cca.num_q_heads, cca.head_dim)
    assert key.shape == (seq, batch, cca.num_k_heads, cca.head_dim)
    assert value.shape == (seq, batch, cca.num_k_heads, cca.head_dim)

    # t = 0 is a left zero pad, and val_proj2 bias starts at 0, so the shifted
    # half of the KV heads is exactly zero. The current half is not.
    shifted_heads = value[0, :, cca.num_k_heads // 2 :]
    assert torch.count_nonzero(shifted_heads) == 0
    assert torch.count_nonzero(value[0, :, : cca.num_k_heads // 2]) > 0

    raw_q = torch.randn(seq, batch, cca.num_q_heads, cca.head_dim)
    raw_k = torch.randn(seq, batch, cca.num_k_heads, cca.head_dim)
    with torch.no_grad():
        cca.temp.fill_(0.5)
    got_q, got_k = cca._qk_norm_and_temp(raw_q, raw_k)

    eps = 1e-12
    scale = math.sqrt(cca.head_dim)

    def _norm(x):
        return x.float() * (torch.rsqrt(x.float().pow(2).sum(-1, keepdim=True) + eps) * scale)

    torch.testing.assert_close(got_q, _norm(raw_q))
    torch.testing.assert_close(got_k, _norm(raw_k) * 0.5)
    assert not torch.allclose(got_k, _norm(raw_k) * math.exp(0.5))


def test_grouped_qk_mean_matches_independent_formula():
    cfg = _cfg()
    cca = CCA(cfg)
    seq, batch = 3, 2
    nq, nk, groups, dim = cca.num_q_heads, cca.num_k_heads, cca.gqa_groups, cca.head_dim
    query_conv = torch.randn(seq, batch, nq, dim)
    key_conv = torch.randn(seq, batch, nk, dim)
    query_pre = torch.randn(seq, batch, nq, dim)
    key_base = torch.randn(seq, batch, nk, dim)
    query, key = cca._grouped_means(query_conv, key_conv, query_pre, key_base)

    query_pre_g = query_pre.view(seq, batch, nk, groups, dim).float()
    query_conv_g = query_conv.view(seq, batch, nk, groups, dim).float()
    key_term = 0.5 * key_base.float().unsqueeze(-2)
    ref_q = query_conv_g + 0.5 * query_pre_g + key_term
    ref_q = ref_q.reshape(seq, batch, nq, dim)
    ref_k = key_conv.float() + 0.5 * query_pre_g.mean(dim=-2) + 0.5 * key_base.float()
    torch.testing.assert_close(query, ref_q)
    torch.testing.assert_close(key, ref_k)


def _zero_router(router: ZayaRouter) -> None:
    for module in router.modules():
        if isinstance(module, nn.Linear):
            nn.init.zeros_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def test_router_eda_is_a_vector_scale_before_norm():
    cfg = _cfg()
    router = ZayaRouter(cfg)
    seq, batch = 3, 2
    hidden = torch.randn(seq, batch, cfg.hidden_size)
    prev = torch.randn(seq, batch, cfg.zaya_mlp_expansion)
    with torch.no_grad():
        router.router_states_scale.uniform_(0.2, 1.8)
    _, _, nxt = router(hidden, prev)

    flat = hidden.reshape(seq * batch, cfg.hidden_size)
    ref = F.linear(flat, router.down_proj.weight, router.down_proj.bias)
    ref = ref + prev.reshape(seq * batch, cfg.zaya_mlp_expansion) * router.router_states_scale
    torch.testing.assert_close(nxt, ref.view(seq, batch, cfg.zaya_mlp_expansion))


def test_mod_skip_scales_the_hidden_state():
    cfg = _cfg()
    moe = ZayaMoE(cfg)
    _zero_router(moe.router)
    with torch.no_grad():
        moe.router.balancing_biases.zero_()
        moe.router.balancing_biases[-1] = 10.0
    hidden = torch.randn(4, 2, cfg.hidden_size)
    out, _ = moe(hidden, prev_router=None)
    n_scored = cfg.num_moe_experts + 1
    torch.testing.assert_close(out, hidden / n_scored)


def test_alternating_stages_are_forty_and_forty():
    kinds = layer_kinds(80)
    assert kinds.count("a") == 40
    assert kinds.count("m") == 40
    assert kinds[0] == "a" and kinds[1] == "m"
    cfg = _cfg(num_layers=80)
    stack = ZayaStack(cfg)
    assert sum(layer.kind == "a" for layer in stack.layers) == 40
    assert sum(layer.kind == "m" for layer in stack.layers) == 40
    assert stack.layers[0].res_scale.has_residual is False
    assert stack.layers[1].res_scale.has_residual is True
    assert stack.res_scale.has_residual is True


def test_pid_updates_on_train_and_freezes_on_eval():
    cfg = _cfg()
    router = ZayaRouter(cfg)
    _zero_router(router)
    width = cfg.zaya_mlp_expansion
    with torch.no_grad():
        router.down_proj.bias.fill_(1.0)
        router.router_mlp[0].bias.fill_(1.0)
        router.router_mlp[2].bias.zero_()
        router.router_mlp[2].bias[0] = 5.0
        router.router_mlp[4].weight.zero_()
        router.router_mlp[4].weight[0, 0] = 1.0
    hidden = torch.randn(6, 2, cfg.hidden_size)

    router.train()
    before = router.balancing_biases.detach().clone()
    probs, index, _ = router(hidden, None)
    assert torch.all(index == 0)
    assert not torch.allclose(before, router.balancing_biases)
    assert int(router.balance_step.item()) == 1
    # Mix weights are the unbiased probabilities, not the biased scores.
    assert torch.all(probs > 0)

    router.eval()
    frozen = router.balancing_biases.detach().clone()
    router(hidden, None)
    torch.testing.assert_close(frozen, router.balancing_biases)
    assert int(router.balance_step.item()) == 1
    del width


def test_stack_backward_is_finite():
    cfg = _cfg(num_layers=2)
    stack = ZayaStack(cfg)
    hidden = torch.randn(4, 2, cfg.hidden_size, requires_grad=True)
    out = stack(hidden, position_ids=None, attention_mask=None)
    out.sum().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    saw_grad = False
    for param in stack.parameters():
        if param.grad is None:
            continue
        assert torch.isfinite(param.grad).all()
        saw_grad = True
    assert saw_grad


def test_explicit_zaya_layers_override_the_alternation():
    kinds = layer_kinds(4, ["a", "a", "m", 16])
    assert kinds == ["a", "a", "m", "m"]
    with pytest.raises(ValueError):
        layer_kinds(3, ["a", "m"])
