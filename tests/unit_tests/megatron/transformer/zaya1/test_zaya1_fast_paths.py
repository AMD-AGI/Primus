###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CPU checks that the ZAYA1 fast paths compute the reference formulas."""

import torch

from primus.backends.megatron.core.models.zaya1.zaya1_modules import (
    ZayaMoE,
    _align_fp8_leading,
    _unpad_fp8_leading,
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


def _masked_moe(moe, hidden, probs, index):
    """Per-expert boolean masks: the dispatch the sorted one replaced."""
    flat = hidden.reshape(-1, hidden.shape[-1])
    idx = index[..., 0].reshape(-1)
    prob = probs[..., 0].reshape(-1, 1).to(dtype=flat.dtype)
    out = flat.new_zeros(flat.shape)
    for expert_id, expert in enumerate(moe.local_experts):
        chosen = idx == expert_id
        if torch.any(chosen):
            out[chosen] = expert(flat[chosen]) * prob[chosen]
    skip = idx == moe.num_real_experts
    out[skip] = flat[skip] * prob[skip]
    return out.view_as(hidden)


def test_sorted_dispatch_matches_per_expert_masks():
    torch.manual_seed(0)
    cfg = _cfg()
    moe = ZayaMoE(cfg).eval()
    hidden = torch.randn(16, 3, cfg.hidden_size, requires_grad=True)
    probs, index, _ = moe.router(hidden, None)
    assert len(torch.unique(index)) >= 3

    out, _ = moe(hidden, None)
    upstream = torch.randn_like(out)
    out.backward(upstream)
    got_input = hidden.grad.clone()
    got_params = {name: p.grad.clone() for name, p in moe.named_parameters() if p.grad is not None}

    hidden.grad = None
    moe.zero_grad(set_to_none=True)
    ref = _masked_moe(moe, hidden, probs, index)
    ref.backward(upstream)
    torch.testing.assert_close(out, ref)
    torch.testing.assert_close(got_input, hidden.grad)
    for name, p in moe.named_parameters():
        if p.grad is not None:
            torch.testing.assert_close(got_params[name], p.grad, msg=name)


def test_every_expert_gets_a_gradient_without_tokens():
    torch.manual_seed(0)
    cfg = _cfg()
    moe = ZayaMoE(cfg).eval()
    with torch.no_grad():
        moe.router.balancing_biases.zero_()
        moe.router.balancing_biases[0] = 10.0
    hidden = torch.randn(8, 2, cfg.hidden_size)
    out, _ = moe(hidden, None)
    out.sum().backward()
    for expert_id, expert in enumerate(moe.local_experts):
        for name, p in expert.named_parameters():
            assert p.grad is not None, f"expert {expert_id} {name}"
            if expert_id:
                assert torch.count_nonzero(p.grad) == 0


def test_fp8_leading_pad_gives_an_empty_slice_eight_rows():
    padded, meta = _align_fp8_leading(torch.zeros(0, 16))
    assert padded.shape == (8, 16)
    restored = _unpad_fp8_leading(torch.ones(8, 4), meta)
    assert restored.shape == (0, 4)
