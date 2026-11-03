###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Checks that the ZAYA1 fast paths compute the reference formulas, eagerly on CPU and compiled on GPU."""

import pytest
import torch
import torch.nn.functional as F

from primus.backends.megatron.core.models.zaya1.zaya1_modules import (
    CCA,
    ResidualScaling,
    RMSNorm,
    ZayaMoE,
    ZayaStack,
    _align_fp8_leading,
    _apply_rope,
    _cca_query_key,
    _fused_cca_query_key,
    _fused_scale_merge_norm,
    _residual_affine,
    _rope_cos_sin,
    _scale_merge_norm,
    _unpad_fp8_leading,
)

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="jit_fuser compiles GPU inputs only")


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


def _rel(got, ref):
    return ((got.float() - ref.float()).norm() / ref.float().norm()).item()


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


def test_every_stack_parameter_gets_a_gradient():
    torch.manual_seed(0)
    cfg = _cfg(num_layers=6)
    stack = ZayaStack(cfg).train()
    with torch.no_grad():
        for layer in stack.layers:
            if layer.zaya_block is not None:
                layer.zaya_block.router.balancing_biases.zero_()
                layer.zaya_block.router.balancing_biases[0] = 10.0
    hidden = torch.randn(8, 2, cfg.hidden_size, requires_grad=True)
    stack(hidden, position_ids=None, attention_mask=None).sum().backward()
    assert [name for name, p in stack.named_parameters() if p.grad is None] == []


@pytest.mark.parametrize("has_residual", [True, False])
def test_stage_boundary_is_scaling_merge_then_norm(has_residual):
    torch.manual_seed(0)
    width = 32
    scaling = ResidualScaling(width, has_residual=has_residual)
    norm = RMSNorm(width, 1e-5)
    with torch.no_grad():
        for p in (*scaling.parameters(), norm.weight):
            p.add_(0.1 * torch.randn_like(p))
    hidden = torch.randn(4, 2, width, dtype=torch.bfloat16)
    residual = torch.randn(4, 2, width) if has_residual else None

    scaled_res, scaled_hidden = scaling(residual, hidden)
    ref_merged = scaled_hidden if residual is None else scaled_res + scaled_hidden
    merged, normed = _fused_scale_merge_norm(
        hidden, residual, *_residual_affine(scaling), norm.weight, norm.eps
    )
    torch.testing.assert_close(merged, ref_merged, rtol=0, atol=0)
    torch.testing.assert_close(normed, norm(ref_merged.to(torch.bfloat16)), rtol=0, atol=0)


@needs_gpu
def test_compiled_stage_boundary_matches_eager():
    torch.manual_seed(0)
    width = 2048
    device = torch.device("cuda")
    hidden = torch.randn(16, 4, width, device=device, dtype=torch.bfloat16, requires_grad=True)
    residual = torch.randn(16, 4, width, device=device, requires_grad=True)
    params = [
        (base + 0.1 * torch.randn(width, device=device)).to(torch.bfloat16).requires_grad_()
        for base in (1.0, 0.0, 1.0, 0.0, 1.0)
    ]
    upstream = (torch.randn_like(residual), torch.randn_like(hidden))

    results = []
    for fn in (_fused_scale_merge_norm, _scale_merge_norm):
        outs = fn(hidden, residual, *params, 1e-5)
        grads = torch.autograd.grad(outs, (hidden, residual, *params), upstream)
        results.append((outs, grads))
    (got_outs, got_grads), (ref_outs, ref_grads) = results
    torch.testing.assert_close(got_outs[0], ref_outs[0])
    # Eager rounds the residual to bf16 before the norm and the compiled kernel
    # does not, so elements differ by a bf16 ulp; compare whole tensors.
    for got, ref in zip(got_outs[1:] + got_grads, ref_outs[1:] + ref_grads):
        assert _rel(got, ref) < 1e-2


def test_partial_rope_rotates_each_half_pair_by_its_angle():
    torch.manual_seed(0)
    seq, batch, heads, head_dim, rotary_dim = 6, 2, 3, 16, 8
    half = rotary_dim // 2
    x = torch.randn(seq, batch, heads, head_dim)
    pos = torch.randint(0, 100, (batch, seq))
    got = _apply_rope(x, *_rope_cos_sin(pos, seq, rotary_dim, 10000.0, x.device))

    inv_freq = 1.0 / 10000.0 ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim)
    angle = pos.t().float()[:, :, None, None] * inv_freq
    pair = torch.complex(x[..., :half], x[..., half:rotary_dim]) * torch.polar(torch.ones_like(angle), angle)
    torch.testing.assert_close(got[..., :half], pair.real)
    torch.testing.assert_close(got[..., half:rotary_dim], pair.imag)
    torch.testing.assert_close(got[..., rotary_dim:], x[..., rotary_dim:])


def _main_cca_query_key(cca, hidden):
    """CCA query and key as main computes them: ``nn.Conv1d`` over the padded, concatenated projections."""
    seq, batch, _ = hidden.shape
    q_raw, k_raw = cca.linear_q(hidden), cca.linear_k(hidden)
    pad = (cca.cca_time0 - 1) + (cca.cca_time1 - 1)
    qk_ncl = F.pad(torch.cat((q_raw, k_raw), dim=-1).permute(1, 2, 0), (pad, 0))
    qk_out = cca.conv_qk(qk_ncl).permute(2, 0, 1)
    query, key = cca._grouped_means(
        qk_out[..., : cca.latent_q].view(seq, batch, cca.num_q_heads, cca.head_dim),
        qk_out[..., cca.latent_q :].view(seq, batch, cca.num_k_heads, cca.head_dim),
        q_raw.view(seq, batch, cca.num_q_heads, cca.head_dim),
        k_raw.view(seq, batch, cca.num_k_heads, cca.head_dim),
    )
    query, key = cca._qk_norm_and_temp(query, key)
    rope = _rope_cos_sin(None, seq, cca.rotary_dim, cca.rope_base, hidden.device)
    return _apply_rope(query, *rope), _apply_rope(key, *rope)


@pytest.mark.parametrize("kernels", [(2, 2), (3, 2), (2, 4)])
def test_cca_convolutions_as_gemms_match_conv1d(kernels):
    torch.manual_seed(0)
    cca = CCA(_cfg(cca_time0=kernels[0], cca_time1=kernels[1]))
    with torch.no_grad():
        for p in (*cca.conv_qk.parameters(), cca.temp):
            p.normal_()
    hidden = torch.randn(7, 3, 32, requires_grad=True)
    leaves = (hidden, cca.linear_q.weight, cca.linear_k.weight, *cca.conv_qk.parameters(), cca.temp)

    got = cca.project(hidden, position_ids=None)[:2]
    ref = _main_cca_query_key(cca, hidden)
    upstream = tuple(torch.randn_like(t) for t in ref)
    for g, r in zip(got, ref):
        torch.testing.assert_close(g, r)
    got_grads = torch.autograd.grad(got, leaves, upstream)
    ref_grads = torch.autograd.grad(ref, leaves, upstream)
    for g, r in zip(got_grads, ref_grads):
        torch.testing.assert_close(g, r, rtol=1e-5, atol=1e-5)


@needs_gpu
def test_compiled_cca_query_key_matches_eager():
    torch.manual_seed(0)
    device = torch.device("cuda")
    seq, batch, q_heads, k_heads, head_dim = 32, 4, 8, 2, 128
    width = (q_heads + k_heads) * head_dim

    def leaf(*shape, scale=1.0):
        return (scale * torch.randn(*shape, device=device)).to(torch.bfloat16).requires_grad_()

    inputs = (
        leaf(seq, batch, q_heads, head_dim),
        leaf(seq, batch, k_heads, head_dim),
        leaf(width, 1, 2, scale=0.5),
        leaf(width, scale=0.1),
        leaf(width, head_dim, 2, scale=0.1),
        leaf(width, scale=0.1),
    )
    temp = (1 + 0.1 * torch.randn(k_heads, device=device)).to(torch.bfloat16).requires_grad_()
    pos = torch.arange(seq, device=device).expand(batch, -1)
    rope = _rope_cos_sin(pos, seq, head_dim // 2, 10000.0, device)
    upstream = (torch.randn_like(inputs[0]), torch.randn_like(inputs[1]))

    results = []
    for fn in (_fused_cca_query_key, _cca_query_key):
        outs = fn(*inputs, temp, False, rope)
        results.append(outs + torch.autograd.grad(outs, (*inputs, temp), upstream))
    for got, ref in zip(*results):
        assert _rel(got, ref) < 1e-2


def test_fp8_leading_pad_gives_an_empty_slice_eight_rows():
    padded, meta = _align_fp8_leading(torch.zeros(0, 16))
    assert padded.shape == (8, 16)
    restored = _unpad_fp8_leading(torch.ones(8, 4), meta)
    assert restored.shape == (0, 4)
