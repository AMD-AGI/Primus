###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The MoE's router and split dispatch, with zaya_sort_dispatch on (sort) and off (boolean masks),
against main's router and boolean-mask dispatch, bit for bit."""

import pytest
import torch
import torch.nn.functional as F
from test_zaya1_math import _cfg  # noqa: E402  (pytest puts this dir on sys.path)

from primus.backends.megatron.core.models.zaya1.zaya1_modules import ZayaMoE


def _main_expert(expert, hidden):
    """SwiGLUExpert.forward on main."""
    gate, up = expert.linear_fc1(hidden).chunk(2, dim=-1)
    return expert.linear_fc2(F.silu(gate) * up)


def _main_router(router, hidden_states, prev_router):
    """ZayaRouter.forward on main, in eval (no balancing-bias update)."""
    seq, batch, _ = hidden_states.shape
    routed = router.down_proj(hidden_states.reshape(seq * batch, router.hidden_size))
    if router.use_eda and prev_router is not None:
        routed = routed + prev_router.reshape(seq * batch, router.width) * router.router_states_scale
    next_router = routed.view(seq, batch, router.width)
    logits = router.rmsnorm_eda(routed)
    for stage in router.router_mlp:
        logits = stage(logits)
    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    with torch.no_grad():
        biased = probs.detach().float() + router.balancing_biases.float()
        _, index = torch.topk(biased, k=router.topk, dim=-1)
    route_prob = torch.gather(probs, dim=-1, index=index)
    route_prob = route_prob.to(dtype=hidden_states.dtype).view(seq, batch, router.topk)
    return route_prob, index.view(seq, batch, router.topk), next_router


def _main_dispatch(moe, hidden_states, prev_router):
    """ZayaMoE.forward on main, which masks out each expert's tokens in turn."""
    probs, index, next_router = _main_router(moe.router, hidden_states, prev_router)
    # Top-1 is the released configuration. The mix weight is the unbiased
    # gathered probability.
    prob = probs[..., 0]
    idx = index[..., 0]
    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    flat_idx = idx.reshape(-1)
    flat_prob = prob.reshape(-1, 1).to(dtype=flat.dtype)
    out = flat.new_zeros(flat.shape)
    for expert_id, expert in enumerate(moe.local_experts):
        chosen = flat_idx == expert_id
        if not torch.any(chosen):
            continue
        out[chosen] = _main_expert(expert, flat[chosen]) * flat_prob[chosen]
    if moe.router.use_mod:
        skip = flat_idx == moe.num_real_experts
        if torch.any(skip):
            out[skip] = flat[skip] * flat_prob[skip]
    out = out.view_as(hidden_states)
    return out, next_router


def _branch_dispatch(moe, hidden_states, prev_router):
    return moe(hidden_states, prev_router)


def _run(moe, dispatch, hidden, prev, out_weight, router_weight):
    moe.zero_grad(set_to_none=True)
    hidden = hidden.detach().clone().requires_grad_(True)
    prev = prev.detach().clone().requires_grad_(True)
    out, next_router = dispatch(moe, hidden, prev)
    loss = (out.float() * out_weight).sum() + (next_router.float() * router_weight).sum()
    loss.backward()
    grads = {f"{name} grad": param.grad for name, param in moe.named_parameters()}
    return {"output": out, "hidden_states grad": hidden.grad, "prev_router grad": prev.grad, **grads}


@pytest.mark.parametrize("sort_dispatch", [True, False])
@pytest.mark.parametrize(
    "device, dtype, hidden_size",
    [
        ("cpu", torch.float32, 64),
        ("cpu", torch.bfloat16, 64),
        ("cuda", torch.bfloat16, 64),
        ("cuda", torch.float32, 64),
        # ZAYA1-8B's hidden size: in fp32 the GPU rounds one gate-gradient sum
        # over all tokens differently from main's sum per expert.
        ("cuda", torch.float32, 2048),
    ],
)
def test_split_dispatch_matches_main_bit_for_bit(device, dtype, hidden_size, sort_dispatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("No GPU.")
    torch.manual_seed(0)
    cfg = _cfg(
        hidden_size=hidden_size,
        moe_ffn_hidden_size=32,
        zaya_mlp_expansion=16,
        zaya_sort_dispatch=sort_dispatch,
    )
    moe = ZayaMoE(cfg)
    assert not moe.use_turbo_gemm and moe.sort_dispatch is sort_dispatch
    with torch.no_grad():
        for module in moe.router.router_mlp:
            if hasattr(module, "weight"):
                module.weight.mul_(20.0)  # spread the tokens over the experts and MoD
        # Expert 1 gets no tokens, and MoD, the last slot, gets some instead of
        # none. On CPU at hidden size 64 the groups are 2, 0, 37, 76 and 13 tokens.
        moe.router.balancing_biases.copy_(torch.tensor([0.0, -10.0, 0.0, 0.0, 0.0]))
    # Eval keeps the balancing biases still between the two runs.
    moe = moe.to(device=device, dtype=dtype).eval()
    seq, batch = 32, 4
    hidden = torch.randn(seq, batch, cfg.hidden_size).to(device=device, dtype=dtype)
    prev = torch.randn(seq, batch, cfg.zaya_mlp_expansion).to(device=device, dtype=dtype)
    out_weight = torch.randn(seq, batch, cfg.hidden_size).to(device)
    router_weight = torch.randn(seq, batch, cfg.zaya_mlp_expansion).to(device)
    with torch.no_grad():
        _, index, _ = moe.router(hidden, prev)
    counts = torch.bincount(index.reshape(-1), minlength=moe.router.num_scored).tolist()
    assert counts[1] == 0 and counts[-1] > 0 and sum(c > 0 for c in counts[:-1]) >= 2, counts

    expected = _run(moe, _main_dispatch, hidden, prev, out_weight, router_weight)
    actual = _run(moe, _branch_dispatch, hidden, prev, out_weight, router_weight)
    assert actual.keys() == expected.keys()
    assert expected["local_experts.1.linear_fc1.weight grad"] is None
    for name, ref in expected.items():
        got = actual[name]
        if ref is None:
            assert got is None, name
            continue
        assert got.dtype == ref.dtype and torch.equal(got, ref), name
