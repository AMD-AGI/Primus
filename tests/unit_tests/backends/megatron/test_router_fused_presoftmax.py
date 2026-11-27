###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Turbo's fused router (``moe_use_fused_router_with_aux_score``) must match Megatron's routing."""

from types import SimpleNamespace

import pytest
import torch
from megatron.core.transformer.moe.moe_utils import topk_routing_with_score_function
from megatron.core.transformer.moe.router import TopKRouter

import primus.backends.megatron.core.transformer.moe.router as router_mod

NUM_TOKENS, NUM_EXPERTS, TOPK = 64, 8, 2


def _router(pre_softmax, scaling_factor=None, num_groups=None, group_topk=None):
    router = object.__new__(router_mod.PrimusTopKRouter)
    router.config = SimpleNamespace(
        num_moe_experts=NUM_EXPERTS,
        moe_router_topk=TOPK,
        moe_router_num_groups=num_groups,
        moe_router_group_topk=group_topk,
        moe_router_score_function="softmax",
        moe_router_pre_softmax=pre_softmax,
        moe_router_topk_scaling_factor=scaling_factor,
        moe_expert_capacity_factor=None,
        moe_z_loss_coeff=None,
    )
    router.topk = TOPK
    router.training = False
    router.enable_expert_bias = False
    return router


def _have_turbo():
    try:
        import primus_turbo.pytorch  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available()


@pytest.mark.skipif(not _have_turbo(), reason="needs primus_turbo and a GPU")
@pytest.mark.parametrize("pre_softmax", [False, True])
@pytest.mark.parametrize("scaling_factor", [None, 2.5])
def test_fused_router_matches_megatron(pre_softmax, scaling_factor):
    torch.manual_seed(0)
    logits = torch.randn(NUM_TOKENS, NUM_EXPERTS, device="cuda", dtype=torch.float32)
    upstream = torch.randn(NUM_TOKENS, NUM_EXPERTS, device="cuda", dtype=torch.float32)

    fused_logits = logits.clone().requires_grad_()
    probs, routing_map = _router(pre_softmax, scaling_factor).fused_router_and_auxiliary_loss(fused_logits)
    (probs * upstream).sum().backward()

    ref_logits = logits.clone().requires_grad_()
    ref_probs, ref_map = topk_routing_with_score_function(
        ref_logits,
        TOPK,
        use_pre_softmax=pre_softmax,
        scaling_factor=scaling_factor,
        score_function="softmax",
    )
    (ref_probs * upstream).sum().backward()

    assert torch.equal(routing_map, ref_map)
    torch.testing.assert_close(probs, ref_probs)
    torch.testing.assert_close(fused_logits.grad, ref_logits.grad)
    if not pre_softmax:
        torch.testing.assert_close(probs.sum(dim=-1), torch.full_like(probs[:, 0], scaling_factor or 1.0))


@pytest.mark.parametrize("pre_softmax, expect_fused", [(False, False), (True, True)])
def test_grouped_post_softmax_falls_back_to_megatron(monkeypatch, pre_softmax, expect_fused):
    args = SimpleNamespace(
        router_logit_softcapping=None,
        enable_primus_turbo=True,
        moe_use_fused_router_with_aux_score=True,
        moe_router_force_load_balancing=False,
    )
    monkeypatch.setattr(router_mod, "get_args", lambda: args)
    calls = []

    def fused(self, logits):
        calls.append("fused")
        return logits, torch.ones_like(logits, dtype=torch.bool)

    def megatron(self, logits, **kwargs):
        calls.append("megatron")
        return logits, torch.ones_like(logits, dtype=torch.bool)

    monkeypatch.setattr(router_mod.PrimusTopKRouter, "fused_router_and_auxiliary_loss", fused)
    monkeypatch.setattr(TopKRouter, "routing", megatron)

    _router(pre_softmax, num_groups=4, group_topk=2).routing(torch.randn(NUM_TOKENS, NUM_EXPERTS))
    assert calls == ["fused" if expect_fused else "megatron"]
