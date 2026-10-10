###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``moe_router_force_load_balancing_type: even`` in ``PrimusTopKRouter.routing``."""

from types import SimpleNamespace

import pytest
import torch
from megatron.core.transformer.moe.router import TopKRouter

import primus.backends.megatron.core.transformer.moe.router as router_mod
from primus.backends.megatron.core.transformer.moe.router import PrimusTopKRouter

NUM_TOKENS, NUM_EXPERTS, TOPK = 16, 32, 4


def _router():
    router = object.__new__(PrimusTopKRouter)
    router.config = SimpleNamespace(num_moe_experts=NUM_EXPERTS)
    router.topk = TOPK
    return router


def _sparse_topk_routing(self, logits, **kwargs):
    """Megatron's contract: scores are non-zero only on each token's real top-k."""
    vals, idx = torch.topk(torch.sigmoid(logits), TOPK, dim=1)
    scores = torch.zeros_like(logits).scatter(1, idx, vals / vals.sum(dim=1, keepdim=True))
    return scores, scores > 0


@pytest.mark.parametrize("moe_enable_deepep", [False, True])
def test_even_routing_keeps_scores_on_the_round_robin_slots(monkeypatch, moe_enable_deepep):
    args = SimpleNamespace(
        router_logit_softcapping=None,
        enable_primus_turbo=False,
        moe_use_fused_router_with_aux_score=False,
        moe_router_force_load_balancing=True,
        moe_router_force_load_balancing_type="even",
        moe_enable_deepep=moe_enable_deepep,
    )
    monkeypatch.setattr(router_mod, "get_args", lambda: args)
    monkeypatch.setattr(TopKRouter, "routing", _sparse_topk_routing)

    logits = torch.randn(NUM_TOKENS, NUM_EXPERTS, generator=torch.Generator().manual_seed(0))
    real_scores, _ = _sparse_topk_routing(None, logits)
    scores, routing_map = _router().routing(logits)

    slot = torch.arange(NUM_TOKENS * TOPK).view(NUM_TOKENS, TOPK) % NUM_EXPERTS
    expected_map = torch.zeros_like(routing_map).scatter(1, slot, True)
    assert torch.equal(routing_map, expected_map)
    # DeepEP's "even" dispatch gathers scores at these slots and MegaMoE takes their top-k,
    # so every routed slot must carry a real top-k weight.
    assert torch.equal(scores > 0, routing_map)
    torch.testing.assert_close(scores.sum(dim=1), real_scores.sum(dim=1))
