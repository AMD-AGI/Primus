###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Regression tests for GPT-OSS router gate-bias initialization."""

from types import SimpleNamespace

import pytest
import torch

import primus.backends.torchtitan.patches.gptoss_router_bias_patches as bias_patch
from primus.core.patches import PatchContext
from primus.core.patches.patch_registry import PatchRegistry

PATCH_ID = "torchtitan.gptoss.router_bias_init"


def _ctx(model_name: str, backend_version: str = "0.2.2") -> PatchContext:
    params = SimpleNamespace(model=SimpleNamespace(name=model_name))
    module_config = SimpleNamespace(params=params)
    return PatchContext(
        backend="torchtitan",
        backend_version=backend_version,
        phase="setup",
        extra={"module_config": module_config},
    )


def test_patch_registration_and_scope():
    patch = PatchRegistry.get(PATCH_ID)
    assert patch is not None
    assert patch.backend == "torchtitan"
    assert patch.backend_version_patterns == ["0.2.2"]
    assert patch.applies_to(_ctx("gpt_oss"))
    assert not patch.applies_to(_ctx("deepseek_v3"))
    assert not patch.applies_to(_ctx("gpt_oss", backend_version="0.3.0"))


def test_patch_initializes_bias_after_to_empty(monkeypatch):
    pytest.importorskip("torchtitan")
    from torchtitan.models.moe.moe import TokenChoiceTopKRouter

    def make_materialized_router():
        with torch.device("meta"):
            router = TokenChoiceTopKRouter(
                dim=16,
                num_experts=8,
                num_expert_groups=None,
                num_limited_groups=None,
                top_k=2,
                score_func="softmax",
                route_norm=True,
                route_scale=1.0,
                gate_bias=True,
            )
        return router.to_empty(device="cpu")

    original = TokenChoiceTopKRouter.init_weights
    # Record the original value so pytest restores it after the handler mutates
    # the class directly.
    monkeypatch.setattr(TokenChoiceTopKRouter, "init_weights", original)

    unpatched_router = make_materialized_router()
    with torch.no_grad():
        # A sentinel models arbitrary allocator contents after to_empty().
        unpatched_router.gate.bias.fill_(float("nan"))
        original(unpatched_router, 0.02)
    unpatched_scores = torch.softmax(
        unpatched_router.gate(torch.randn(4, 16)).float(),
        dim=1,
    )
    assert not torch.isfinite(unpatched_router.gate.bias).all()
    assert not torch.isfinite(unpatched_scores).all()

    bias_patch.patch_gptoss_router_bias_init(_ctx("gpt_oss"))
    wrapped = TokenChoiceTopKRouter.init_weights
    assert wrapped is not original

    # Calling the patch twice must not stack wrappers in a long-lived process.
    bias_patch.patch_gptoss_router_bias_init(_ctx("gpt_oss"))
    assert TokenChoiceTopKRouter.init_weights is wrapped

    router = make_materialized_router()
    with torch.no_grad():
        router.gate.bias.fill_(float("nan"))
        router.init_weights(0.02)

    assert torch.isfinite(router.gate.weight).all()
    torch.testing.assert_close(router.gate.bias, torch.zeros_like(router.gate.bias))

    scores = torch.softmax(router.gate(torch.randn(4, 16)).float(), dim=1)
    assert torch.isfinite(scores).all()
