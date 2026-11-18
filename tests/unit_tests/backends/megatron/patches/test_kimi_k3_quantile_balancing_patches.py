###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The Quantile Balancing patch takes the call ``finalize_model_grads`` makes on Megatron-core 0.19."""

import importlib
from types import SimpleNamespace

import pytest

from primus.backends.megatron.core.transformer.kimi_k3.moe import k3_quantile_balancing
from primus.backends.megatron.patches.kimi_k3_quantile_balancing_patches import (
    patch_quantile_balancing,
)


@pytest.fixture
def finalize_module(monkeypatch):
    module = importlib.import_module("megatron.core.distributed.finalize_model_grads")
    calls = []
    monkeypatch.setattr(
        module,
        "_update_router_expert_bias",
        lambda model, config, tp_dp_cp_group=None: calls.append(("sign", tp_dp_cp_group)),
    )
    return module, calls


def _install(monkeypatch, finalize_module, routers):
    module, calls = finalize_module
    monkeypatch.setattr(k3_quantile_balancing, "collect_quantile_balancing_routers", lambda model: routers)
    monkeypatch.setattr(
        k3_quantile_balancing,
        "update_router_expert_bias_quantile",
        lambda model, config, *, group=None, step=None: calls.append(("quantile", group, step)),
    )
    patch_quantile_balancing(SimpleNamespace())
    return module._update_router_expert_bias, calls


def test_quantile_rule_uses_the_passed_group(monkeypatch, finalize_module):
    update, calls = _install(monkeypatch, finalize_module, routers=[object()])
    group = object()

    update([object()], SimpleNamespace(), tp_dp_cp_group=group)

    assert calls == [("quantile", group, 1)]


def test_sign_rule_fallback_forwards_the_group(monkeypatch, finalize_module):
    update, calls = _install(monkeypatch, finalize_module, routers=[])
    group = object()

    update([object()], SimpleNamespace(), tp_dp_cp_group=group)

    assert calls == [("sign", group)]
