###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

from types import SimpleNamespace

import pytest
import torch

from primus.backends.megatron.core.transformer.experts import PrimusGroupedMLP


def _grouped_mlp_state(**overrides):
    values = {
        "config": SimpleNamespace(fp8="e4m3", fp4=False),
        "moe_router_padding_for_quantization": False,
        "turbo_grouped_gemm_without_padding": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    "config",
    [
        SimpleNamespace(fp8="e4m3", fp4=False),
        SimpleNamespace(fp8=None, fp4="e2m1"),
    ],
)
def test_turbo_grouped_gemm_without_padding_skips_explicit_padding_for_low_precision(config):
    mlp = _grouped_mlp_state(config=config)

    assert not PrimusGroupedMLP._use_explicit_quantization_padding(mlp)


def test_disabled_turbo_grouped_gemm_without_padding_keeps_explicit_padding():
    mlp = _grouped_mlp_state(turbo_grouped_gemm_without_padding=False)

    assert PrimusGroupedMLP._use_explicit_quantization_padding(mlp)


@pytest.mark.parametrize(
    "overrides",
    [
        {"config": SimpleNamespace(fp8=None, fp4=False)},
        {"moe_router_padding_for_quantization": True},
    ],
)
def test_explicit_padding_is_not_duplicated(overrides):
    mlp = _grouped_mlp_state(**overrides)

    assert not PrimusGroupedMLP._use_explicit_quantization_padding(mlp)


class _FakeGroupedLinear(torch.nn.Module):
    def __init__(self, name, events):
        super().__init__()
        self.name = name
        self.events = events
        self.register_forward_pre_hook(lambda module, _inputs: events.append(f"{module.name}.pre_hook"))

    def prepare_weights(self, x, turbo_op, weight_role):
        self.events.append(f"{self.name}.prepare_weights({turbo_op}, {weight_role})")
        return x, f"{self.name}.weight", "pattern"


def test_fused_grouped_mlp_runs_linear_pre_hooks_before_preparing_weights():
    events = []
    mlp = SimpleNamespace(
        linear_fc1=_FakeGroupedLinear("fc1", events),
        linear_fc2=_FakeGroupedLinear("fc2", events),
    )
    x = torch.zeros(2, 4)

    out = PrimusGroupedMLP._prepare_fused_grouped_mlp_weights(mlp, x)

    assert events == [
        "fc1.pre_hook",
        "fc2.pre_hook",
        "fc1.prepare_weights(grouped_mlp, fc1)",
        "fc2.prepare_weights(grouped_mlp, fc2)",
    ]
    assert out == (x, "fc1.weight", "fc2.weight", "pattern")
