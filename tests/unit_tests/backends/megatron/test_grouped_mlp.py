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


class _ProbsFusingActivation:
    """An activation that reports whether it folds the router probability in, and records which entry ran."""

    def __init__(self, fuses_probs):
        self.fuses_probs = fuses_probs
        self.calls = []

    def __call__(self, x):
        self.calls.append("plain")
        return 2 * x

    def forward_with_probs(self, x, probs):
        self.calls.append("with_probs")
        return 3 * x * probs


def _bias_act_state(activation):
    return SimpleNamespace(config=SimpleNamespace(use_te_activation_func=True), activation_func=activation)


def test_bias_act_func_hands_the_probs_to_an_activation_that_fuses_them():
    act = _ProbsFusingActivation(fuses_probs=True)
    x, probs = torch.ones(4, 8), torch.full((4, 1), 0.5)

    out = PrimusGroupedMLP.bias_act_func(_bias_act_state(act), x, None, probs)

    assert act.calls == ["with_probs"]
    torch.testing.assert_close(out, 3 * x * probs)


@pytest.mark.parametrize(
    "fuses_probs, with_bias, with_probs",
    [(False, False, True), (True, True, True), (True, False, False)],
)
def test_bias_act_func_keeps_the_unfused_path(fuses_probs, with_bias, with_probs):
    act = _ProbsFusingActivation(fuses_probs=fuses_probs)
    x = torch.ones(4, 8)
    bias = torch.ones(8) if with_bias else None
    probs = torch.full((4, 1), 0.5) if with_probs else None

    out = PrimusGroupedMLP.bias_act_func(_bias_act_state(act), x, bias, probs)

    assert act.calls == ["plain"]
    expected = 2 * (x + bias if with_bias else x)
    torch.testing.assert_close(out, expected * probs if with_probs else expected)
