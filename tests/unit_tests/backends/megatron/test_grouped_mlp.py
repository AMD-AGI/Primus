###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

from types import SimpleNamespace

import pytest

from primus.backends.megatron.core.transformer.experts import PrimusGroupedMLP


def _grouped_mlp_state(**overrides):
    values = {
        "config": SimpleNamespace(fp8="e4m3", fp4=False),
        "moe_router_padding_for_quantization": False,
        "use_turbo_ragged_grouped_gemm": True,
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
def test_turbo_ragged_grouped_gemm_skips_explicit_padding_for_low_precision(config):
    mlp = _grouped_mlp_state(config=config)

    assert not PrimusGroupedMLP._use_explicit_quantization_padding(mlp)


def test_disabled_turbo_ragged_grouped_gemm_keeps_explicit_padding():
    mlp = _grouped_mlp_state(use_turbo_ragged_grouped_gemm=False)

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
