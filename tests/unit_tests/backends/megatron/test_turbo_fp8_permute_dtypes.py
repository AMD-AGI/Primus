###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the FP8 dtypes the Turbo DeepEP dispatcher asks the permute for under
``turbo_fp8_permute``. Quantizing before permute is only exact for tensorwise current
scaling, so every other state must keep the bf16 permute.
"""

from types import SimpleNamespace

import pytest

from tests.utils import skip_if_no_cuda

skip_if_no_cuda()

from primus_turbo.pytorch.core.low_precision import (  # noqa: E402  isort:skip
    Format,
    ScalingGranularity,
    float8_e4m3,
    float8_e5m2,
)

from primus.backends.megatron.core.extensions.primus_turbo import (  # noqa: E402  isort:skip
    PrimusTurboDeepEPTokenDispatcher,
    PrimusTurboLowPrecisionGlobalStateManager,
    PrimusTurboQuantConfig,
)


def _dtypes(monkeypatch, *, flag=True, turbo_fp8=True, quant_config=None):
    monkeypatch.setattr(PrimusTurboLowPrecisionGlobalStateManager, "PRIMUS_TURBO_FP8_ENABLED", turbo_fp8)
    monkeypatch.setattr(PrimusTurboLowPrecisionGlobalStateManager, "PRIMUS_TURBO_QUANT_CONFIG", quant_config)
    dispatcher = SimpleNamespace(turbo_fp8_permute=flag)
    return PrimusTurboDeepEPTokenDispatcher._fp8_permute_dtypes(dispatcher)


@pytest.mark.parametrize(
    ("fmt", "expected"),
    [
        (Format.HYBRID, (float8_e4m3, float8_e5m2)),
        (Format.E4M3, (float8_e4m3, float8_e4m3)),
    ],
)
def test_tensorwise_returns_input_and_grad_dtypes(monkeypatch, fmt, expected):
    quant_config = PrimusTurboQuantConfig(format=fmt, granularity=ScalingGranularity.TENSORWISE)
    assert _dtypes(monkeypatch, quant_config=quant_config) == expected


def test_flag_off_keeps_bf16_permute(monkeypatch):
    quant_config = PrimusTurboQuantConfig(format=Format.HYBRID, granularity=ScalingGranularity.TENSORWISE)
    assert _dtypes(monkeypatch, flag=False, quant_config=quant_config) == (None, None)


def test_turbo_fp8_off_keeps_bf16_permute(monkeypatch):
    """A layer outside Turbo FP8 (e.g. pinned to BF16) must not get FP8 tokens."""
    quant_config = PrimusTurboQuantConfig(format=Format.HYBRID, granularity=ScalingGranularity.TENSORWISE)
    assert _dtypes(monkeypatch, turbo_fp8=False, quant_config=quant_config) == (None, None)


def test_blockwise_keeps_bf16_permute(monkeypatch):
    quant_config = PrimusTurboQuantConfig(
        format=Format.HYBRID, granularity=ScalingGranularity.BLOCKWISE, block_size=128
    )
    assert _dtypes(monkeypatch, quant_config=quant_config) == (None, None)
