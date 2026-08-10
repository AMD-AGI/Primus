###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

from types import SimpleNamespace

import pytest

from primus.backends.megatron.patches.args.rocm_arg_validation import (
    validate_turbo_ragged_grouped_gemm,
)


def _ragged_args(**overrides):
    values = {
        "use_turbo_ragged_grouped_gemm": True,
        "enable_primus_turbo": True,
        "use_turbo_grouped_gemm": True,
        "fp8": "e4m3",
        "fp8_recipe": "tensorwise",
        "fp4": False,
        "moe_router_padding_for_quantization": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_turbo_ragged_grouped_gemm_accepts_tensorwise_fp8():
    validate_turbo_ragged_grouped_gemm(_ragged_args())


@pytest.mark.parametrize(
    "overrides",
    [
        {"fp8_recipe": "blockwise"},
        {"fp8_recipe": "mxfp8"},
        {"fp8": None, "fp8_recipe": None, "fp4": "e2m1", "fp4_recipe": "mxfp4"},
        {"fp8": None, "fp8_recipe": None, "fp4": False},
    ],
)
def test_turbo_ragged_grouped_gemm_accepts_all_supported_precision_modes(overrides):
    validate_turbo_ragged_grouped_gemm(_ragged_args(**overrides))


def test_turbo_ragged_grouped_gemm_disabled_is_noop():
    validate_turbo_ragged_grouped_gemm(SimpleNamespace())


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"enable_primus_turbo": False}, "requires enable_primus_turbo"),
        ({"use_turbo_grouped_gemm": False}, "requires enable_primus_turbo"),
        (
            {"moe_router_padding_for_quantization": True},
            "requires moe_router_padding_for_quantization=False",
        ),
    ],
)
def test_turbo_ragged_grouped_gemm_rejects_unsupported_config(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_turbo_ragged_grouped_gemm(_ragged_args(**overrides))
