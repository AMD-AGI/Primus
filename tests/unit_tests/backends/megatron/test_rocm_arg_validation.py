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


def test_turbo_ragged_grouped_gemm_disabled_is_noop():
    validate_turbo_ragged_grouped_gemm(SimpleNamespace())


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"enable_primus_turbo": False}, "requires enable_primus_turbo"),
        ({"use_turbo_grouped_gemm": False}, "requires enable_primus_turbo"),
        ({"fp8": None}, "supports only tensorwise FP8"),
        ({"fp8_recipe": "blockwise"}, "supports only tensorwise FP8"),
        ({"fp4": True}, "supports only tensorwise FP8"),
        (
            {"moe_router_padding_for_quantization": True},
            "requires moe_router_padding_for_quantization=False",
        ),
    ],
)
def test_turbo_ragged_grouped_gemm_rejects_unsupported_config(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_turbo_ragged_grouped_gemm(_ragged_args(**overrides))
