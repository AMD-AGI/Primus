###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

from types import SimpleNamespace

import pytest

from primus.backends.megatron.patches.args.rocm_arg_validation import (
    validate_turbo_fp8_permute,
    validate_turbo_grouped_gemm_without_padding,
)


def _no_padding_args(**overrides):
    values = {
        "turbo_grouped_gemm_without_padding": True,
        "enable_primus_turbo": True,
        "use_turbo_grouped_gemm": True,
        "fp8": "e4m3",
        "fp8_recipe": "tensorwise",
        "fp4": False,
        "moe_router_padding_for_quantization": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_turbo_grouped_gemm_without_padding_accepts_tensorwise_fp8():
    validate_turbo_grouped_gemm_without_padding(_no_padding_args())


@pytest.mark.parametrize(
    "overrides",
    [
        {"fp8_recipe": "blockwise"},
        {"fp8_recipe": "mxfp8"},
        {"fp8": None, "fp8_recipe": None, "fp4": "e2m1", "fp4_recipe": "mxfp4"},
        {"fp8": None, "fp8_recipe": None, "fp4": False},
    ],
)
def test_turbo_grouped_gemm_without_padding_accepts_all_supported_precision_modes(overrides):
    validate_turbo_grouped_gemm_without_padding(_no_padding_args(**overrides))


def test_turbo_grouped_gemm_without_padding_disabled_is_noop():
    validate_turbo_grouped_gemm_without_padding(SimpleNamespace())


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
def test_turbo_grouped_gemm_without_padding_rejects_unsupported_config(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_turbo_grouped_gemm_without_padding(_no_padding_args(**overrides))


def _fp8_permute_args(**overrides):
    values = {
        "turbo_fp8_permute": True,
        "enable_primus_turbo": True,
        "use_turbo_deepep": True,
        "turbo_fused_grouped_gemm": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"recompute_granularity": "full", "recompute_modules": ["moe_act"]},
        {"recompute_granularity": "selective", "recompute_modules": ["core_attn"]},
        {"recompute_granularity": "selective", "recompute_modules": None},
    ],
)
def test_turbo_fp8_permute_accepts_supported_config(overrides):
    validate_turbo_fp8_permute(_fp8_permute_args(**overrides))


def test_turbo_fp8_permute_disabled_is_noop():
    validate_turbo_fp8_permute(SimpleNamespace())
    validate_turbo_fp8_permute(SimpleNamespace(turbo_fp8_permute=False, use_turbo_deepep=False))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"enable_primus_turbo": False}, "requires enable_primus_turbo=True"),
        ({"use_turbo_deepep": False}, "requires use_turbo_deepep=True"),
        ({"turbo_fused_grouped_gemm": False}, "requires turbo_fused_grouped_gemm=True"),
        (
            {"recompute_granularity": "selective", "recompute_modules": ["core_attn", "moe_act"]},
            "does not support recompute of moe_act",
        ),
    ],
)
def test_turbo_fp8_permute_rejects_unsupported_config(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_turbo_fp8_permute(_fp8_permute_args(**overrides))
