###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the conditions of the FLUX performance patches.

WHAT THESE ARE DEFENDING:
  Every ``*_patches.py`` module is imported for every run, so a FLUX patch whose condition only
  looked at its flag would rewrite diffusers' FLUX classes in a run that trains another model.
  Each condition must decline unless its flag is set, decline for another configured transformer,
  and apply when the config names FLUX or names no class at all (a ``from_pretrained`` fine-tune).

  The fused ops compute projections from their weights, which would skip the forward of a linear
  that a low-precision swap replaced, so they must take only a stock ``torch.nn.Linear``.
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from primus.core.patches import PatchContext  # noqa: E402

FLUX_PATCHES = [
    ("flux_mlp_gelu_patches", "primus_perf.flux_fused_gelu"),
    ("flux_qk_norm_rope_patches", "primus_perf.flux_fused_qk_norm_rope"),
    ("flux_adaln_patches", "primus_perf.flux_fused_adaln"),
    ("flux_adaln_dual_patches", "primus_perf.flux_fused_adaln_dual"),
]


def _ctx(transformer_cls):
    spec = SimpleNamespace(transformer_cls=transformer_cls) if transformer_cls else None
    params = SimpleNamespace(model=SimpleNamespace(pipeline_spec=spec))
    return PatchContext(
        backend="automodel",
        phase="before_train",
        extra={"module_config": SimpleNamespace(params=params)},
    )


def _condition(module_name):
    import importlib

    return importlib.import_module(f"primus.backends.automodel.patches.{module_name}")._enabled


class TestFluxPatchConditions:
    @pytest.mark.parametrize("module_name,flag", FLUX_PATCHES)
    def test_off_by_default(self, module_name, flag):
        assert _condition(module_name)(_ctx("FluxTransformer2DModel")) is False

    @pytest.mark.parametrize("module_name,flag", FLUX_PATCHES)
    def test_applies_to_flux(self, module_name, flag, set_option):
        set_option(flag, True)
        assert _condition(module_name)(_ctx("FluxTransformer2DModel")) is True

    @pytest.mark.parametrize("module_name,flag", FLUX_PATCHES)
    def test_declines_another_transformer(self, module_name, flag, set_option):
        set_option(flag, True)
        assert _condition(module_name)(_ctx("WanTransformer3DModel")) is False

    @pytest.mark.parametrize("module_name,flag", FLUX_PATCHES)
    def test_applies_when_no_class_is_named(self, module_name, flag, set_option):
        set_option(flag, True)
        assert _condition(module_name)(_ctx(None)) is True


class SwappedLinear(torch.nn.Linear):
    """What a low-precision swap puts in place of a linear: a subclass with its own forward."""


class TestStockLinearOnly:
    def test_stock_linear(self):
        from primus.backends.automodel.models.flux.plain_modules import is_plain_linear

        assert is_plain_linear(torch.nn.Linear(4, 4)) is True

    def test_declines_a_swapped_linear(self):
        from primus.backends.automodel.models.flux.plain_modules import is_plain_linear

        assert is_plain_linear(SwappedLinear(4, 4)) is False

    def test_declines_without_bias_or_with_hooks(self):
        from primus.backends.automodel.models.flux.plain_modules import is_plain_linear

        hooked = torch.nn.Linear(4, 4)
        hooked.register_forward_pre_hook(lambda module, args: None)
        assert is_plain_linear(torch.nn.Linear(4, 4, bias=False)) is False
        assert is_plain_linear(hooked) is False

    def test_fused_gelu_keeps_a_swapped_linear(self):
        from primus.backends.automodel.models.flux import mlp_gelu

        x = SimpleNamespace(is_cuda=True, dtype=torch.bfloat16)
        assert mlp_gelu.fusable(x, torch.nn.Linear(4, 4, dtype=torch.bfloat16), True) is True
        assert mlp_gelu.fusable(x, SwappedLinear(4, 4, dtype=torch.bfloat16), True) is False


class TestAffineFreeLayerNormOnly:
    def test_affine_free_layer_norm(self):
        from primus.backends.automodel.models.flux.plain_modules import (
            is_plain_layer_norm,
        )

        assert is_plain_layer_norm(torch.nn.LayerNorm(4, elementwise_affine=False)) is True

    def test_declines_weights_hooks_and_subclasses(self):
        from primus.backends.automodel.models.flux.plain_modules import (
            is_plain_layer_norm,
        )

        class AnotherLayerNorm(torch.nn.LayerNorm):
            pass

        hooked = torch.nn.LayerNorm(4, elementwise_affine=False)
        hooked.register_forward_hook(lambda module, args, out: None)
        assert is_plain_layer_norm(torch.nn.LayerNorm(4)) is False
        assert is_plain_layer_norm(hooked) is False
        assert is_plain_layer_norm(AnotherLayerNorm(4, elementwise_affine=False)) is False
        assert is_plain_layer_norm(torch.nn.RMSNorm(4, elementwise_affine=False)) is False
