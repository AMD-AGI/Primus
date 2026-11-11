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
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from primus.core.patches import PatchContext  # noqa: E402

FLUX_PATCHES = [
    ("flux_mlp_gelu_patches", "primus_perf.flux_fused_gelu"),
    ("flux_qk_norm_rope_patches", "primus_perf.flux_fused_qk_norm_rope"),
    ("flux_block_outputs_patches", "primus_perf.flux_unalias_block_outputs"),
    ("flux_last_block_patches", "primus_perf.flux_last_block_image_only"),
    ("flux_adaln_flydsl_patches", "primus_perf.flux_fused_adaln_flydsl"),
    ("flux_linear1_fold_patches", "primus_perf.flux_fold_linear1"),
    ("flux_adaln_dual_flydsl_patches", "primus_perf.flux_fused_adaln_dual_flydsl"),
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
