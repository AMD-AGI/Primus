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
  that a low-precision swap replaced, so they must take only a stock ``torch.nn.Linear``. The
  linear1 fold makes the non-deterministic attention override's kernel call itself, so it must
  decline without that override, with FP8 attention, under context parallelism and with a linear
  swap, and its patch must read the last two from the config.
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


class SwappedLinear(torch.nn.Linear):
    """What a low-precision swap puts in place of a linear: a subclass with its own forward."""


class TestStockLinearOnly:
    def test_stock_linear(self):
        from primus.backends.automodel.models.flux.plain_linear import is_plain_linear

        assert is_plain_linear(torch.nn.Linear(4, 4)) is True

    def test_declines_a_swapped_linear(self):
        from primus.backends.automodel.models.flux.plain_linear import is_plain_linear

        assert is_plain_linear(SwappedLinear(4, 4)) is False

    def test_declines_without_bias_or_with_hooks(self):
        from primus.backends.automodel.models.flux.plain_linear import is_plain_linear

        hooked = torch.nn.Linear(4, 4)
        hooked.register_forward_pre_hook(lambda module, args: None)
        assert is_plain_linear(torch.nn.Linear(4, 4, bias=False)) is False
        assert is_plain_linear(hooked) is False

    def test_fused_gelu_keeps_a_swapped_linear(self):
        from primus.backends.automodel.models.flux import mlp_gelu

        x = SimpleNamespace(is_cuda=True, dtype=torch.bfloat16)
        assert mlp_gelu.fusable(x, torch.nn.Linear(4, 4, dtype=torch.bfloat16), True) is True
        assert mlp_gelu.fusable(x, SwappedLinear(4, 4, dtype=torch.bfloat16), True) is False


def _run_ctx(**sections):
    params = SimpleNamespace(**{name: SimpleNamespace(**keys) for name, keys in sections.items()})
    return PatchContext(
        backend="automodel",
        phase="before_train",
        extra={"module_config": SimpleNamespace(params=params)},
    )


class TestFoldPrerequisites:
    def test_needs_the_nondeterministic_attention_override(self, set_option):
        from primus.backends.automodel.models.flux import linear1_fold

        reason = linear1_fold.unsupported_reason(context_parallel=False, linear_swap=False)
        assert "nondeterministic_attention" in reason
        set_option("primus_turbo.nondeterministic_attention", True)
        assert linear1_fold.unsupported_reason(context_parallel=False, linear_swap=False) is None

    def test_declines_fp8_attention(self, set_option):
        from primus.backends.automodel.models.flux import linear1_fold

        set_option("primus_turbo.nondeterministic_attention", True)
        set_option("primus_turbo.fp8_attention", True)
        reason = linear1_fold.unsupported_reason(context_parallel=False, linear_swap=False)
        assert "fp8_attention" in reason

    @pytest.mark.parametrize(
        "context_parallel,linear_swap,named",
        [(True, False, "context parallelism"), (False, True, "transformer_engine_linear")],
    )
    def test_declines_the_run(self, context_parallel, linear_swap, named, set_option):
        from primus.backends.automodel.models.flux import linear1_fold

        set_option("primus_turbo.nondeterministic_attention", True)
        reason = linear1_fold.unsupported_reason(context_parallel=context_parallel, linear_swap=linear_swap)
        assert named in reason
        assert linear1_fold.install(context_parallel=context_parallel, linear_swap=linear_swap) is False

    @pytest.mark.parametrize(
        "sections,expected",
        [
            ({}, {"context_parallel": False, "linear_swap": False}),
            ({"fsdp": {"cp_size": 1}}, {"context_parallel": False, "linear_swap": False}),
            ({"fsdp": {"cp_size": "2"}}, {"context_parallel": True, "linear_swap": False}),
            (
                {"model": {"transformer_engine_linear": True}},
                {"context_parallel": False, "linear_swap": True},
            ),
            (
                {"model": {"transformer_engine_linear": "false"}},
                {"context_parallel": False, "linear_swap": False},
            ),
        ],
    )
    def test_patch_reads_the_run_from_the_config(self, sections, expected, monkeypatch):
        from primus.backends.automodel.models.flux import linear1_fold
        from primus.backends.automodel.patches import flux_linear1_fold_patches

        seen = {}
        monkeypatch.setattr(linear1_fold, "install", lambda **kwargs: seen.update(kwargs))
        flux_linear1_fold_patches.apply(_run_ctx(**sections))
        assert seen == expected
