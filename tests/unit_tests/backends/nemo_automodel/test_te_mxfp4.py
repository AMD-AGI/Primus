###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the Transformer Engine native MXFP4 path.

WHAT THESE ARE DEFENDING:
  The autocast has to be entered per transformer block, not once at the
  transformer's forward, and the reason is an interaction with activation
  checkpointing: a non-reentrant checkpoint re-runs the block's forward during the
  backward pass, standalone, outside whatever context managers were active
  originally. A top-level autocast therefore does not cover the recompute, the
  two executions quantize differently, and the mismatch either raises or produces
  wrong gradients.

  That makes ``test_every_block_forward_is_wrapped`` and
  ``test_a_second_block_list_is_also_wrapped`` the load-bearing tests here. The
  second matters because wrapping only the first list found leaves the rest
  running in bf16 silently.

  The other quiet failure is a swap with no wrap: a bare TE Linear runs in bf16
  outside an autocast, so swapping without wrapping changes nothing while looking
  like it worked.

TE is faked throughout. What is under test is this module's decisions -- which
layers to swap, where to put the autocast, how to cope with TE's API spellings --
not TE's own behaviour, and requiring a TE-native image to check any of that
would mean never checking it.
"""

import sys
import types
from contextlib import contextmanager

import pytest

from primus.backends.nemo_automodel.quantization import _common, te_mxfp4_linear

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

requires_torch = pytest.mark.skipif(torch is None, reason="needs torch")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ("PRIMUS_TE_MXFP4", "PRIMUS_TURBO_FP4", "PRIMUS_TURBO_FP8", "NVTE_MXFP4_USE_HADAMARD"):
        monkeypatch.delenv(var, raising=False)


class Recipe:
    """Stands in for MXFP4BlockScaling, which is a plain settings object."""

    def __init__(self):
        self.use_hadamard = None


@pytest.fixture
def fake_te(monkeypatch):
    """Install a fake transformer_engine exposing the newer autocast spelling."""
    entered = []

    @contextmanager
    def autocast(enabled=True, recipe=None):
        entered.append(recipe)
        yield

    pytorch_mod = types.ModuleType("transformer_engine.pytorch")
    pytorch_mod.autocast = autocast
    if torch is not None:

        class TELinear(torch.nn.Linear):
            def __init__(self, in_features, out_features, bias=True, params_dtype=None, device=None):
                super().__init__(in_features, out_features, bias=bias, device=device, dtype=params_dtype)

        pytorch_mod.Linear = TELinear

    recipe_mod = types.ModuleType("transformer_engine.common.recipe")
    recipe_mod.MXFP4BlockScaling = Recipe
    common_mod = types.ModuleType("transformer_engine.common")
    common_mod.recipe = recipe_mod
    root = types.ModuleType("transformer_engine")
    root.pytorch = pytorch_mod
    root.common = common_mod

    for name, mod in (
        ("transformer_engine", root),
        ("transformer_engine.pytorch", pytorch_mod),
        ("transformer_engine.common", common_mod),
        ("transformer_engine.common.recipe", recipe_mod),
    ):
        monkeypatch.setitem(sys.modules, name, mod)

    pytorch_mod.entered = entered
    return pytorch_mod


class TestPrecedence:
    def test_te_mxfp4_has_the_highest_precedence(self):
        """Naming both a precision and an implementation is the most specific
        request, so it should not lose to a broader one."""
        by_name = {e.name: e for e in _common.registered_backends()}
        assert by_name[te_mxfp4_linear.BACKEND_NAME].precedence > by_name["turbo_mxfp4"].precedence
        assert by_name[te_mxfp4_linear.BACKEND_NAME].precedence > by_name["turbo_fp8"].precedence

    def test_it_wins_against_both_others(self, monkeypatch):
        monkeypatch.setenv("PRIMUS_TE_MXFP4", "1")
        monkeypatch.setenv("PRIMUS_TURBO_FP4", "1")
        monkeypatch.setenv("PRIMUS_TURBO_FP8", "1")
        assert _common.active_backend().name == te_mxfp4_linear.BACKEND_NAME
        assert _common.is_active("turbo_mxfp4") is False
        assert _common.is_active("turbo_fp8") is False

    def test_it_is_off_by_default(self):
        assert te_mxfp4_linear.is_enabled() is False


class TestRecipe:
    def test_hadamard_defaults_off(self, fake_te):
        assert te_mxfp4_linear.build_recipe().use_hadamard is False

    def test_hadamard_follows_the_env(self, fake_te, monkeypatch):
        """Set explicitly so the recipe object and the environment cannot
        disagree about whether the transform is applied."""
        monkeypatch.setenv("NVTE_MXFP4_USE_HADAMARD", "1")
        assert te_mxfp4_linear.build_recipe().use_hadamard is True


class TestAutocastApiDrift:
    """TE has spelled this three ways. Probed rather than version-sniffed,
    because the version number has not been a reliable guide."""

    def test_the_new_autocast_spelling_is_used_when_present(self, fake_te):
        recipe = Recipe()
        with te_mxfp4_linear.autocast_for(recipe):
            pass
        assert fake_te.entered == [recipe]

    def test_it_falls_back_to_fp8_autocast_with_recipe(self, monkeypatch, fake_te):
        calls = []

        @contextmanager
        def fp8_autocast(enabled=True, recipe=None):
            calls.append(("recipe", recipe))
            yield

        monkeypatch.delattr(fake_te, "autocast")
        fake_te.fp8_autocast = fp8_autocast
        recipe = Recipe()
        with te_mxfp4_linear.autocast_for(recipe):
            pass
        assert calls == [("recipe", recipe)]

    def test_it_falls_back_to_the_classic_fp8_recipe_keyword(self, monkeypatch, fake_te):
        calls = []

        @contextmanager
        def fp8_autocast(enabled=True, fp8_recipe=None):
            calls.append(("fp8_recipe", fp8_recipe))
            yield

        monkeypatch.delattr(fake_te, "autocast")
        fake_te.fp8_autocast = fp8_autocast
        recipe = Recipe()
        with te_mxfp4_linear.autocast_for(recipe):
            pass
        assert calls == [("fp8_recipe", recipe)]


@requires_torch
class TestBlockWrapping:
    """The heart of it. See the module docstring for why per-block."""

    def _model(self, **lists):
        model = torch.nn.Module()
        for attr, count in lists.items():
            setattr(model, attr, torch.nn.ModuleList([torch.nn.Linear(128, 128) for _ in range(count)]))
        return model

    def test_every_block_forward_is_wrapped(self, fake_te):
        model = self._model(blocks=4)
        description = te_mxfp4_linear.wrap_block_forwards(model, Recipe())
        assert "blocks[4]" in description
        assert all(getattr(b, "_primus_te_mxfp4_wrapped", False) for b in model.blocks)

    def test_the_wrapped_forward_actually_enters_the_autocast(self, fake_te):
        """Wrapping the attribute is not the same as the autocast being entered
        when the block runs."""
        model = self._model(blocks=2)
        recipe = Recipe()
        te_mxfp4_linear.wrap_block_forwards(model, recipe)
        model.blocks[0](torch.zeros(1, 128))
        assert fake_te.entered == [recipe]

    def test_the_top_level_forward_is_not_wrapped(self, fake_te):
        """A top-level autocast is exactly what does not survive the checkpoint
        recompute, so per-block wrapping must not also wrap the top."""
        model = self._model(blocks=2)
        te_mxfp4_linear.wrap_block_forwards(model, Recipe())
        assert getattr(model, "_primus_te_mxfp4_wrapped", False) is False

    def test_a_second_block_list_is_also_wrapped(self, fake_te):
        """Wrapping only the first list found leaves the rest running their TE
        Linears in bf16, silently."""
        model = self._model(transformer_blocks=19, single_transformer_blocks=38)
        description = te_mxfp4_linear.wrap_block_forwards(model, Recipe())
        assert "transformer_blocks[19]" in description
        assert "single_transformer_blocks[38]" in description
        assert all(
            getattr(b, "_primus_te_mxfp4_wrapped", False)
            for b in list(model.transformer_blocks) + list(model.single_transformer_blocks)
        )

    def test_wrapping_is_idempotent(self, fake_te):
        """A double wrap would enter the autocast twice per forward and count the
        blocks twice in the summary."""
        model = self._model(blocks=3)
        te_mxfp4_linear.wrap_block_forwards(model, Recipe())
        second = te_mxfp4_linear.wrap_block_forwards(model, Recipe())
        assert "blocks[0]" in second

    def test_an_empty_block_list_is_skipped(self, fake_te):
        model = self._model(blocks=0, transformer_blocks=2)
        description = te_mxfp4_linear.wrap_block_forwards(model, Recipe())
        assert "blocks[" not in description.replace("transformer_blocks[", "")
        assert "transformer_blocks[2]" in description

    def test_no_block_list_falls_back_and_warns(self, fake_te, caplog):
        """Correct but incompatible with activation checkpointing, so it has to
        say so rather than look like a success."""
        model = torch.nn.Module()
        model.proj = torch.nn.Linear(128, 128)
        with caplog.at_level("WARNING"):
            description = te_mxfp4_linear.wrap_block_forwards(model, Recipe())
        assert "whole module" in description
        assert any("activation checkpointing" in r.message for r in caplog.records)
        assert getattr(model, "_primus_te_mxfp4_wrapped", False) is True


@requires_torch
class TestReplaceLinears:
    def _model(self):
        model = torch.nn.Module()
        model.blocks = torch.nn.ModuleList([torch.nn.Linear(256, 256) for _ in range(2)])
        model.misaligned = torch.nn.Linear(100, 256)
        return model

    def test_a_swap_with_no_wrap_would_be_a_no_op(self, fake_te, monkeypatch, caplog):
        """The quiet failure: a bare TE Linear runs bf16 outside an autocast, so
        swapping without wrapping changes nothing while looking like it worked."""
        monkeypatch.setattr(
            "primus.backends.nemo_automodel.quantization._fp4_common." "is_fp4_training_safe_linear",
            lambda fqn, linear: linear.in_features % 128 == 0,
        )
        model = self._model()
        with caplog.at_level("INFO"):
            converted = te_mxfp4_linear.replace_linears(model, "toy")
        assert converted == 2
        joined = " ".join(r.getMessage() for r in caplog.records)
        assert "per-block" in joined, "the summary must report the autocast, not just the swap"

    def test_the_ineligible_linear_is_left_alone(self, fake_te, monkeypatch):
        monkeypatch.setattr(
            "primus.backends.nemo_automodel.quantization._fp4_common." "is_fp4_training_safe_linear",
            lambda fqn, linear: linear.in_features % 128 == 0,
        )
        model = self._model()
        te_mxfp4_linear.replace_linears(model, "toy")
        assert type(model.misaligned) is torch.nn.Linear

    def test_nothing_is_wrapped_when_nothing_is_swapped(self, fake_te, monkeypatch, caplog):
        monkeypatch.setattr(
            "primus.backends.nemo_automodel.quantization._fp4_common." "is_fp4_training_safe_linear",
            lambda fqn, linear: False,
        )
        model = self._model()
        with caplog.at_level("INFO"):
            assert te_mxfp4_linear.replace_linears(model, "toy") == 0
        joined = " ".join(r.getMessage() for r in caplog.records)
        assert "no Linear was swapped" in joined

    def test_weights_survive_the_swap(self, fake_te, monkeypatch):
        monkeypatch.setattr(
            "primus.backends.nemo_automodel.quantization._fp4_common." "is_fp4_training_safe_linear",
            lambda fqn, linear: linear.in_features % 128 == 0,
        )
        model = self._model()
        expected = model.blocks[0].weight.detach().clone()
        te_mxfp4_linear.replace_linears(model, "toy")
        assert torch.allclose(model.blocks[0].weight, expected)


class TestPatchRegistration:
    def _patches(self):
        import primus.backends.nemo_automodel.patches  # noqa: F401
        from primus.core.patches.patch_registry import PatchRegistry

        return {p.id: p for p in PatchRegistry.iter_patches(backend="nemo_automodel", phase="before_train")}

    def test_the_patch_is_registered(self):
        assert "nemo_automodel.quantization.te_mxfp4_linear" in self._patches()

    def test_it_is_gated_off_by_default(self):
        patch = self._patches()["nemo_automodel.quantization.te_mxfp4_linear"]
        assert patch.condition(None) is False

    def test_only_this_one_is_active_when_all_three_are_requested(self, monkeypatch):
        """All three rebind the same symbol, so more than one active condition
        would mean a later swap silently replaced an earlier one."""
        monkeypatch.setenv("PRIMUS_TE_MXFP4", "1")
        monkeypatch.setenv("PRIMUS_TURBO_FP4", "1")
        monkeypatch.setenv("PRIMUS_TURBO_FP8", "1")
        by_id = self._patches()
        active = [
            pid
            for pid in (
                "nemo_automodel.quantization.te_mxfp4_linear",
                "nemo_automodel.quantization.mxfp4_linear",
                "nemo_automodel.quantization.fp8_linear",
            )
            if by_id[pid].condition(None)
        ]
        assert active == ["nemo_automodel.quantization.te_mxfp4_linear"]

    def test_it_precedes_the_model_strategies(self):
        """The autocast has to be wrapped before the strategies wrap the blocks
        in checkpoints."""
        by_id = self._patches()
        for strategy in (
            "nemo_automodel.models.wan.parallelize",
            "nemo_automodel.models.flux.parallelize",
        ):
            assert by_id["nemo_automodel.quantization.te_mxfp4_linear"].priority < by_id[strategy].priority


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
