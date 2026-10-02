###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the backend's settings, the diffusion profiler wrapper, and the
registration of this backend's first two patches.

The profiler itself needs a GPU and a live recipe to exercise, so what is worth
testing here is everything around it that can fail silently: a setting not
recognising a value the user wrote, a ``primus_*`` section leaking into the
AutoModel config, and a patch that is present in the tree but never actually
registered.

No torch and no nemo_automodel required: torch is imported inside install(),
which is not called here.
"""

from types import SimpleNamespace

import pytest

from primus.backends.nemo_automodel import options
from primus.backends.nemo_automodel.argument_builder import strip_primus_keys
from primus.backends.nemo_automodel.patches._conditions import transformer_is
from primus.backends.nemo_automodel.profiling.torch_profiler import current_rank
from primus.core.patches import PatchContext
from primus.core.patches.patch_registry import PatchRegistry


class TestOptions:
    def test_reads_primus_sections_from_a_namespace(self):
        options.load(SimpleNamespace(primus_profiler=SimpleNamespace(enabled=True, wait=5), model=None))
        assert options.flag("primus_profiler.enabled") is True
        assert options.integer("primus_profiler.wait", 3) == 5

    def test_other_sections_are_not_settings(self):
        options.load({"model": {"enabled": True}})
        assert options.get("model.enabled") is None

    def test_unset_returns_the_default(self):
        assert options.flag("primus_profiler.enabled") is False
        assert options.flag("primus_profiler.record_shapes", True) is True
        assert options.integer("primus_profiler.wait", 3) == 3
        assert options.string("primus_profiler.tag", "run") == "run"

    def test_load_replaces_earlier_settings(self):
        options.load({"primus_profiler": {"enabled": True}})
        options.load({})
        assert options.flag("primus_profiler.enabled") is False

    @pytest.mark.parametrize("raw", [True, "1", "true", "True", "yes", "on", " on "])
    def test_true_spellings(self, set_option, raw):
        """A value from a CLI override can arrive as a string."""
        set_option("primus_profiler.enabled", raw)
        assert options.flag("primus_profiler.enabled") is True

    @pytest.mark.parametrize("raw", [False, "0", "false", "no", "off", ""])
    def test_false_spellings(self, set_option, raw):
        set_option("primus_profiler.enabled", raw)
        assert options.flag("primus_profiler.enabled", True) is False

    def test_a_malformed_flag_raises(self, set_option):
        """Falling back would quietly run something other than what was asked for."""
        set_option("primus_profiler.enabled", "banana")
        with pytest.raises(ValueError, match="primus_profiler.enabled"):
            options.flag("primus_profiler.enabled")

    @pytest.mark.parametrize("raw", ["seven", True])
    def test_a_malformed_integer_raises(self, set_option, raw):
        set_option("primus_profiler.wait", raw)
        with pytest.raises(ValueError, match="primus_profiler.wait"):
            options.integer("primus_profiler.wait", 3)

    def test_a_section_must_be_a_mapping(self):
        with pytest.raises(ValueError, match="primus_profiler"):
            options.load({"primus_profiler": True})

    def test_primus_sections_never_reach_automodel(self):
        cleaned = strip_primus_keys({"model": {}, "primus_turbo": {"fp8_linear": True}, "framework": "x"})
        assert cleaned == {"model": {}}

    def test_current_rank_prefers_launcher_env(self, monkeypatch):
        """Must work before process-group init, which is when patches run."""
        monkeypatch.setenv("RANK", "3")
        assert current_rank() == 3
        monkeypatch.delenv("RANK")
        monkeypatch.setenv("LOCAL_RANK", "2")
        assert current_rank() == 2

    def test_current_rank_defaults_to_zero(self, monkeypatch):
        for key in ("RANK", "LOCAL_RANK", "OMPI_COMM_WORLD_RANK"):
            monkeypatch.delenv(key, raising=False)
        assert current_rank() == 0


class TestTransformerIs:
    @staticmethod
    def _ctx(transformer_cls):
        spec = SimpleNamespace(transformer_cls=transformer_cls) if transformer_cls else None
        params = SimpleNamespace(model=SimpleNamespace(pipeline_spec=spec))
        return PatchContext(
            backend="nemo_automodel",
            phase="before_train",
            extra={"module_config": SimpleNamespace(params=params)},
        )

    def test_matches_the_configured_class(self):
        assert transformer_is("FluxTransformer2DModel")(self._ctx("FluxTransformer2DModel")) is True

    def test_declines_a_different_class(self):
        assert transformer_is("FluxTransformer2DModel")(self._ctx("WanTransformer3DModel")) is False

    def test_applies_when_no_class_is_named(self):
        """A from_pretrained fine-tune names none; skipping a needed repair is the worse error."""
        assert transformer_is("WanTransformer3DModel")(self._ctx(None)) is True


class TestProfilerGate:
    def test_disabled_by_default(self):
        from primus.backends.nemo_automodel.profiling import torch_profiler

        assert torch_profiler.is_enabled() is False

    def test_enabled_by_config(self, set_option):
        from primus.backends.nemo_automodel.profiling import torch_profiler

        set_option("primus_profiler.enabled", True)
        assert torch_profiler.is_enabled() is True

    def test_importing_the_module_does_not_require_torch(self):
        """install() imports torch lazily; importing the module must not.

        If this regresses, the patch *condition* would start needing torch, which
        would break config-only tooling and this test suite.
        """
        import importlib
        import sys

        sys.modules.pop("primus.backends.nemo_automodel.profiling.torch_profiler", None)
        mod = importlib.import_module("primus.backends.nemo_automodel.profiling.torch_profiler")
        assert not any(
            line.strip().startswith("import torch") for line in open(mod.__file__).read().splitlines()[:60]
        ), "torch must stay inside install(), not at module import time"


class TestPatchesAreRegistered:
    """A patch file that exists but never registers looks identical to one that works."""

    @pytest.fixture(autouse=True)
    def load_patches(self):
        import primus.backends.nemo_automodel.patches  # noqa: F401

    def _patch(self, patch_id):
        for p in PatchRegistry.iter_patches(backend="nemo_automodel", phase="before_train"):
            if p.id == patch_id:
                return p
        return None

    def test_ddp_activation_checkpointing_is_registered(self):
        patch = self._patch("nemo_automodel.distributed.ddp_activation_checkpointing")
        assert patch is not None, "ddp_activation_checkpointing patch was not discovered"
        assert patch.description

    def test_profiler_is_registered(self):
        patch = self._patch("nemo_automodel.profiling.torch_profiler")
        assert patch is not None, "torch_profiler patch was not discovered"

    def test_ddp_repair_is_unconditional(self):
        """It repairs a value the user already set, so it has nothing to gate on."""
        patch = self._patch("nemo_automodel.distributed.ddp_activation_checkpointing")
        assert patch.condition is None

    def test_profiler_is_gated(self, set_option):
        patch = self._patch("nemo_automodel.profiling.torch_profiler")
        assert patch.condition is not None

        assert patch.condition(None) is False
        set_option("primus_profiler.enabled", True)
        assert patch.condition(None) is True

    def test_ddp_repair_runs_before_the_profiler(self):
        """The profiler wraps the train loop; the repair must already be applied."""
        repair = self._patch("nemo_automodel.distributed.ddp_activation_checkpointing")
        profiler = self._patch("nemo_automodel.profiling.torch_profiler")
        assert repair.priority < profiler.priority


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
