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

No torch and no AutoModel required: torch is imported inside install(),
which is not called here.
"""

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from primus.backends.automodel import options
from primus.backends.automodel.argument_builder import strip_primus_keys
from primus.backends.automodel.patches._conditions import transformer_is
from primus.backends.automodel.profiling.torch_profiler import (
    current_rank,
    resolve_output_dir,
)
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

    def test_undeclared_names_misspelled_keys_and_sections(self, monkeypatch):
        """A typo reads as unset, which turns its feature off without a word."""
        monkeypatch.setattr(options, "_declared", {})
        options.declare("primus_area", "feature", "level")
        options.load(
            {
                "primus_area": {"feature": True, "featrue": True},
                "primus_aera": {"feature": True},
                "model": {"featrue": True},
            }
        )
        assert options.undeclared() == ["primus_aera", "primus_area.featrue"]

    def test_declared_keys_are_not_reported(self, monkeypatch):
        monkeypatch.setattr(options, "_declared", {})
        options.declare("primus_area", "feature")
        options.declare("primus_area", "level")
        options.load({"primus_area": {"feature": True, "level": 2}})
        assert options.undeclared() == []

    def test_only_primus_sections_can_be_declared(self):
        with pytest.raises(ValueError, match="model"):
            options.declare("model", "enabled")

    def test_current_rank_prefers_launcher_env(self, monkeypatch):
        """Must work before process-group init, which is when patches run."""
        monkeypatch.setenv("RANK", "3")
        assert current_rank() == 3
        monkeypatch.delenv("RANK")
        monkeypatch.setenv("LOCAL_RANK", "2")
        assert current_rank() == 2

    def test_current_rank_defaults_to_zero(self, monkeypatch):
        for key in ("RANK", "LOCAL_RANK"):
            monkeypatch.delenv(key, raising=False)
        assert current_rank() == 0


class TestProfilerOutputDir:
    def test_a_relative_dir_goes_under_the_experiment(self):
        assert resolve_output_dir("diffusion_profile", "/ws/g/u/exp") == "/ws/g/u/exp/diffusion_profile"

    def test_an_absolute_dir_is_used_as_given(self):
        assert resolve_output_dir("/data/traces", "/ws/g/u/exp") == "/data/traces"

    def test_without_an_experiment_a_relative_dir_stays_relative(self):
        assert resolve_output_dir("diffusion_profile", None) == "diffusion_profile"

    def test_the_patch_passes_the_experiment_directory(self, monkeypatch):
        from primus.backends.automodel.patches import torch_profiler_patches
        from primus.backends.automodel.profiling import torch_profiler

        seen = {}
        monkeypatch.setattr(
            torch_profiler, "install", lambda exp_root=None: seen.setdefault("exp_root", exp_root)
        )
        ctx = PatchContext(
            backend="automodel",
            phase="before_train",
            extra={"primus_config": SimpleNamespace(exp_root_path="/ws/g/u/exp")},
        )

        torch_profiler_patches.apply(ctx)

        assert seen == {"exp_root": "/ws/g/u/exp"}


class TestTransformerIs:
    @staticmethod
    def _ctx(transformer_cls):
        spec = SimpleNamespace(transformer_cls=transformer_cls) if transformer_cls else None
        params = SimpleNamespace(model=SimpleNamespace(pipeline_spec=spec))
        return PatchContext(
            backend="automodel",
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
        from primus.backends.automodel.profiling import torch_profiler

        assert torch_profiler.is_enabled() is False

    def test_enabled_by_config(self, set_option):
        from primus.backends.automodel.profiling import torch_profiler

        set_option("primus_profiler.enabled", True)
        assert torch_profiler.is_enabled() is True

    def test_importing_the_module_does_not_require_torch(self):
        """install() imports torch lazily; importing the module must not.

        If this regresses, the patch *condition* would start needing torch, which
        would break config-only tooling and this test suite.
        """
        import importlib
        import sys

        sys.modules.pop("primus.backends.automodel.profiling.torch_profiler", None)
        mod = importlib.import_module("primus.backends.automodel.profiling.torch_profiler")
        assert not any(
            line.strip().startswith("import torch") for line in open(mod.__file__).read().splitlines()[:60]
        ), "torch must stay inside install(), not at module import time"


class TestPatchesAreRegistered:
    """A patch file that exists but never registers looks identical to one that works."""

    @pytest.fixture(autouse=True)
    def load_patches(self):
        import primus.backends.automodel.patches  # noqa: F401

    def _patch(self, patch_id):
        for p in PatchRegistry.iter_patches(backend="automodel", phase="before_train"):
            if p.id == patch_id:
                return p
        return None

    def test_ddp_activation_checkpointing_is_registered(self):
        patch = self._patch("automodel.distributed.ddp_activation_checkpointing")
        assert patch is not None, "ddp_activation_checkpointing patch was not discovered"
        assert patch.description

    def test_profiler_is_registered(self):
        patch = self._patch("automodel.profiling.torch_profiler")
        assert patch is not None, "torch_profiler patch was not discovered"

    @staticmethod
    def _ctx(**sections):
        params = SimpleNamespace(**{name: SimpleNamespace(**keys) for name, keys in sections.items()})
        return PatchContext(
            backend="automodel",
            phase="before_train",
            extra={"module_config": SimpleNamespace(params=params)},
        )

    def test_ddp_repair_applies_when_ddp_asks_for_checkpointing(self):
        patch = self._patch("automodel.distributed.ddp_activation_checkpointing")
        assert patch.condition(self._ctx(ddp={"activation_checkpointing": True})) is True

    @pytest.mark.parametrize(
        "sections",
        [
            {"fsdp": {"activation_checkpointing": True}},
            {"ddp": {"activation_checkpointing": False}},
            {"ddp": {"gradient_as_bucket_view": True}},
            {},
        ],
        ids=["fsdp", "ddp-without-checkpointing", "ddp-unset", "no-distributed-section"],
    )
    def test_ddp_repair_is_skipped_with_nothing_to_forward(self, sections):
        """FSDP2 reads the flag itself; the run log should not claim the repair applied."""
        patch = self._patch("automodel.distributed.ddp_activation_checkpointing")
        assert patch.condition(self._ctx(**sections)) is False
        assert patch.condition(None) is False

    def test_profiler_is_gated(self, set_option):
        patch = self._patch("automodel.profiling.torch_profiler")
        assert patch.condition is not None

        assert patch.condition(None) is False
        set_option("primus_profiler.enabled", True)
        assert patch.condition(None) is True

    def test_ddp_repair_runs_before_the_profiler(self):
        """The profiler wraps the train loop; the repair must already be applied."""
        repair = self._patch("automodel.distributed.ddp_activation_checkpointing")
        profiler = self._patch("automodel.profiling.torch_profiler")
        assert repair.priority < profiler.priority


_REPO = Path(__file__).resolve().parents[4]


def _settings_the_backend_reads():
    """Every ``primus_<area>.<key>`` the backend's code reads, by its string literal."""
    read = set()
    for path in (_REPO / "primus" / "backends" / "automodel").rglob("*.py"):
        text = path.read_text()
        read.update(re.findall(r'"(primus_[a-z0-9]+\.[a-z0-9_]+)"', text))
        section = re.search(r'^_SECTION = "(primus_[a-z0-9]+)"$', text, re.M)
        if section:
            read.update(f"{section[1]}.{key}" for key in re.findall(r'f"\{_SECTION\}\.([a-z0-9_]+)"', text))
    read.discard("primus_my_area.my_feature")  # the example in the patches package docstring
    return read


def _declared_settings():
    import primus.backends.automodel.patches  # noqa: F401

    return {f"{section}.{key}" for section, keys in options._declared.items() for key in keys}


def _primus_settings_in(node):
    if isinstance(node, dict):
        for name, value in node.items():
            if str(name).startswith("primus_") and isinstance(value, dict):
                yield from (f"{name}.{key}" for key in value)
            else:
                yield from _primus_settings_in(value)
    elif isinstance(node, list):
        for item in node:
            yield from _primus_settings_in(item)


class TestSettingsAreDeclared:
    """The trainer's warning about unknown settings is only as good as the declarations."""

    def test_every_setting_the_backend_reads_is_declared(self):
        missing = _settings_the_backend_reads() - _declared_settings()
        assert not missing, f"declare these in the feature's patch module: {sorted(missing)}"

    def test_every_declared_setting_is_read(self):
        unread = _declared_settings() - _settings_the_backend_reads()
        assert not unread, f"declared, but no code reads them: {sorted(unread)}"

    def test_shipped_configs_set_only_declared_settings(self):
        import yaml

        configs = [
            *(_REPO / "examples" / "automodel").rglob("*.yaml"),
            *(_REPO / "primus" / "configs" / "modules" / "automodel").rglob("*.yaml"),
            *(_REPO / "primus" / "configs" / "models" / "automodel").rglob("*.yaml"),
        ]
        assert configs
        declared = _declared_settings()
        unknown = {
            f"{path.relative_to(_REPO)}: {setting}"
            for path in configs
            for setting in _primus_settings_in(yaml.safe_load(path.read_text()))
            if setting not in declared
        }
        assert not unknown, sorted(unknown)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
