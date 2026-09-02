###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the diffusion profiler wrapper, its env parsing, and the
registration of this backend's first two patches.

The profiler itself needs a GPU and a live recipe to exercise, so what is worth
testing here is everything around it that can fail silently: the env gate not
recognising a value the user wrote, and a patch that is present in the tree but
never actually registered. Both produce the same symptom -- the feature simply
does nothing, with no error -- which is the failure mode most likely to survive
review.

No torch and no nemo_automodel required: torch is imported inside install(),
which is not called here.
"""

import pytest

from primus.backends.nemo_automodel._env import current_rank, env_flag, env_int, env_str
from primus.core.patches.patch_registry import PatchRegistry


class TestEnvHelpers:
    @pytest.mark.parametrize("raw", ["1", "true", "True", "TRUE", "yes", "on", " on "])
    def test_truthy_spellings_are_accepted(self, monkeypatch, raw):
        """Rejecting a spelling silently disables a feature, so be generous."""
        monkeypatch.setenv("PRIMUS_TEST_FLAG", raw)
        assert env_flag("PRIMUS_TEST_FLAG") is True

    @pytest.mark.parametrize("raw", ["0", "false", "no", "off", "", "  ", "banana"])
    def test_other_values_are_false(self, monkeypatch, raw):
        monkeypatch.setenv("PRIMUS_TEST_FLAG", raw)
        assert env_flag("PRIMUS_TEST_FLAG") is False

    def test_unset_returns_default(self, monkeypatch):
        monkeypatch.delenv("PRIMUS_TEST_FLAG", raising=False)
        assert env_flag("PRIMUS_TEST_FLAG") is False
        assert env_flag("PRIMUS_TEST_FLAG", True) is True

    def test_env_int_parses_and_falls_back(self, monkeypatch):
        monkeypatch.setenv("PRIMUS_TEST_INT", "7")
        assert env_int("PRIMUS_TEST_INT", 3) == 7
        # A typo in a diagnostic knob must not take down a training run.
        monkeypatch.setenv("PRIMUS_TEST_INT", "seven")
        assert env_int("PRIMUS_TEST_INT", 3) == 3
        monkeypatch.delenv("PRIMUS_TEST_INT")
        assert env_int("PRIMUS_TEST_INT", 3) == 3

    def test_env_str_treats_empty_as_unset(self, monkeypatch):
        monkeypatch.setenv("PRIMUS_TEST_STR", "   ")
        assert env_str("PRIMUS_TEST_STR", "fallback") == "fallback"
        monkeypatch.setenv("PRIMUS_TEST_STR", " value ")
        assert env_str("PRIMUS_TEST_STR", "fallback") == "value"

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


class TestProfilerGate:
    def test_disabled_by_default(self, monkeypatch):
        from primus.backends.nemo_automodel.profiling import torch_profiler

        monkeypatch.delenv("PRIMUS_DIFFUSION_PROFILE", raising=False)
        assert torch_profiler.is_enabled() is False

    def test_enabled_by_env(self, monkeypatch):
        from primus.backends.nemo_automodel.profiling import torch_profiler

        monkeypatch.setenv("PRIMUS_DIFFUSION_PROFILE", "1")
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

    def test_fsdp2_reshard_is_registered(self):
        patch = self._patch("nemo_automodel.distributed.fsdp2_reshard")
        assert patch is not None, "fsdp2_reshard patch was not discovered"
        assert patch.description

    def test_profiler_is_registered(self):
        patch = self._patch("nemo_automodel.profiling.torch_profiler")
        assert patch is not None, "torch_profiler patch was not discovered"

    def test_reshard_is_unconditional(self):
        """It repairs a value the user already set, so it has nothing to gate on."""
        patch = self._patch("nemo_automodel.distributed.fsdp2_reshard")
        assert patch.condition is None

    def test_profiler_is_gated(self, monkeypatch):
        patch = self._patch("nemo_automodel.profiling.torch_profiler")
        assert patch.condition is not None

        monkeypatch.delenv("PRIMUS_DIFFUSION_PROFILE", raising=False)
        assert patch.condition(None) is False
        monkeypatch.setenv("PRIMUS_DIFFUSION_PROFILE", "1")
        assert patch.condition(None) is True

    def test_reshard_runs_before_the_profiler(self):
        """The profiler wraps the train loop; the repair must already be applied."""
        reshard = self._patch("nemo_automodel.distributed.fsdp2_reshard")
        profiler = self._patch("nemo_automodel.profiling.torch_profiler")
        assert reshard.priority < profiler.priority


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
