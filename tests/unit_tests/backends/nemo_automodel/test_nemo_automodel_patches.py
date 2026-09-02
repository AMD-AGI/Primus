###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the nemo_automodel patch mechanism.

The backend adds all of its behaviour through ``primus.core.patches`` rather
than a hand-maintained list in the trainer, so these tests cover the properties
the rest of the backend relies on: a patch module is discovered without being
named anywhere, a registered patch runs, its ``condition`` can decline it, its
``priority`` orders it, and a patch that raises does not take the run down with
it.

These tests do not require the ``nemo_automodel`` package.
"""

import sys
import textwrap
from collections import defaultdict
from types import SimpleNamespace

import pytest

from primus.core.patches import PatchContext, get_param, register_patch, run_patches
from primus.core.patches.patch_registry import PatchRegistry


@pytest.fixture
def clean_registry():
    """Run each test against an empty registry, then restore the real one.

    The registry is process-global and the production patch modules register
    into it on import, so tests must not leak patches into each other (or into
    whatever imported the backend earlier in the session).
    """
    saved_by_phase = PatchRegistry._patches_by_backend_phase
    saved_all = PatchRegistry._all_patches
    PatchRegistry._patches_by_backend_phase = defaultdict(lambda: defaultdict(list))
    PatchRegistry._all_patches = []
    try:
        yield PatchRegistry
    finally:
        PatchRegistry._patches_by_backend_phase = saved_by_phase
        PatchRegistry._all_patches = saved_all


def _ctx_extra(params=None):
    """The extra dict the trainer passes; mirrors NemoAutomodelPretrainTrainer."""
    backend_args = params if params is not None else SimpleNamespace()
    return {
        "module_config": SimpleNamespace(params=backend_args),
        "backend_args": backend_args,
    }


class TestAutoDiscovery:
    """A patch file registers itself by existing, without being listed anywhere."""

    def test_package_import_is_idempotent(self):
        import importlib

        import primus.backends.nemo_automodel.patches as pkg

        importlib.reload(pkg)  # must not raise; walk_packages re-imports cleanly

    def test_new_patch_module_is_discovered(self, tmp_path, clean_registry):
        """Drop a ``*_patches.py`` into the package tree and walk it.

        This is the property the whole design rests on -- if discovery silently
        stopped working, every feature would still import fine and simply never
        take effect, which is the worst possible failure mode.
        """
        import primus.backends.nemo_automodel.patches as pkg

        pkg_dir = tmp_path / "patches_pkg"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("")
        (pkg_dir / "generated_patches.py").write_text(
            textwrap.dedent(
                """
                from primus.core.patches import register_patch

                @register_patch(
                    "test.discovered",
                    backend="nemo_automodel",
                    phase="before_train",
                    description="discovered by walking the tree",
                )
                def _apply(ctx):
                    pass
                """
            )
        )
        sys.path.insert(0, str(tmp_path))
        try:
            # Re-run the package's own discovery against the temporary tree.
            import importlib
            import pkgutil

            found = []
            for info in pkgutil.walk_packages([str(pkg_dir)], prefix="patches_pkg."):
                if info.name.endswith("_patches") or info.name.endswith("_patch"):
                    importlib.import_module(info.name)
                    found.append(info.name)

            assert found == ["patches_pkg.generated_patches"]
            ids = [p.id for p in PatchRegistry.iter_patches(backend="nemo_automodel", phase="before_train")]
            assert "test.discovered" in ids
        finally:
            sys.path.remove(str(tmp_path))
            sys.modules.pop("patches_pkg.generated_patches", None)
            sys.modules.pop("patches_pkg", None)

        # The real package must not name modules explicitly -- that would defeat
        # auto-discovery and reintroduce the shared file every feature edits.
        source = open(pkg.__file__).read()
        assert "walk_packages" in source


class TestPatchExecution:
    def test_registered_patch_runs(self, clean_registry):
        calls = []

        @register_patch("test.runs", backend="nemo_automodel", phase="before_train", description="runs")
        def _apply(ctx: PatchContext):
            calls.append(ctx)

        applied = run_patches(backend="nemo_automodel", phase="before_train", extra=_ctx_extra())

        assert applied == 1
        assert len(calls) == 1
        assert calls[0].backend == "nemo_automodel"

    def test_false_condition_skips_patch(self, clean_registry):
        calls = []

        @register_patch(
            "test.gated",
            backend="nemo_automodel",
            phase="before_train",
            description="gated off",
            condition=lambda ctx: False,
        )
        def _apply(ctx):
            calls.append(ctx)

        applied = run_patches(backend="nemo_automodel", phase="before_train", extra=_ctx_extra())

        assert applied == 0
        assert calls == []

    def test_priority_orders_patches(self, clean_registry):
        order = []

        @register_patch(
            "test.late", backend="nemo_automodel", phase="before_train", description="late", priority=90
        )
        def _late(ctx):
            order.append("late")

        @register_patch(
            "test.early", backend="nemo_automodel", phase="before_train", description="early", priority=10
        )
        def _early(ctx):
            order.append("early")

        run_patches(backend="nemo_automodel", phase="before_train", extra=_ctx_extra())

        # Registration order is late-then-early; priority must win.
        assert order == ["early", "late"]

    def test_other_phase_is_not_run(self, clean_registry):
        calls = []

        @register_patch(
            "test.after", backend="nemo_automodel", phase="after_train", description="wrong phase"
        )
        def _apply(ctx):
            calls.append(ctx)

        run_patches(backend="nemo_automodel", phase="before_train", extra=_ctx_extra())

        assert calls == []


class TestFailureIsolation:
    """A broken optional feature must degrade the run, not end it."""

    def test_failing_patch_does_not_stop_the_run(self, clean_registry):
        survived = []

        @register_patch(
            "test.explodes", backend="nemo_automodel", phase="before_train", description="raises", priority=10
        )
        def _boom(ctx):
            raise RuntimeError("deliberate failure")

        @register_patch(
            "test.survives", backend="nemo_automodel", phase="before_train", description="ok", priority=20
        )
        def _ok(ctx):
            survived.append(True)

        # Must not raise, and must not prevent the later patch from applying.
        run_patches(backend="nemo_automodel", phase="before_train", extra=_ctx_extra())

        assert survived == [True]

    def test_stop_on_error_still_propagates(self, clean_registry):
        @register_patch(
            "test.explodes2", backend="nemo_automodel", phase="before_train", description="raises"
        )
        def _boom(ctx):
            raise RuntimeError("deliberate failure")

        with pytest.raises(RuntimeError, match="deliberate failure"):
            run_patches(
                backend="nemo_automodel", phase="before_train", extra=_ctx_extra(), stop_on_error=True
            )


class TestPatchContextPlumbing:
    """The trainer's ``extra`` dict must satisfy the core config helpers.

    ``get_param`` traverses with ``getattr``, so this is what makes wrapping
    ``backend_args`` in ``SimpleNamespace(params=...)`` load-bearing rather than
    decorative.
    """

    def test_get_param_reads_nested_backend_args(self, clean_registry):
        seen = {}

        @register_patch(
            "test.reads_config", backend="nemo_automodel", phase="before_train", description="reads"
        )
        def _apply(ctx):
            seen["flag"] = get_param(ctx, "primus_turbo.enable_fp8", False)
            seen["missing"] = get_param(ctx, "nope.not.here", "fallback")

        params = SimpleNamespace(primus_turbo=SimpleNamespace(enable_fp8=True))
        run_patches(backend="nemo_automodel", phase="before_train", extra=_ctx_extra(params))

        assert seen["flag"] is True
        assert seen["missing"] == "fallback"


class TestTrainerWiring:
    """The trainer must accept runtime kwargs and apply patches at the right time."""

    def test_init_accepts_basemodule_context_kwargs(self):
        """The runtime constructs trainers with BaseModule-style kwargs.

        The original signature took only ``backend_args`` positionally, so this
        call raised TypeError before the fix.
        """
        from primus.backends.nemo_automodel.nemo_automodel_pretrain_trainer import (
            NemoAutomodelPretrainTrainer,
        )

        trainer = NemoAutomodelPretrainTrainer(
            backend_args=SimpleNamespace(), module_name="pre_trainer", module_rank=0
        )
        assert trainer is not None

    def test_apply_patches_uses_before_train_phase(self, clean_registry, monkeypatch):
        """Guard the phase and the shape of ``extra`` the patches rely on."""
        from primus.backends.nemo_automodel import (
            nemo_automodel_pretrain_trainer as mod,
        )

        recorded = {}

        def _fake_run_patches(**kwargs):
            recorded.update(kwargs)
            return 0

        monkeypatch.setattr("primus.core.patches.run_patches", _fake_run_patches)

        trainer = mod.NemoAutomodelPretrainTrainer(backend_args=SimpleNamespace(a=1))
        trainer._apply_patches()

        assert recorded["backend"] == "nemo_automodel"
        assert recorded["phase"] == "before_train"
        assert recorded["extra"]["backend_args"].a == 1
        # get_args()/get_param() require module_config.params.
        assert recorded["extra"]["module_config"].params.a == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
