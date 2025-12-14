###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from unittest.mock import Mock

import pytest

from primus.core.patches.context import PatchContext
from primus.core.patches.patch_registry import PatchRegistry, register_patch
from primus.core.patches.patch_runner import (
    _parse_enabled_patches_from_env,
    run_patches,
)


class TestPatchRunner:
    def setup_method(self):
        PatchRegistry.clear()

    def test_env_parsing(self, monkeypatch):
        monkeypatch.delenv("PRIMUS_PATCHES", raising=False)
        assert _parse_enabled_patches_from_env() is None

        monkeypatch.setenv("PRIMUS_PATCHES", "all")
        assert _parse_enabled_patches_from_env() is None

        monkeypatch.setenv("PRIMUS_PATCHES", "none")
        assert _parse_enabled_patches_from_env() == []

        monkeypatch.setenv("PRIMUS_PATCHES", "p1, p2")
        assert _parse_enabled_patches_from_env() == ["p1", "p2"]

    def test_run_patches_execution_and_context_fields(self, monkeypatch):
        # Avoid touching global logger in unit tests
        monkeypatch.setattr("primus.core.patches.patch_runner.log_rank_0", lambda *args, **kwargs: None)

        mock_handler = Mock()

        @register_patch("test.patch")
        def my_patch(ctx):
            mock_handler(ctx)

        extra = {"args": {"lr": 1e-4}}
        count = run_patches(
            backend="megatron",
            phase="setup",
            backend_version="0.8.0",
            primus_version="1.0.0",
            model_name="llama3",
            module_name="pre_trainer",
            platform="MI300X",
            extra=extra,
        )
        assert count == 1
        mock_handler.assert_called_once()

        # Verify context passed
        ctx_arg = mock_handler.call_args[0][0]
        assert isinstance(ctx_arg, PatchContext)
        assert ctx_arg.backend == "megatron"
        assert ctx_arg.phase == "setup"
        assert ctx_arg.backend_version == "0.8.0"
        assert ctx_arg.primus_version == "1.0.0"
        assert ctx_arg.model_name == "llama3"
        assert ctx_arg.module_name == "pre_trainer"
        assert ctx_arg.platform == "MI300X"
        assert ctx_arg.extra == extra

    def test_run_patches_priority_and_tiebreaker(self, monkeypatch):
        monkeypatch.setattr("primus.core.patches.patch_runner.log_rank_0", lambda *args, **kwargs: None)
        order = []

        @register_patch("p2", priority=10)
        def p2(ctx):
            order.append("p2")

        @register_patch("p1", priority=10)
        def p1(ctx):
            order.append("p1")

        @register_patch("p3", priority=5)
        def p3(ctx):
            order.append("p3")

        run_patches(backend="x", phase="y")
        # p3 (priority 5) first, then p1/p2 sorted by id
        assert order == ["p3", "p1", "p2"]

    def test_run_patches_filtering_by_backend_and_enabled_ids(self, monkeypatch):
        called = []

        @register_patch("p1", backend="megatron")
        def p1(ctx):
            called.append("p1")

        @register_patch("p2", backend="torchtitan")
        def p2(ctx):
            called.append("p2")

        # Backend filter only
        called.clear()
        monkeypatch.setattr("primus.core.patches.patch_runner.log_rank_0", lambda *args, **kwargs: None)
        count = run_patches(backend="megatron", phase="x")
        assert count == 1  # Only p1 runs
        assert called == ["p1"]

        # enabled_ids parameter wins over env var
        called.clear()
        monkeypatch.setenv("PRIMUS_PATCHES", "p2")  # would select p2 if used
        count = run_patches(backend="megatron", phase="x", enabled_ids=["p1"])
        assert count == 1
        assert called == ["p1"]

    def test_run_patches_filtering_by_env_enabled_ids(self, monkeypatch):
        called = []

        @register_patch("p1")
        def p1(ctx):
            called.append("p1")

        @register_patch("p2")
        def p2(ctx):
            called.append("p2")

        # "none" disables all
        called.clear()
        monkeypatch.setenv("PRIMUS_PATCHES", "none")
        monkeypatch.setattr("primus.core.patches.patch_runner.log_rank_0", lambda *args, **kwargs: None)
        count = run_patches(backend="x", phase="y")
        assert count == 0
        assert called == []

        # Select subset by IDs
        called.clear()
        monkeypatch.setenv("PRIMUS_PATCHES", "p2")
        count = run_patches(backend="x", phase="y")
        assert count == 1
        assert called == ["p2"]

        # "all" (or unset) enables all
        called.clear()
        monkeypatch.setenv("PRIMUS_PATCHES", "all")
        count = run_patches(backend="x", phase="y")
        assert count == 2
        assert sorted(called) == ["p1", "p2"]

    def test_dry_run(self, monkeypatch):
        monkeypatch.setattr("primus.core.patches.patch_runner.log_rank_0", lambda *args, **kwargs: None)
        mock_handler = Mock()

        @register_patch("p1")
        def p1(ctx):
            mock_handler()

        count = run_patches(backend="x", phase="y", dry_run=True)
        assert count == 1
        mock_handler.assert_not_called()

    def test_stop_on_error(self, monkeypatch):
        monkeypatch.setattr("primus.core.patches.patch_runner.log_rank_0", lambda *args, **kwargs: None)

        @register_patch("fail")
        def p1(ctx):
            raise ValueError("boom")

        # Default: continue (no exception)
        run_patches(backend="x", phase="y")

        # Stop on error
        with pytest.raises(ValueError, match="boom"):
            run_patches(backend="x", phase="y", stop_on_error=True)
