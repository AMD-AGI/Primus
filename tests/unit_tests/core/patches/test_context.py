###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for PatchContext helper functions.
"""

from types import SimpleNamespace

import pytest

from primus.core.patches.context import PatchContext, get_args, get_param


class TestGetArgs:
    """Tests for get_args helper function."""

    def test_get_args_success(self):
        """Test get_args with valid context."""
        params = SimpleNamespace(foo="bar", num=42)
        module_config = SimpleNamespace(params=params)
        ctx = PatchContext(
            backend="test",
            phase="setup",
            extra={"module_config": module_config},
        )

        result = get_args(ctx)
        assert result is params
        assert result.foo == "bar"
        assert result.num == 42

    def test_get_args_missing_module_config(self):
        """Test get_args raises when module_config is missing."""
        ctx = PatchContext(backend="test", phase="setup", extra={})

        with pytest.raises(AssertionError, match="module_config is required"):
            get_args(ctx)

    def test_get_args_missing_params(self):
        """Test get_args raises when params is missing."""
        module_config = SimpleNamespace()  # No params attribute
        ctx = PatchContext(
            backend="test",
            phase="setup",
            extra={"module_config": module_config},
        )

        with pytest.raises(AssertionError, match="params is required"):
            get_args(ctx)


class TestGetParam:
    """Tests for get_param helper function."""

    def test_get_param_simple_path(self):
        """Test get_param with a simple one-level path."""
        params = SimpleNamespace(foo="bar", num=42)
        module_config = SimpleNamespace(params=params)
        ctx = PatchContext(
            backend="test",
            phase="setup",
            extra={"module_config": module_config},
        )

        assert get_param(ctx, "foo") == "bar"
        assert get_param(ctx, "num") == 42

    def test_get_param_nested_path(self):
        """Test get_param with nested dot-separated path."""
        params = SimpleNamespace(
            primus_turbo=SimpleNamespace(
                enable_fp8=True,
                enable_embedding_autocast=False,
            ),
            metrics=SimpleNamespace(
                enable_wandb=True,
            ),
        )
        module_config = SimpleNamespace(params=params)
        ctx = PatchContext(
            backend="test",
            phase="setup",
            extra={"module_config": module_config},
        )

        assert get_param(ctx, "primus_turbo.enable_fp8") is True
        assert get_param(ctx, "primus_turbo.enable_embedding_autocast") is False
        assert get_param(ctx, "metrics.enable_wandb") is True

    def test_get_param_deep_nested_path(self):
        """Test get_param with deeply nested path."""
        params = SimpleNamespace(
            level1=SimpleNamespace(
                level2=SimpleNamespace(
                    level3=SimpleNamespace(value="deep"),
                ),
            ),
        )
        module_config = SimpleNamespace(params=params)
        ctx = PatchContext(
            backend="test",
            phase="setup",
            extra={"module_config": module_config},
        )

        assert get_param(ctx, "level1.level2.level3.value") == "deep"

    def test_get_param_missing_path_returns_default(self):
        """Test get_param returns default when path doesn't exist."""
        params = SimpleNamespace(foo="bar")
        module_config = SimpleNamespace(params=params)
        ctx = PatchContext(
            backend="test",
            phase="setup",
            extra={"module_config": module_config},
        )

        # Missing top-level attribute
        assert get_param(ctx, "missing") is None
        assert get_param(ctx, "missing", "default") == "default"

        # Missing nested attribute
        assert get_param(ctx, "foo.bar") is None
        assert get_param(ctx, "foo.bar", False) is False

    def test_get_param_partial_path_returns_default(self):
        """Test get_param returns default when partial path doesn't exist."""
        params = SimpleNamespace(
            primus_turbo=SimpleNamespace(enable_fp8=True),
        )
        module_config = SimpleNamespace(params=params)
        ctx = PatchContext(
            backend="test",
            phase="setup",
            extra={"module_config": module_config},
        )

        # First part exists, but second part doesn't
        assert get_param(ctx, "primus_turbo.missing_field") is None
        assert get_param(ctx, "primus_turbo.missing_field", 123) == 123

    def test_get_param_missing_module_config_returns_default(self):
        """Test get_param returns default when module_config is missing."""
        ctx = PatchContext(backend="test", phase="setup", extra={})

        assert get_param(ctx, "foo.bar") is None
        assert get_param(ctx, "foo.bar", "fallback") == "fallback"

    def test_get_param_missing_params_returns_default(self):
        """Test get_param returns default when params is missing."""
        module_config = SimpleNamespace()  # No params attribute
        ctx = PatchContext(
            backend="test",
            phase="setup",
            extra={"module_config": module_config},
        )

        assert get_param(ctx, "foo.bar") is None
        assert get_param(ctx, "foo.bar", False) is False

    def test_get_param_with_zero_value(self):
        """Test get_param correctly returns 0 and doesn't confuse it with None."""
        params = SimpleNamespace(count=0, flag=False)
        module_config = SimpleNamespace(params=params)
        ctx = PatchContext(
            backend="test",
            phase="setup",
            extra={"module_config": module_config},
        )

        assert get_param(ctx, "count") == 0
        assert get_param(ctx, "flag") is False

    def test_get_param_in_condition_lambda(self):
        """Test get_param works in condition lambdas (real-world usage)."""
        params = SimpleNamespace(
            primus_turbo=SimpleNamespace(enable_embedding_autocast=True),
        )
        module_config = SimpleNamespace(params=params)
        ctx = PatchContext(
            backend="torchtitan",
            phase="setup",
            extra={"module_config": module_config},
        )

        # Simulate condition lambda usage
        condition = lambda ctx: get_param(ctx, "primus_turbo.enable_embedding_autocast", False)
        assert condition(ctx) is True

        # With missing field
        condition2 = lambda ctx: get_param(ctx, "primus_turbo.missing", False)
        assert condition2(ctx) is False
