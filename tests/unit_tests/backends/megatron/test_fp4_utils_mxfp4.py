# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Unit tests for fp4_utils.py MXFP4 recipe and context manager changes.

Tests recipe error handling and gradient stochastic-rounding configuration.
"""

from types import SimpleNamespace

import pytest

from tests.utils import PrimusUT


class TestPrimusTurboFP4Selection:
    """Verify TE FP4 is bypassed only for explicit Turbo FP4 autocast."""

    def test_requires_turbo_fp4_autocast(self, monkeypatch):
        import megatron.training.global_vars as global_vars

        from primus.backends.megatron.core import fp4_utils

        monkeypatch.setattr(fp4_utils, "HAVE_TURBO", True)
        monkeypatch.setattr(
            global_vars,
            "get_args",
            lambda: SimpleNamespace(enable_primus_turbo=True),
        )

        assert not fp4_utils._primus_turbo_enabled()

    def test_enabled_when_both_flags_are_set(self, monkeypatch):
        import megatron.training.global_vars as global_vars

        from primus.backends.megatron.core import fp4_utils

        monkeypatch.setattr(fp4_utils, "HAVE_TURBO", True)
        monkeypatch.setattr(
            global_vars,
            "get_args",
            lambda: SimpleNamespace(enable_primus_turbo=True),
        )

        assert fp4_utils._primus_turbo_enabled()


class TestMXFP4GradientStochasticRounding:
    """Verify Megatron and diffusion configs resolve the SR option correctly."""

    def test_explicit_config_value_takes_precedence(self, monkeypatch):
        import megatron.training.global_vars as global_vars

        from primus.backends.megatron.core.fp4_utils import _mxfp4_gradient_sr_enabled

        def unexpected_get_args():
            raise AssertionError("global args should not be read for an explicit config value")

        monkeypatch.setattr(global_vars, "get_args", unexpected_get_args)

        assert _mxfp4_gradient_sr_enabled(SimpleNamespace(mxfp4_gradient_stochastic_rounding=True))
        assert not _mxfp4_gradient_sr_enabled(SimpleNamespace(mxfp4_gradient_stochastic_rounding=False))

    def test_megatron_config_falls_back_to_global_args(self, monkeypatch):
        import megatron.training.global_vars as global_vars

        from primus.backends.megatron.core.fp4_utils import _mxfp4_gradient_sr_enabled

        args = SimpleNamespace(mxfp4_gradient_stochastic_rounding=True)
        monkeypatch.setattr(global_vars, "get_args", lambda: args)

        assert _mxfp4_gradient_sr_enabled(SimpleNamespace())

    def test_unavailable_global_args_default_to_disabled(self, monkeypatch):
        import megatron.training.global_vars as global_vars

        from primus.backends.megatron.core.fp4_utils import _mxfp4_gradient_sr_enabled

        def unavailable_get_args():
            raise RuntimeError("global args are not initialized")

        monkeypatch.setattr(global_vars, "get_args", unavailable_get_args)

        assert not _mxfp4_gradient_sr_enabled(SimpleNamespace())


class TestGetFp4RecipeMXFP4(PrimusUT):
    """Verify get_fp4_recipe returns correct recipe objects for MXFP4."""

    @pytest.fixture(autouse=True)
    def setup_parallel(self, init_parallel_state):
        pass

    def test_unsupported_recipe_produces_error(self):
        pytest.importorskip("transformer_engine")

        from primus.backends.megatron.core.fp4_utils import get_fp4_recipe

        config = SimpleNamespace(fp4_recipe="nonexistent_recipe")
        result = get_fp4_recipe(config)

        if isinstance(result, tuple):
            recipe, reason = result
            assert recipe is None, "Unsupported recipe should return None"
            assert (
                "Unsupported" in reason or "unsupported" in reason.lower()
            ), f"Expected 'Unsupported' in reason, got: {reason}"
        else:
            pytest.fail("HAVE_TE-only branch should raise ValueError for unsupported recipe")
