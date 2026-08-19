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
    """Verify TE FP4 is bypassed exactly when Primus-Turbo is enabled."""

    @staticmethod
    def _turbo_enabled_with(monkeypatch, have_turbo=True, **arg_fields):
        import megatron.training.global_vars as global_vars

        from primus.backends.megatron.core import fp4_utils

        monkeypatch.setattr(fp4_utils, "HAVE_TURBO", have_turbo)
        monkeypatch.setattr(global_vars, "get_args", lambda: SimpleNamespace(**arg_fields))

        return fp4_utils._primus_turbo_enabled()

    def test_enabled_when_primus_turbo_is_enabled(self, monkeypatch):
        assert self._turbo_enabled_with(monkeypatch, enable_primus_turbo=True)

    def test_disabled_when_primus_turbo_is_disabled(self, monkeypatch):
        assert not self._turbo_enabled_with(monkeypatch, enable_primus_turbo=False)

    def test_disabled_when_turbo_is_unavailable(self, monkeypatch):
        assert not self._turbo_enabled_with(monkeypatch, have_turbo=False, enable_primus_turbo=True)


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
