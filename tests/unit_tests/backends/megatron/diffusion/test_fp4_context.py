# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Unit tests for get_fp4_context() backend selection in
primus.backends.megatron.core.fp4_utils.

Focus: the fp4_use_native_te_autocast override. When set, get_fp4_context must
take TransformerEngine's native fp8_autocast branch (TE MXFP4BlockScaling ->
AITER a4w4) even with Primus-Turbo enabled; when unset (with Primus-Turbo
enabled) it must take the primus_turbo_fp4_autocast branch.
"""

import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.utils import skip_if_no_cuda

skip_if_no_cuda()

from primus.backends.megatron.core import fp4_utils

# The two-branch get_fp4_context (Primus-Turbo vs TE-native) only exists when
# both TransformerEngine and Primus-Turbo are importable; otherwise the fallback
# definitions have no branch to exercise.
pytestmark = pytest.mark.skipif(
    not (fp4_utils.HAVE_TE and fp4_utils.HAVE_TURBO),
    reason="requires both TransformerEngine and Primus-Turbo installed",
)


def _fp4_config(**overrides):
    """Minimal config that reaches the Turbo-vs-TE-native branch of get_fp4_context."""
    config = SimpleNamespace(
        fp4="mxfp4",
        fp4_recipe="mxfp4",
        transformer_impl="transformer_engine",
        first_last_layers_bf16=False,
        num_layers=10,
        tp_only_amax_red=False,
        fp4_use_native_te_autocast=False,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


class TestGetFp4ContextBackendSelection:
    """get_fp4_context() Turbo-vs-TE-native branch selection."""

    def test_native_te_autocast_forces_te_branch_over_turbo(self):
        """fp4_use_native_te_autocast=True -> TE fp8_autocast, even with Turbo enabled."""
        config = _fp4_config(fp4_use_native_te_autocast=True)
        te_sentinel = contextlib.nullcontext()

        with patch.object(
            fp4_utils, "_primus_turbo_enabled", return_value=True
        ), patch.object(
            fp4_utils, "get_fp4_recipe", return_value=(MagicMock(name="recipe"), "")
        ), patch(
            "megatron.core.parallel_state.model_parallel_is_initialized",
            return_value=False,
        ), patch(
            "transformer_engine.pytorch.fp8_autocast", return_value=te_sentinel
        ) as mock_te, patch(
            "primus.backends.megatron.core.extensions.primus_turbo.primus_turbo_fp4_autocast"
        ) as mock_turbo:
            ctx = fp4_utils.get_fp4_context(config)

        mock_te.assert_called_once()
        mock_turbo.assert_not_called()
        assert ctx is te_sentinel

    def test_turbo_branch_taken_when_native_te_autocast_disabled(self):
        """fp4_use_native_te_autocast=False + Turbo enabled -> primus_turbo_fp4_autocast."""
        config = _fp4_config(fp4_use_native_te_autocast=False)
        turbo_sentinel = contextlib.nullcontext()

        with patch.object(
            fp4_utils, "_primus_turbo_enabled", return_value=True
        ), patch.object(
            fp4_utils, "get_fp4_recipe", return_value=(MagicMock(name="recipe"), "")
        ), patch.object(
            fp4_utils, "get_fp4_quant_config", return_value=(MagicMock(name="quant"), "")
        ), patch(
            "megatron.core.parallel_state.model_parallel_is_initialized",
            return_value=False,
        ), patch(
            "transformer_engine.pytorch.fp8_autocast"
        ) as mock_te, patch(
            "primus.backends.megatron.core.extensions.primus_turbo.primus_turbo_fp4_autocast",
            return_value=turbo_sentinel,
        ) as mock_turbo:
            ctx = fp4_utils.get_fp4_context(config)

        mock_turbo.assert_called_once()
        mock_te.assert_not_called()
        assert ctx is turbo_sentinel
