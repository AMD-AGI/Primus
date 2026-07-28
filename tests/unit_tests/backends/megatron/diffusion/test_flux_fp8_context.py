# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Unit tests for FluxModel.get_fp8_context() behavior.

Verifies that the local spec FP8 path returns nullcontext (no global state),
while the non-local path delegates to Megatron's get_fp8_context utility.
"""

import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tests.utils import skip_if_no_cuda

skip_if_no_cuda()

from primus.backends.megatron.core.models.diffusion.flux.model import Flux


class TestFluxFP8Context:
    """Tests for Flux.get_fp8_context() without constructing a full model."""

    def test_get_fp8_context_local_returns_nullcontext(self):
        """Local spec should return nullcontext to avoid global FP8 state."""
        mock_model = MagicMock(spec=Flux)
        mock_model.config = SimpleNamespace(
            transformer_impl="local",
            fp8="e4m3",
        )

        ctx = Flux.get_fp8_context(mock_model)
        assert isinstance(ctx, contextlib.nullcontext)

    def test_get_fp8_context_non_local_delegates(self):
        """Non-local spec should delegate to megatron.core.fp8_utils.get_fp8_context."""
        mock_model = MagicMock(spec=Flux)
        mock_model.config = SimpleNamespace(
            transformer_impl="transformer_engine",
            fp8="e4m3",
        )

        sentinel = contextlib.nullcontext()
        with patch(
            "megatron.core.fp8_utils.get_fp8_context",
            return_value=sentinel,
        ) as mock_get:
            ctx = Flux.get_fp8_context(mock_model)
            mock_get.assert_called_once_with(mock_model.config)
            assert ctx is sentinel
