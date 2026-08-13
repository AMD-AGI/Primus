# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Unit tests for FluxModel.get_fp8_context() behavior.

Verifies that the local spec path returns nullcontext (no global state), while
the non-local (TransformerEngine) path delegates to the appropriate Megatron
quantization context: FP8 via megatron.core.fp8_utils, FP4 via
megatron.core.fp4_utils, with FP8 taking precedence when both are set.
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

    def test_get_fp8_context_local_with_fp4_returns_nullcontext(self):
        """Local spec must stay nullcontext even when fp4 is set (per-module FP4)."""
        mock_model = MagicMock(spec=Flux)
        mock_model.config = SimpleNamespace(
            transformer_impl="local",
            fp8=None,
            fp4="mxfp4",
        )

        ctx = Flux.get_fp8_context(mock_model)
        assert isinstance(ctx, contextlib.nullcontext)

    def test_get_fp8_context_non_local_fp4_delegates(self):
        """Non-local fp4-only run should delegate to megatron.core.fp4_utils.get_fp4_context."""
        mock_model = MagicMock(spec=Flux)
        mock_model.config = SimpleNamespace(
            transformer_impl="transformer_engine",
            fp8=None,
            fp4="mxfp4",
        )

        sentinel = contextlib.nullcontext()
        with patch(
            "megatron.core.fp4_utils.get_fp4_context",
            return_value=sentinel,
        ) as mock_get:
            ctx = Flux.get_fp8_context(mock_model)
            mock_get.assert_called_once_with(mock_model.config)
            assert ctx is sentinel

    def test_get_fp8_context_fp8_takes_precedence_over_fp4(self):
        """When both fp8 and fp4 are set, FP8 wins and FP4 context is not built."""
        mock_model = MagicMock(spec=Flux)
        mock_model.config = SimpleNamespace(
            transformer_impl="transformer_engine",
            fp8="e4m3",
            fp4="mxfp4",
        )

        fp8_sentinel = contextlib.nullcontext()
        with patch(
            "megatron.core.fp8_utils.get_fp8_context",
            return_value=fp8_sentinel,
        ) as mock_fp8, patch(
            "megatron.core.fp4_utils.get_fp4_context",
        ) as mock_fp4:
            ctx = Flux.get_fp8_context(mock_model)
            mock_fp8.assert_called_once_with(mock_model.config)
            mock_fp4.assert_not_called()
            assert ctx is fp8_sentinel

    def test_get_fp8_context_no_quantization_returns_nullcontext(self):
        """Non-local with neither fp8 nor fp4 set should return nullcontext."""
        mock_model = MagicMock(spec=Flux)
        mock_model.config = SimpleNamespace(
            transformer_impl="transformer_engine",
            fp8=None,
            fp4=None,
        )

        ctx = Flux.get_fp8_context(mock_model)
        assert isinstance(ctx, contextlib.nullcontext)
