###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for MoE expert ↔ communication overlap.

Tests the overlap forward construction and shared expert helper logic.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from primus.backends.megatron.core.transformer.moe.expert_comm_overlap import (
    _compute_shared_expert,
    make_overlapped_forward,
)


class TestComputeSharedExpert(unittest.TestCase):
    """Test the _compute_shared_expert helper function."""

    def test_no_shared_expert_returns_none(self):
        """When use_shared_expert is False, returns None."""
        moe_layer = MagicMock()
        moe_layer.use_shared_expert = False
        moe_layer.shared_expert_overlap = False

        import torch

        result = _compute_shared_expert(moe_layer, torch.randn(4, 16))
        self.assertIsNone(result)

    def test_shared_expert_overlap_returns_none(self):
        """When shared_expert_overlap is True, returns None (Megatron handles it)."""
        moe_layer = MagicMock()
        moe_layer.use_shared_expert = True
        moe_layer.shared_expert_overlap = True

        import torch

        result = _compute_shared_expert(moe_layer, torch.randn(4, 16))
        self.assertIsNone(result)

    def test_shared_expert_no_recompute(self):
        """When shared expert is enabled without recompute, calls shared_experts directly."""
        import torch

        moe_layer = MagicMock()
        moe_layer.use_shared_expert = True
        moe_layer.shared_expert_overlap = False
        moe_layer.shared_experts_recompute = False

        expected = torch.randn(4, 16)
        moe_layer.shared_experts.return_value = expected

        residual = torch.randn(4, 16)
        result = _compute_shared_expert(moe_layer, residual)

        moe_layer.shared_experts.assert_called_once_with(residual)
        self.assertTrue(torch.equal(result, expected))


class TestMakeOverlappedForward(unittest.TestCase):
    """Test that make_overlapped_forward produces a valid callable."""

    def test_returns_callable(self):
        """make_overlapped_forward should return a callable."""
        original = MagicMock()
        patched = make_overlapped_forward(original)
        self.assertTrue(callable(patched))

    def test_patched_forward_calls_router_dispatch_combine(self):
        """Patched forward should call router, dispatch, experts, combine."""
        import torch

        original = MagicMock()
        patched = make_overlapped_forward(original)

        # Build a mock MoE layer
        moe_layer = MagicMock()
        moe_layer.training = False
        moe_layer.moe_layer_recompute = False
        moe_layer.use_shared_expert = False
        moe_layer.shared_expert_overlap = False
        moe_layer.config.sequence_parallel = True

        # Mock attn_tp_group
        mock_group = MagicMock()
        mock_group.size.return_value = 1
        moe_layer.attn_tp_group = mock_group

        hidden = torch.randn(2, 4, 16)
        residual = hidden.clone()

        # Setup return values for the mock chain
        moe_layer.router_and_preprocess.return_value = (hidden, torch.ones(8), residual)
        moe_layer.dispatch.return_value = (hidden, torch.ones(8))
        moe_layer.token_dispatcher.dispatch_postprocess.return_value = (
            hidden.view(-1, 16),
            torch.tensor([4, 4]),
            torch.ones(8),
        )
        moe_layer.experts.return_value = (hidden.view(-1, 16), None)
        moe_layer.token_dispatcher.combine_preprocess.return_value = hidden.view(-1, 16)
        moe_layer.token_dispatcher.token_combine.return_value = hidden.view(-1, 16)
        moe_layer.token_dispatcher.combine_postprocess.return_value = hidden

        # Call patched forward
        output, mlp_bias = patched(moe_layer, hidden)

        # Verify call sequence
        moe_layer.router_and_preprocess.assert_called_once()
        moe_layer.dispatch.assert_called_once()
        moe_layer.token_dispatcher.dispatch_postprocess.assert_called_once()
        moe_layer.experts.assert_called_once()
        moe_layer.token_dispatcher.combine_preprocess.assert_called_once()
        moe_layer.token_dispatcher.token_combine.assert_called_once()
        moe_layer.token_dispatcher.combine_postprocess.assert_called_once()

        self.assertIsNone(mlp_bias)


if __name__ == "__main__":
    unittest.main()
