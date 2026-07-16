# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Tests for loss computation utilities.

These tests verify the correctness of loss computation functions,
ensuring they produce expected results and maintain backward compatibility.
"""

import torch

from tests.utils import skip_if_no_cuda

skip_if_no_cuda()

from primus.backends.megatron.training.diffusion.loss_computation import (
    compute_flow_matching_loss,
)
from tests.utils import PrimusUT


class TestFlowMatchingLoss(PrimusUT):
    """Tests for flow matching loss computation."""

    def test_basic_computation(self):
        """Test flow matching loss matches manual calculation."""
        prediction = torch.randn(2, 16, 64, 64)
        clean = torch.randn(2, 16, 64, 64)
        noise = torch.randn(2, 16, 64, 64)

        loss = compute_flow_matching_loss(prediction, clean, noise)

        # Manual calculation
        target = noise - clean
        expected = torch.nn.functional.mse_loss(prediction.float(), target.float())

        assert torch.allclose(loss, expected)
        assert loss.dim() == 0  # Scalar

    def test_loss_with_partial_mask(self):
        """Test loss computation with partial masking."""
        prediction = torch.randn(2, 16, 64, 64)
        clean = torch.randn(2, 16, 64, 64)
        noise = torch.randn(2, 16, 64, 64)

        # Create partial mask (half valid)
        loss_mask = torch.zeros(2, 16, 64, 64)
        loss_mask[:, :, :32, :] = 1.0  # First half valid

        loss_with_mask = compute_flow_matching_loss(prediction, clean, noise, loss_mask)

        # Should be different from unmasked loss
        loss_without_mask = compute_flow_matching_loss(prediction, clean, noise)

        assert not torch.allclose(loss_with_mask, loss_without_mask)
