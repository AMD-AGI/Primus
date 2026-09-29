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


def _inputs(*shape):
    g = torch.Generator().manual_seed(0)
    return tuple(torch.randn(*shape, generator=g) for _ in range(3))


def _masked_mean(prediction, clean, noise, mask):
    """Mean of the squared error over the elements ``mask`` covers once broadcast."""
    per_element = (prediction.float() - (noise - clean).float()) ** 2
    covered = (
        mask.float().reshape(mask.shape + (1,) * (per_element.dim() - mask.dim())).expand_as(per_element)
    )
    return (per_element * covered).sum() / covered.sum()


class TestFlowMatchingLossMask(PrimusUT):
    """A mask averages over the loss elements it covers, whatever its shape."""

    def test_all_ones_sample_mask_matches_unmasked(self):
        prediction, clean, noise = _inputs(4, 16, 32, 32)

        masked = compute_flow_matching_loss(prediction, clean, noise, torch.ones(4))

        torch.testing.assert_close(masked, compute_flow_matching_loss(prediction, clean, noise))

    def test_sample_mask_excludes_padding(self):
        """A padded slot contributes nothing: the loss is the real samples' own loss."""
        prediction, clean, noise = _inputs(4, 16, 32, 32)

        masked = compute_flow_matching_loss(prediction, clean, noise, torch.tensor([1.0, 1.0, 0.0, 0.0]))

        torch.testing.assert_close(masked, compute_flow_matching_loss(prediction[:2], clean[:2], noise[:2]))

    def test_soft_sample_mask_is_a_weighted_element_mean(self):
        prediction, clean, noise = _inputs(4, 16, 8, 8)
        mask = torch.tensor([0.5, 1.0, 1.0, 0.25])

        masked = compute_flow_matching_loss(prediction, clean, noise, mask)

        torch.testing.assert_close(masked, _masked_mean(prediction, clean, noise, mask))

    def test_spatial_mask_counts_every_channel(self):
        prediction, clean, noise = _inputs(2, 16, 8, 8)
        mask = (torch.rand(2, 1, 8, 8, generator=torch.Generator().manual_seed(1)) > 0.3).float()

        masked = compute_flow_matching_loss(prediction, clean, noise, mask)

        torch.testing.assert_close(masked, _masked_mean(prediction, clean, noise, mask))

    def test_elementwise_mask_is_the_mean_over_kept_elements(self):
        prediction, clean, noise = _inputs(2, 16, 8, 8)
        mask = torch.zeros(2, 16, 8, 8)
        mask[:, :, :4, :] = 1.0

        masked = compute_flow_matching_loss(prediction, clean, noise, mask)

        torch.testing.assert_close(masked, _masked_mean(prediction, clean, noise, mask))

    def test_all_zero_mask_gives_zero_not_nan(self):
        """A micro-batch that is all padding must not put NaN into the gradients."""
        prediction, clean, noise = _inputs(2, 16, 8, 8)
        prediction.requires_grad_(True)

        masked = compute_flow_matching_loss(prediction, clean, noise, torch.zeros(2))
        masked.backward()

        assert masked.item() == 0.0
        assert torch.equal(prediction.grad, torch.zeros_like(prediction))

    def test_bf16_mask_counts_exactly(self):
        """The forward step casts the batch, mask included, to the compute dtype.

        3 * 16 * 33 * 33 kept elements is not representable in bf16, so counting
        in the mask's dtype would be off.
        """
        prediction, clean, noise = _inputs(4, 16, 33, 33)
        mask = torch.tensor([1.0, 1.0, 1.0, 0.0])

        masked = compute_flow_matching_loss(prediction, clean, noise, mask.bfloat16())

        torch.testing.assert_close(masked, compute_flow_matching_loss(prediction, clean, noise, mask))
