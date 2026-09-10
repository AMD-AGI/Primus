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
    compute_weighted_flow_matching_loss,
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


class TestWeightedFlowMatchingLoss(PrimusUT):
    """Tests for the per-timestep weighted flow matching loss.

    The weighted helper exists alongside the unweighted one rather than
    replacing it, so the load-bearing property is that weight=1 reproduces
    the unweighted objective exactly. If that drifts, every Wan run silently
    optimizes a different loss than the Flux runs it is compared against.
    """

    @staticmethod
    def _fixture():
        torch.manual_seed(0)
        prediction = torch.randn(2, 16, 8, 32, 32)
        clean = torch.randn(2, 16, 8, 32, 32)
        noise = torch.randn(2, 16, 8, 32, 32)
        # The unweighted helper derives this target internally.
        return prediction, clean, noise, noise - clean

    def test_unit_weight_recovers_unweighted_loss(self):
        """weight=1 per sample is exactly the unweighted objective."""
        prediction, clean, noise, target = self._fixture()

        weighted = compute_weighted_flow_matching_loss(prediction, target, torch.ones(2))
        unweighted = compute_flow_matching_loss(prediction, clean, noise)

        assert torch.allclose(weighted, unweighted)
        assert weighted.dim() == 0

    def test_scalar_unit_weight_recovers_unweighted_loss(self):
        """A 0-dim weight broadcasts without the [batch] reshape path."""
        prediction, clean, noise, target = self._fixture()

        weighted = compute_weighted_flow_matching_loss(prediction, target, torch.tensor(1.0))
        unweighted = compute_flow_matching_loss(prediction, clean, noise)

        assert torch.allclose(weighted, unweighted)

    def test_unit_weight_recovers_unweighted_loss_under_mask(self):
        """The equivalence also holds on the masked reduction path."""
        prediction, clean, noise, target = self._fixture()
        loss_mask = torch.tensor([1.0, 0.0])

        weighted = compute_weighted_flow_matching_loss(prediction, target, torch.ones(2), loss_mask)
        unweighted = compute_flow_matching_loss(prediction, clean, noise, loss_mask)

        assert torch.allclose(weighted, unweighted)

    def test_uniform_weight_scales_the_loss(self):
        """A constant weight scales the squared error linearly."""
        prediction, _, _, target = self._fixture()
        batch = prediction.shape[0]

        base = compute_weighted_flow_matching_loss(prediction, target, torch.ones(batch))
        scaled = compute_weighted_flow_matching_loss(prediction, target, torch.full((batch,), 2.0))

        assert torch.allclose(scaled, base * 2.0)

    def test_per_sample_weight_broadcasts_over_latent_dims(self):
        """A [batch] weight applies per sample, not per element."""
        prediction, _, _, target = self._fixture()
        weight = torch.tensor([0.0, 1.0])

        actual = compute_weighted_flow_matching_loss(prediction, target, weight)

        # Zeroing the first sample leaves the second sample's error, averaged
        # over the whole tensor -- so half the second sample's own mean.
        second = ((target[1].float() - prediction[1].float()) ** 2).mean()
        assert torch.allclose(actual, second / 2.0)

    def test_caller_supplied_target_is_not_rederived(self):
        """The target comes from the caller, unlike the unweighted helper."""
        prediction, clean, noise, _ = self._fixture()
        batch = prediction.shape[0]

        # A target that is deliberately not noise - clean.
        alt_target = torch.zeros_like(prediction)

        with_alt = compute_weighted_flow_matching_loss(prediction, alt_target, torch.ones(batch))
        with_derived = compute_weighted_flow_matching_loss(prediction, noise - clean, torch.ones(batch))

        assert not torch.allclose(with_alt, with_derived)
        assert torch.allclose(with_alt, (prediction.float() ** 2).mean())
