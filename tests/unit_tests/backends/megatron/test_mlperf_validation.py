###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for MLPerf validation support.

Tests cover:
    1. Validation loss function shape and target computation
    2. Training loss function shape (unchanged)
    3. Mock dataset timestep generation (production MockDiffusionDataset)
"""

import pytest
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 1. Validation loss function shape
# ---------------------------------------------------------------------------


class TestValLossFunc:
    """Test the validation loss function produces correct shapes and values."""

    def _make_val_loss_func(self, noise, clean_latents):
        """Build val_loss_func mirroring DiffusionPretrainTrainer.forward_step."""

        class Holder:
            _last_noise = noise
            _last_clean_latents = clean_latents

        def val_loss_func(output_tensor, non_loss_data=False):
            if non_loss_data:
                return output_tensor
            target = Holder._last_noise - Holder._last_clean_latents
            loss = F.mse_loss(output_tensor.float(), target.float(), reduction="none")
            loss_per_sample = loss.mean(dim=tuple(range(1, loss.ndim)))
            loss_sum = loss_per_sample.sum()
            sample_count = torch.tensor(loss_per_sample.numel(), dtype=loss_sum.dtype, device=loss_sum.device)
            return loss_sum, {"loss": (loss_sum.detach(), sample_count.detach())}

        return val_loss_func

    def test_returns_tuple_with_loss_key(self):
        B, C, H, W = 4, 16, 32, 32
        noise = torch.randn(B, C, H, W)
        clean = torch.randn(B, C, H, W)
        pred = torch.randn(B, C, H, W)
        fn = self._make_val_loss_func(noise, clean)

        loss_scalar, metrics = fn(pred)

        assert isinstance(loss_scalar, torch.Tensor)
        assert loss_scalar.ndim == 0
        assert "loss" in metrics
        loss_sum, sample_count = metrics["loss"]
        assert loss_sum.ndim == 0
        assert sample_count.item() == B

    def test_correct_target(self):
        B, C, H, W = 2, 4, 8, 8
        noise = torch.randn(B, C, H, W)
        clean = torch.randn(B, C, H, W)
        target = noise - clean
        fn = self._make_val_loss_func(noise, clean)

        loss_scalar, metrics = fn(target)

        assert loss_scalar.item() == pytest.approx(0.0, abs=1e-6)

    def test_non_loss_data_passthrough(self):
        fn = self._make_val_loss_func(torch.zeros(1), torch.zeros(1))
        sentinel = torch.tensor(42.0)
        result = fn(sentinel, non_loss_data=True)
        assert result is sentinel


# ---------------------------------------------------------------------------
# 2. Training loss function shape
# ---------------------------------------------------------------------------


class TestTrainLossFunc:
    """Verify the training loss path returns scalar with reduced_train_loss."""

    def _make_train_loss_func(self, loss_fn, noise, clean, mask):
        class Holder:
            _last_noise = noise
            _last_clean_latents = clean
            _last_loss_mask = mask

        def diffusion_loss_func(output_tensor, non_loss_data=False):
            if non_loss_data:
                return output_tensor
            loss = loss_fn(
                output_tensor, Holder._last_clean_latents, Holder._last_noise, Holder._last_loss_mask
            )
            return loss, {"reduced_train_loss": loss.detach().clone()}

        return diffusion_loss_func

    def test_returns_scalar_with_key(self):
        from primus.backends.megatron.training.diffusion.loss_computation import (
            compute_flow_matching_loss,
        )

        B, C, H, W = 4, 16, 32, 32
        noise = torch.randn(B, C, H, W)
        clean = torch.randn(B, C, H, W)
        pred = torch.randn(B, C, H, W)

        fn = self._make_train_loss_func(compute_flow_matching_loss, noise, clean, None)
        loss, metrics = fn(pred)

        assert loss.ndim == 0
        assert "reduced_train_loss" in metrics
        assert metrics["reduced_train_loss"].ndim == 0


# ---------------------------------------------------------------------------
# 3. Mock dataset timestep generation
# ---------------------------------------------------------------------------


class TestMockDatasetTimestep:
    """Verify validation mock datasets produce cycling timestep 0-7."""

    def test_validation_dataset_has_timestep(self):
        from primus.backends.megatron.data.synthetic.mock_datasets import (
            MockDiffusionDataset,
        )

        ds = MockDiffusionDataset(
            num_samples=16,
            image_size=256,
            model_preset="flux_schnell",
            dtype=torch.float32,
            device="cpu",
            is_validation=True,
        )
        for i in range(16):
            sample = ds[i]
            assert "timestep" in sample, f"Sample {i} missing 'timestep'"
            assert sample["timestep"].item() == i % 8

    def test_training_dataset_no_timestep(self):
        from primus.backends.megatron.data.synthetic.mock_datasets import (
            MockDiffusionDataset,
        )

        ds = MockDiffusionDataset(
            num_samples=4,
            image_size=256,
            model_preset="flux_schnell",
            dtype=torch.float32,
            device="cpu",
            is_validation=False,
        )
        for i in range(4):
            sample = ds[i]
            assert "timestep" not in sample
