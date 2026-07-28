# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Schema/shape/reproducibility contract tests for the synthetic Flux mock datasets.

These guard the sample contract (keys, tensor shapes, seed reproducibility) that
the Flux forward step depends on for mock-data training runs. Relocated here from
tests/integration_tests/... because they are CPU-only unit tests of the synthetic
dataset providers and do not exercise an end-to-end TaskEncoder/dataloader path.
"""

import pytest
import torch

from primus.backends.megatron.data.synthetic import (
    MockFluxDataset,
    MockFluxSchnellDataset,
    PreGeneratedMockFluxSchnellDataset,
)
from primus.backends.megatron.data.synthetic_dataset_provider import (
    SyntheticDatasetProvider,
)
from tests.unit_tests.backends.megatron.diffusion.helpers import assert_tensor_shape
from tests.utils import PrimusUT


class TestMockFluxDataset(PrimusUT):
    """Contract tests for MockFluxDataset (Flux dev)."""

    def test_mock_dataset_sample_structure(self):
        dataset = MockFluxDataset(num_samples=10, seed=42)
        sample = dataset[0]

        expected_keys = {
            "latents",
            "prompt_embeds",
            "pooled_prompt_embeds",
            "img_ids",
            "txt_ids",
            "caption",
        }
        assert set(sample.keys()) == expected_keys

    def test_mock_dataset_sample_shapes(self):
        dataset = MockFluxDataset(num_samples=10, image_size=1024, seed=42)
        sample = dataset[0]

        # Latents: (C, H, W) -- 1024/8 = 128
        assert_tensor_shape(sample["latents"], (16, 128, 128), "latents")
        # T5 embeddings: (S, D)
        assert_tensor_shape(sample["prompt_embeds"], (512, 4096), "t5_embeddings")
        # CLIP pooled: (D,)
        assert_tensor_shape(sample["pooled_prompt_embeds"], (768,), "clip_pooled")
        # Image IDs: (N, 3) where N = (H/2) * (W/2)
        expected_img_ids_len = (128 // 2) * (128 // 2)
        assert_tensor_shape(sample["img_ids"], (expected_img_ids_len, 3), "img_ids")
        # Text IDs: (S, 3)
        assert_tensor_shape(sample["txt_ids"], (512, 3), "txt_ids")

    def test_mock_dataset_reproducibility(self):
        dataset1 = MockFluxDataset(num_samples=10, seed=42)
        dataset2 = MockFluxDataset(num_samples=10, seed=42)

        sample1 = dataset1[0]
        sample2 = dataset2[0]

        assert torch.allclose(sample1["latents"], sample2["latents"])
        assert torch.allclose(sample1["prompt_embeds"], sample2["prompt_embeds"])
        assert torch.allclose(sample1["pooled_prompt_embeds"], sample2["pooled_prompt_embeds"])

    def test_position_ids_format(self):
        dataset = MockFluxDataset(num_samples=5, seed=42)
        sample = dataset[0]

        img_ids = sample["img_ids"]
        txt_ids = sample["txt_ids"]

        # 3D RoPE format for both image and text position IDs.
        assert img_ids.shape[-1] == 3, "Image IDs should have 3 dimensions for RoPE"
        assert txt_ids.shape[-1] == 3, "Text IDs should have 3 dimensions for RoPE"
        # Text IDs are all zeros.
        assert torch.allclose(txt_ids, torch.zeros_like(txt_ids)), "Text IDs should be all zeros"

    def test_latent_size_scales_with_image_size(self):
        # Latent spatial size must be image_size/8 (VAE downsample contract).
        for image_size, latent in ((512, 64), (2048, 256)):
            dataset = MockFluxDataset(num_samples=5, image_size=image_size, seed=42)
            sample = dataset[0]
            assert_tensor_shape(sample["latents"], (16, latent, latent), "latents")
            expected_img_ids_len = (latent // 2) * (latent // 2)
            assert_tensor_shape(sample["img_ids"], (expected_img_ids_len, 3), "img_ids")


class TestMockFluxSchnellDataset(PrimusUT):
    """Contract tests for MockFluxSchnellDataset (Flux Schnell / MLPerf v5.1)."""

    def test_schnell_sample_shapes(self):
        # Schnell uses T5 seq_len=256 (vs 512 for dev).
        dataset = MockFluxSchnellDataset(num_samples=10, image_size=256, seed=42)
        sample = dataset[0]

        assert_tensor_shape(sample["latents"], (16, 32, 32), "latents")
        assert_tensor_shape(sample["prompt_embeds"], (256, 4096), "prompt_embeds")
        assert_tensor_shape(sample["pooled_prompt_embeds"], (768,), "pooled_prompt_embeds")
        assert_tensor_shape(sample["txt_ids"], (256, 3), "txt_ids")

    def test_schnell_pregenerated_dataset(self):
        dataset = PreGeneratedMockFluxSchnellDataset(num_samples=10, image_size=256, seed=42)

        sample0_a = dataset[0]
        sample0_b = dataset[0]

        assert torch.allclose(sample0_a["latents"], sample0_b["latents"])
        assert torch.allclose(sample0_a["prompt_embeds"], sample0_b["prompt_embeds"])


class TestSyntheticDatasetProviderLookup(PrimusUT):
    """Tests for SyntheticDatasetProvider DEFAULT_DATASETS lookup."""

    def test_schnell_class_resolves(self):
        provider = SyntheticDatasetProvider(model_type="flux_schnell")
        cls = provider._import_dataset_class()
        assert cls is not None
        assert cls.__name__ == "PreGeneratedMockFluxSchnellDataset"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
