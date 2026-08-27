# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Tests for diffusion TaskEncoders.

Tests both EncodedDiffusionTaskEncoder (pre-encoded data) and
RawDiffusionTaskEncoder (raw images and text).
"""


import pytest
import torch

from primus.backends.megatron.data.diffusion.task_encoders import (
    DiffusionSample,
    EncodedDiffusionTaskEncoder,
    RawDiffusionTaskEncoder,
    cook_preencoded_diffusion,
    cook_raw_images,
)


class TestCookPreencodedDiffusion:
    """Tests for cook_preencoded_diffusion cooker function."""

    def test_cook_with_tensor_data(self, preencoded_sample):
        """Test cooking with direct tensor data."""
        result = cook_preencoded_diffusion(preencoded_sample)

        assert isinstance(result, DiffusionSample)
        assert result.latents.shape == (16, 64, 64)
        assert result.prompt_embeds.shape == (256, 4096)
        assert result.pooled_prompt_embeds.shape == (768,)
        assert result.caption == "A test caption"

    def test_cook_with_bytes_data(self, preencoded_sample_bytes):
        """Test cooking with tensor data as bytes."""
        result = cook_preencoded_diffusion(preencoded_sample_bytes)

        assert isinstance(result, DiffusionSample)
        assert result.latents.shape == (16, 64, 64)
        assert result.prompt_embeds.shape == (256, 4096)
        assert result.pooled_prompt_embeds.shape == (768,)
        assert result.caption == "Another test caption"

    def test_cook_ignores_text_ids(self, preencoded_sample, sample_text_ids):
        """Test that text_ids in sample are ignored (position IDs generated at runtime)."""
        # Even if text_ids are provided in the sample, they should not be stored
        # because position IDs are generated at runtime based on actual tensor shapes
        preencoded_sample["text_ids.pth"] = sample_text_ids
        result = cook_preencoded_diffusion(preencoded_sample)

        # Verify that text_ids attribute doesn't exist (not part of DiffusionSample)
        assert not hasattr(result, "text_ids"), "text_ids should not be stored in DiffusionSample"

    def test_cook_missing_required_key_latents(self, preencoded_sample):
        """Test error when latents are missing."""
        del preencoded_sample["latents.pth"]

        # Production now emits a distinct message for missing latents (vs
        # missing prompt_embeds / pooled_prompt_embeds) since 'latents.pth' is
        # interchangeable with 'mean.pth' / 'logvar.pth' for resample-mode.
        with pytest.raises(
            ValueError,
            match=r"Sample must have 'latents\.pth' or 'mean\.pth'/'logvar\.pth'",
        ):
            cook_preencoded_diffusion(preencoded_sample)

    def test_cook_missing_required_key_prompt_embeds(self, preencoded_sample):
        """Test error when prompt_embeds are missing."""
        del preencoded_sample["prompt_embeds.pth"]

        with pytest.raises(ValueError, match="Sample missing required keys"):
            cook_preencoded_diffusion(preencoded_sample)

    def test_cook_missing_required_key_pooled(self, preencoded_sample):
        """Test error when pooled_prompt_embeds are missing."""
        del preencoded_sample["pooled_prompt_embeds.pth"]

        with pytest.raises(ValueError, match="Sample missing required keys"):
            cook_preencoded_diffusion(preencoded_sample)

    def test_cook_caption_as_str(self, preencoded_sample):
        """Test caption handling when it's a string."""
        preencoded_sample["caption.txt"] = "String caption"
        result = cook_preencoded_diffusion(preencoded_sample)

        assert result.caption == "String caption"

    def test_cook_caption_missing(self, preencoded_sample):
        """Test caption handling when missing."""
        del preencoded_sample["caption.txt"]
        result = cook_preencoded_diffusion(preencoded_sample)

        assert result.caption == ""

    def test_cook_preserves_sample_keys(self, preencoded_sample):
        """Test that sample metadata keys are preserved."""
        result = cook_preencoded_diffusion(preencoded_sample)

        assert result.__key__ == "sample_001"
        assert result.__subflavors__ == {"encoding": "preencoded"}


class TestCookRawImages:
    """Tests for cook_raw_images cooker function."""

    def test_cook_with_png(self, sample_image_bytes):
        """Test cooking raw PNG image."""
        sample = {
            "__key__": "test_png",
            "__restore_key__": lambda: "test_png",
            "__subflavors__": {"encoding": "raw"},
            "images": sample_image_bytes,
            "txt": b"PNG caption",
        }
        result = cook_raw_images(sample)

        assert result["images"] == sample_image_bytes
        assert result["txt"] == b"PNG caption"

    def test_cook_preserves_metadata(self, raw_sample):
        """Test that metadata keys are preserved."""
        result = cook_raw_images(raw_sample)

        assert result["__key__"] == "sample_003"
        assert result["__subflavors__"] == {"encoding": "raw"}


class TestEncodedDiffusionTaskEncoder:
    """Tests for EncodedDiffusionTaskEncoder."""

    def test_batch_with_all_fields(self, sample_latents, sample_prompt_embeds, sample_pooled_prompt_embeds):
        """Test batching samples with all standard fields (position IDs generated at runtime)."""
        encoder = EncodedDiffusionTaskEncoder(worker_config=None)

        samples = [
            DiffusionSample(
                __key__="s1",
                __restore_key__=lambda: "s1",
                __subflavors__={"encoding": "preencoded"},
                latents=sample_latents.squeeze(0),
                prompt_embeds=sample_prompt_embeds.squeeze(0),
                pooled_prompt_embeds=sample_pooled_prompt_embeds.squeeze(0),
                caption="Caption 1",
            ),
            DiffusionSample(
                __key__="s2",
                __restore_key__=lambda: "s2",
                __subflavors__={"encoding": "preencoded"},
                latents=sample_latents.squeeze(0),
                prompt_embeds=sample_prompt_embeds.squeeze(0),
                pooled_prompt_embeds=sample_pooled_prompt_embeds.squeeze(0),
                caption="Caption 2",
            ),
        ]

        batch = encoder.batch(samples)

        assert "latents" in batch
        assert "prompt_embeds" in batch
        assert "pooled_prompt_embeds" in batch
        # text_ids not in batch - position IDs are generated at runtime
        # Note: 'captions' may or may not be in batch depending on implementation

        assert batch["latents"].shape[0] == 2
        assert batch["prompt_embeds"].shape[0] == 2
        assert batch["pooled_prompt_embeds"].shape[0] == 2


class TestValidationTimestepCollation:
    """Tests for per-sample timestep validation during batching.

    MLPerf validation is defined per-sample at t in {0..7}; a batch that
    silently loses the field falls back to a positional assignment in
    forward_step, so these are error paths rather than conveniences.
    """

    @staticmethod
    def _sample(key, timestep, latents, prompt_embeds, pooled):
        return DiffusionSample(
            __key__=key,
            __restore_key__=lambda: key,
            __subflavors__={"encoding": "preencoded"},
            latents=latents.squeeze(0),
            prompt_embeds=prompt_embeds.squeeze(0),
            pooled_prompt_embeds=pooled.squeeze(0),
            timestep=None if timestep is None else torch.tensor(timestep),
        )

    def _batch(self, timesteps, sample_latents, sample_prompt_embeds, sample_pooled_prompt_embeds):
        encoder = EncodedDiffusionTaskEncoder(worker_config=None)
        samples = [
            self._sample(f"s{i}", t, sample_latents, sample_prompt_embeds, sample_pooled_prompt_embeds)
            for i, t in enumerate(timesteps)
        ]
        return encoder.batch(samples)

    def test_timesteps_are_collated(self, sample_latents, sample_prompt_embeds, sample_pooled_prompt_embeds):
        batch = self._batch([0, 4, 7], sample_latents, sample_prompt_embeds, sample_pooled_prompt_embeds)

        assert "timestep" in batch
        assert batch["timestep"].tolist() == [0, 4, 7]

    def test_absent_timesteps_leave_key_out(
        self, sample_latents, sample_prompt_embeds, sample_pooled_prompt_embeds
    ):
        """The training split has no timestep; batching must not invent one."""
        batch = self._batch([None, None], sample_latents, sample_prompt_embeds, sample_pooled_prompt_embeds)

        assert "timestep" not in batch

    def test_partially_missing_timestep_raises_with_keys(
        self, sample_latents, sample_prompt_embeds, sample_pooled_prompt_embeds
    ):
        """Checking samples[0] alone would let this through or crash opaquely."""
        with pytest.raises(ValueError, match="lack a 'timestep' field") as excinfo:
            self._batch([3, None, 5], sample_latents, sample_prompt_embeds, sample_pooled_prompt_embeds)

        assert "s1" in str(excinfo.value)

    def test_missing_timestep_on_first_sample_is_still_caught(
        self, sample_latents, sample_prompt_embeds, sample_pooled_prompt_embeds
    ):
        """The inverse ordering: samples[0] absent but later ones present."""
        with pytest.raises(ValueError, match="lack a 'timestep' field"):
            self._batch([None, 2, 6], sample_latents, sample_prompt_embeds, sample_pooled_prompt_embeds)

    def test_out_of_range_timestep_raises(
        self, sample_latents, sample_prompt_embeds, sample_pooled_prompt_embeds
    ):
        with pytest.raises(ValueError, match=r"must be in \[0, 7\]") as excinfo:
            self._batch([0, 9, 3], sample_latents, sample_prompt_embeds, sample_pooled_prompt_embeds)

        assert "s1" in str(excinfo.value)

    def test_negative_timestep_raises(
        self, sample_latents, sample_prompt_embeds, sample_pooled_prompt_embeds
    ):
        with pytest.raises(ValueError, match=r"must be in \[0, 7\]"):
            self._batch([-1, 3], sample_latents, sample_prompt_embeds, sample_pooled_prompt_embeds)

    def test_float_timestep_raises(self, sample_latents, sample_prompt_embeds, sample_pooled_prompt_embeds):
        """sigma = t / 8 expects an integer index, not a pre-divided float."""
        with pytest.raises(ValueError, match="must be an integer type"):
            self._batch([0.5, 0.25], sample_latents, sample_prompt_embeds, sample_pooled_prompt_embeds)


class TestRawDiffusionTaskEncoder:
    """Tests for RawDiffusionTaskEncoder."""

    def test_batch_with_raw_data(self, sample_image_bytes):
        """Test batching raw image samples."""
        encoder = RawDiffusionTaskEncoder(worker_config=None)

        samples = [
            {"images": sample_image_bytes, "txt": b"Caption 1"},
            {"images": sample_image_bytes, "txt": b"Caption 2"},
        ]

        batch = encoder.batch(samples)

        assert "images" in batch
        assert "txt" in batch
        assert len(batch["images"]) == 2
        assert len(batch["txt"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
