# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Consolidated tests for encoder wrapper classes (CLIP, T5, VAE).

This file contains parametrized tests for common wrapper functionality across
different encoder types to avoid duplication. Tests that are specific to individual
encoder implementations (e.g., VAE scale/shift, CLIP pooled embeddings) remain
in their respective test files.

Encoder-specific tests remain in:
- test_clip.py: CLIP-specific logic (pooled embeddings extraction)
- test_t5.py: T5-specific logic (padding/truncation)
- test_vae.py: VAE-specific logic (scale/shift, latent shape calculation)
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from primus.backends.megatron.data.diffusion.encoders.config import (
    CLIPLConfig,
    T5XXLConfig,
    VAEConfig,
)


@pytest.fixture
def encoder_specs():
    """Define specifications for all encoder types."""
    return {
        "clip": {
            "module_path": "primus.backends.megatron.data.diffusion.encoders.text.clip_l",
            "encoder_cls_name": "CLIPLEncoder",
            "config_cls": CLIPLConfig,
            "config_kwargs": {"type": "clip_l", "model_path": "/path/to/clip"},
            "model_cls_name": "CLIPTextModel",
            "tokenizer_cls_name": "CLIPTokenizer",
            "max_length": 77,
            "embedding_dim": 768,
            "has_tokenizer": True,
        },
        "t5": {
            "module_path": "primus.backends.megatron.data.diffusion.encoders.text.t5_xxl",
            "encoder_cls_name": "T5XXLEncoder",
            "config_cls": T5XXLConfig,
            "config_kwargs": {"type": "t5_xxl", "model_path": "/path/to/t5"},
            "model_cls_name": "T5EncoderModel",
            "tokenizer_cls_name": "T5Tokenizer",
            "max_length": 512,
            "embedding_dim": 4096,
            "has_tokenizer": True,
        },
        "vae": {
            "module_path": "primus.backends.megatron.data.diffusion.encoders.image.autoencoder_kl",
            "encoder_cls_name": "AutoencoderKL",
            "config_cls": VAEConfig,
            "config_kwargs": {"type": "autoencoder_kl", "model_path": "/path/to/vae"},
            "model_cls_name": "DiffusersAutoencoderKL",
            "tokenizer_cls_name": None,
            "scale_factor": 0.3611,
            "shift_factor": 0.1159,
            "in_channels": 3,
            "out_channels": 16,
            "has_tokenizer": False,
        },
    }


@pytest.fixture
def mock_clip_model():
    """Create a mock CLIP model."""
    model = Mock()
    model.return_value = MagicMock(
        last_hidden_state=torch.randn(1, 77, 768), pooler_output=torch.randn(1, 768)
    )
    return model


@pytest.fixture
def mock_clip_tokenizer():
    """Create a mock CLIP tokenizer."""
    tokenizer = Mock()
    tokenizer.return_value = {
        "input_ids": torch.randint(0, 49408, (1, 77)),
        "attention_mask": torch.ones(1, 77),
    }
    return tokenizer


@pytest.fixture
def mock_t5_model():
    """Create a mock T5 model."""
    model = Mock()
    model.return_value = MagicMock(last_hidden_state=torch.randn(1, 512, 4096))
    return model


@pytest.fixture
def mock_t5_tokenizer():
    """Create a mock T5 tokenizer."""
    tokenizer = Mock()
    tokenizer.return_value = {
        "input_ids": torch.randint(0, 32000, (1, 512)),
        "attention_mask": torch.ones(1, 512),
    }
    return tokenizer


@pytest.fixture
def mock_vae_model():
    """Create a mock VAE model."""
    model = Mock()
    # Mock encode method
    mock_dist = Mock()
    mock_dist.sample.return_value = torch.randn(1, 16, 64, 64)
    model.encode.return_value = mock_dist
    # Mock decode method - return tuple like real diffusers AutoencoderKL
    model.decode.return_value = (torch.randn(1, 3, 512, 512),)
    return model


class TestEncoderFromPretrained:
    """Test from_pretrained for all encoder types."""

    @pytest.mark.parametrize("encoder_type", ["clip", "t5"])
    def test_text_encoder_from_pretrained_without_config(
        self,
        encoder_type,
        encoder_specs,
        mock_clip_model,
        mock_clip_tokenizer,
        mock_t5_model,
        mock_t5_tokenizer,
    ):
        """Test loading text encoders without explicit config."""
        spec = encoder_specs[encoder_type]

        # Select appropriate mocks
        if encoder_type == "clip":
            mock_model = mock_clip_model
            mock_tokenizer = mock_clip_tokenizer
        else:  # t5
            mock_model = mock_t5_model
            mock_tokenizer = mock_t5_tokenizer

        with patch(f"{spec['module_path']}.{spec['model_cls_name']}") as mock_model_cls, patch(
            f"{spec['module_path']}.{spec['tokenizer_cls_name']}"
        ) as mock_tokenizer_cls:

            mock_model_cls.from_pretrained.return_value = mock_model
            mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

            # Import encoder class
            module = __import__(spec["module_path"], fromlist=[spec["encoder_cls_name"]])
            encoder_cls = getattr(module, spec["encoder_cls_name"])

            encoder = encoder_cls.from_pretrained("/path")

            assert encoder is not None
            assert encoder.config.model_path == "/path"
            assert encoder.config.precision == "bf16"


class TestEncoderPrecisionMapping:
    """Test precision to dtype mapping for all encoders."""

    @pytest.mark.parametrize(
        "encoder_type,precision,expected_dtype",
        [
            ("clip", "bf16", torch.bfloat16),
            ("clip", "fp16", torch.float16),
            ("clip", "fp32", torch.float32),
            ("t5", "bf16", torch.bfloat16),
            ("t5", "fp16", torch.float16),
            ("t5", "fp32", torch.float32),
            ("vae", "bf16", torch.bfloat16),
            ("vae", "fp16", torch.float16),
            ("vae", "fp32", torch.float32),
        ],
    )
    def test_precision_dtype_mapping(
        self,
        encoder_type,
        precision,
        expected_dtype,
        encoder_specs,
        mock_clip_model,
        mock_clip_tokenizer,
        mock_t5_model,
        mock_t5_tokenizer,
        mock_vae_model,
    ):
        """Test that precision config is correctly mapped to torch dtype."""
        spec = encoder_specs[encoder_type]

        # Select appropriate mocks
        if encoder_type == "clip":
            mock_model = mock_clip_model
            mock_tokenizer = mock_clip_tokenizer
        elif encoder_type == "t5":
            mock_model = mock_t5_model
            mock_tokenizer = mock_t5_tokenizer
        else:  # vae
            mock_model = mock_vae_model
            mock_tokenizer = None

        if spec["has_tokenizer"]:
            with patch(f"{spec['module_path']}.{spec['model_cls_name']}") as mock_model_cls, patch(
                f"{spec['module_path']}.{spec['tokenizer_cls_name']}"
            ) as mock_tokenizer_cls:

                mock_model_cls.from_pretrained.return_value = mock_model
                mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

                # Import encoder class
                module = __import__(spec["module_path"], fromlist=[spec["encoder_cls_name"]])
                encoder_cls = getattr(module, spec["encoder_cls_name"])

                config = spec["config_cls"](**spec["config_kwargs"], precision=precision)
                encoder = encoder_cls.from_pretrained("/path", config=config)

                # Check that from_pretrained was called with correct dtype
                call_kwargs = mock_model_cls.from_pretrained.call_args[1]
                assert call_kwargs["torch_dtype"] == expected_dtype
        else:
            # VAE case
            with patch(f"{spec['module_path']}.{spec['model_cls_name']}") as mock_model_cls:
                mock_model_cls.from_pretrained.return_value = mock_model

                # Import encoder class
                module = __import__(spec["module_path"], fromlist=[spec["encoder_cls_name"]])
                encoder_cls = getattr(module, spec["encoder_cls_name"])

                config = spec["config_cls"](**spec["config_kwargs"], precision=precision)
                encoder_cls.from_pretrained("/path", config=config)

                # Check that from_pretrained was called with correct dtype
                call_kwargs = mock_model_cls.from_pretrained.call_args[1]
                assert call_kwargs["torch_dtype"] == expected_dtype


class TestEncoderDeviceHandling:
    """Test device handling for all encoder types."""

    @pytest.mark.parametrize("encoder_type", ["clip", "t5", "vae"])
    def test_device_handling(
        self,
        encoder_type,
        encoder_specs,
        mock_clip_model,
        mock_clip_tokenizer,
        mock_t5_model,
        mock_t5_tokenizer,
        mock_vae_model,
    ):
        """Test encoder device configuration."""
        spec = encoder_specs[encoder_type]

        # Select appropriate mocks
        if encoder_type == "clip":
            mock_model = mock_clip_model
            mock_tokenizer = mock_clip_tokenizer
        elif encoder_type == "t5":
            mock_model = mock_t5_model
            mock_tokenizer = mock_t5_tokenizer
        else:  # vae
            mock_model = mock_vae_model
            mock_tokenizer = None

        if spec["has_tokenizer"]:
            with patch(f"{spec['module_path']}.{spec['model_cls_name']}") as mock_model_cls, patch(
                f"{spec['module_path']}.{spec['tokenizer_cls_name']}"
            ) as mock_tokenizer_cls:

                mock_model_cls.from_pretrained.return_value = mock_model
                mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

                # Import encoder class
                module = __import__(spec["module_path"], fromlist=[spec["encoder_cls_name"]])
                encoder_cls = getattr(module, spec["encoder_cls_name"])

                config = spec["config_cls"](**spec["config_kwargs"], device="cpu")
                encoder = encoder_cls.from_pretrained("/path", config=config)

                assert encoder.device.type == "cpu"
        else:
            # VAE case
            with patch(f"{spec['module_path']}.{spec['model_cls_name']}") as mock_model_cls:
                mock_model_cls.from_pretrained.return_value = mock_model

                # Import encoder class
                module = __import__(spec["module_path"], fromlist=[spec["encoder_cls_name"]])
                encoder_cls = getattr(module, spec["encoder_cls_name"])

                config = spec["config_cls"](**spec["config_kwargs"], device="cpu")
                vae = encoder_cls.from_pretrained("/path", config=config)

                assert vae.device.type == "cpu"


class TestEncoderImportErrors:
    """Test import error handling when external libraries are not available."""

    @pytest.mark.parametrize("encoder_type", ["clip", "t5"])
    def test_text_encoder_import_error(self, encoder_type, encoder_specs):
        """Test that ImportError is raised when transformers is not installed."""
        spec = encoder_specs[encoder_type]

        with patch(f"{spec['module_path']}.{spec['model_cls_name']}", None), patch(
            f"{spec['module_path']}.{spec['tokenizer_cls_name']}", None
        ):

            # Import encoder class
            module = __import__(spec["module_path"], fromlist=[spec["encoder_cls_name"]])
            encoder_cls = getattr(module, spec["encoder_cls_name"])

            config = spec["config_cls"](**spec["config_kwargs"])

            with pytest.raises(ImportError, match="transformers library is required"):
                encoder_cls(config)

    def test_vae_import_error(self, encoder_specs):
        """Test that ImportError is raised when diffusers is not installed."""
        spec = encoder_specs["vae"]

        with patch(f"{spec['module_path']}.{spec['model_cls_name']}", None):
            # Import encoder class
            module = __import__(spec["module_path"], fromlist=[spec["encoder_cls_name"]])
            encoder_cls = getattr(module, spec["encoder_cls_name"])

            config = spec["config_cls"](**spec["config_kwargs"])

            with pytest.raises(ImportError, match="diffusers library is required"):
                encoder_cls(config)


class TestEncoderCacheDir:
    """Test cache_dir configuration support across all encoders."""

    @pytest.mark.parametrize("encoder_type", ["clip", "t5", "vae"])
    def test_cache_dir_passed_to_from_pretrained(self, encoder_type, encoder_specs):
        """Test that cache_dir is passed through to HuggingFace from_pretrained."""
        spec = encoder_specs[encoder_type]
        config_cls = spec["config_cls"]
        config_kwargs = spec["config_kwargs"].copy()
        cache_dir = "/custom/cache/path"

        # Create config with cache_dir
        config = config_cls(**config_kwargs, cache_dir=cache_dir)

        # Mock the model/tokenizer classes
        mock_model = MagicMock()
        mock_model_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_model_instance.to.return_value = mock_model_instance

        # FIXED: Use just attribute names for patch.multiple (not full module paths)
        patches = {spec["model_cls_name"]: mock_model}

        # Add tokenizer patch for text encoders
        if spec["has_tokenizer"]:
            mock_tokenizer = MagicMock()
            mock_tokenizer.from_pretrained.return_value = Mock()
            patches[spec["tokenizer_cls_name"]] = mock_tokenizer

        # Also mock the helper function
        mock_helper = MagicMock(return_value=mock_model_instance)
        patches["load_pretrained_with_subfolder_fallback"] = mock_helper

        with patch.dict("sys.modules", {k: MagicMock() for k in ["transformers", "diffusers"]}):
            with patch.multiple(spec["module_path"], **patches):
                # Import and call from_pretrained
                module = __import__(spec["module_path"], fromlist=[spec["encoder_cls_name"]])
                encoder_cls = getattr(module, spec["encoder_cls_name"])

                encoder_cls.from_pretrained("/path", config=config)

                # The helper must be invoked, and cache_dir must be forwarded to it.
                assert mock_helper.called, "load_pretrained_with_subfolder_fallback was never called"
                call_kwargs = mock_helper.call_args.kwargs
                assert "cache_dir" in call_kwargs, "cache_dir not passed to helper function"
                assert call_kwargs["cache_dir"] == cache_dir

    def test_flux_encoder_config_cache_dir(self):
        """Test that FluxEncoderConfig.from_pretrained_flux accepts cache_dir."""
        from primus.backends.megatron.data.diffusion.encoders.config import (
            FluxEncoderConfig,
        )

        cache_dir = "/shared/hf_cache"
        flux_config = FluxEncoderConfig.from_pretrained_flux(
            model_path="black-forest-labs/FLUX.1-dev", cache_dir=cache_dir
        )

        # All sub-configs should have the same cache_dir
        assert flux_config.vae.cache_dir == cache_dir
        assert flux_config.t5.cache_dir == cache_dir
        assert flux_config.clip.cache_dir == cache_dir


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
