# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Shared fixtures for data module tests.

Provides mocked encoders, sample data, and utilities for testing.
"""

import io
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image

# ============================================================================
# Mock Encoder Fixtures
# ============================================================================


@pytest.fixture
def mock_vae_model():
    """Mock VAE model for testing."""
    mock = MagicMock()

    # Mock encode method
    def mock_encode(images):
        batch_size = images.shape[0] if isinstance(images, torch.Tensor) else 1
        # Return mock latent distribution
        latent_dist = MagicMock()
        latent_dist.sample = MagicMock(return_value=torch.randn(batch_size, 16, 64, 64))
        return MagicMock(latent_dist=latent_dist)

    mock.encode = mock_encode
    mock.to = MagicMock(return_value=mock)
    mock.eval = MagicMock(return_value=mock)

    return mock


@pytest.fixture
def mock_t5_model():
    """Mock T5 model for testing."""
    mock = MagicMock()

    # Mock encoder
    def mock_forward(input_ids, **kwargs):
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        hidden_dim = 4096  # T5-XXL hidden dim
        return MagicMock(last_hidden_state=torch.randn(batch_size, seq_len, hidden_dim))

    mock.encoder = MagicMock()
    mock.encoder.return_value = mock_forward(torch.zeros(1, 77, dtype=torch.long))
    mock.to = MagicMock(return_value=mock)
    mock.eval = MagicMock(return_value=mock)

    return mock


@pytest.fixture
def mock_t5_tokenizer():
    """Mock T5 tokenizer for testing."""
    mock = MagicMock()

    def mock_tokenize(text, **kwargs):
        if isinstance(text, str):
            text = [text]
        batch_size = len(text)
        max_length = kwargs.get("max_length", 77)
        return {
            "input_ids": torch.randint(0, 32000, (batch_size, max_length)),
            "attention_mask": torch.ones(batch_size, max_length),
        }

    mock.return_value = mock_tokenize(["test"])
    mock.side_effect = None
    mock.__call__ = mock_tokenize

    return mock


@pytest.fixture
def mock_clip_model():
    """Mock CLIP model for testing."""
    mock = MagicMock()

    # Mock text encoder
    def mock_text_model(input_ids, **kwargs):
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        hidden_dim = 768  # CLIP-L hidden dim
        pooled_dim = 768
        return MagicMock(
            last_hidden_state=torch.randn(batch_size, seq_len, hidden_dim),
            pooler_output=torch.randn(batch_size, pooled_dim),
        )

    mock.text_model = MagicMock()
    mock.text_model.return_value = mock_text_model(torch.zeros(1, 77, dtype=torch.long))
    mock.to = MagicMock(return_value=mock)
    mock.eval = MagicMock(return_value=mock)

    return mock


@pytest.fixture
def mock_clip_tokenizer():
    """Mock CLIP tokenizer for testing."""
    mock = MagicMock()

    def mock_tokenize(text, **kwargs):
        if isinstance(text, str):
            text = [text]
        batch_size = len(text)
        max_length = kwargs.get("max_length", 77)
        return {
            "input_ids": torch.randint(0, 49408, (batch_size, max_length)),
            "attention_mask": torch.ones(batch_size, max_length),
        }

    mock.return_value = mock_tokenize(["test"])
    mock.__call__ = mock_tokenize

    return mock


# ============================================================================
# Sample Data Fixtures
# ============================================================================


@pytest.fixture
def sample_image():
    """Create a sample PIL Image."""
    return Image.new("RGB", (512, 512), color="red")


@pytest.fixture
def sample_image_tensor():
    """Create a sample image tensor."""
    return torch.randn(3, 512, 512)


@pytest.fixture
def sample_image_bytes():
    """Create sample image as bytes."""
    img = Image.new("RGB", (512, 512), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def sample_text():
    """Sample text caption."""
    return "A beautiful landscape with mountains and trees"


@pytest.fixture
def sample_latents():
    """Sample VAE latents (without batch dimension for single sample)."""
    return torch.randn(16, 64, 64)


@pytest.fixture
def sample_prompt_embeds():
    """Sample T5 text embeddings (without batch dimension for single sample)."""
    return torch.randn(256, 4096)


@pytest.fixture
def sample_pooled_prompt_embeds():
    """Sample CLIP pooled embeddings (without batch dimension for single sample)."""
    return torch.randn(768)


@pytest.fixture
def sample_text_ids():
    """Sample text position IDs (without batch dimension for single sample)."""
    return torch.randn(256, 3)


# ============================================================================
# WebDataset Sample Fixtures
# ============================================================================


@pytest.fixture
def preencoded_sample(sample_latents, sample_prompt_embeds, sample_pooled_prompt_embeds):
    """Sample with pre-encoded data (tensor format)."""
    return {
        "__key__": "sample_001",
        "__restore_key__": lambda: "sample_001",
        "__subflavors__": {"encoding": "preencoded"},
        "latents.pth": sample_latents,  # Already correct shape (16, 64, 64)
        "prompt_embeds.pth": sample_prompt_embeds,  # Already correct shape (256, 4096)
        "pooled_prompt_embeds.pth": sample_pooled_prompt_embeds,  # Already correct shape (768,)
        "text_ids.pth": None,  # Optional
        "caption.txt": b"A test caption",
    }


@pytest.fixture
def preencoded_sample_bytes(sample_latents, sample_prompt_embeds, sample_pooled_prompt_embeds):
    """Sample with pre-encoded data (bytes format)."""

    def tensor_to_bytes(tensor):
        buf = io.BytesIO()
        torch.save(tensor, buf)
        buf.seek(0)
        return buf.getvalue()

    return {
        "__key__": "sample_002",
        "__restore_key__": lambda: "sample_002",
        "__subflavors__": {"encoding": "preencoded"},
        "latents.pth": tensor_to_bytes(sample_latents),  # Convert to bytes
        "prompt_embeds.pth": tensor_to_bytes(sample_prompt_embeds),  # Convert to bytes
        "pooled_prompt_embeds.pth": tensor_to_bytes(sample_pooled_prompt_embeds),  # Convert to bytes
        "caption.txt": b"Another test caption",
    }


@pytest.fixture
def raw_sample(sample_image_bytes):
    """Sample with raw image data."""
    return {
        "__key__": "sample_003",
        "__restore_key__": lambda: "sample_003",
        "__subflavors__": {"encoding": "raw"},
        "images": sample_image_bytes,
        "txt": b"Raw image caption",
    }


# ============================================================================
# Temporary Directory Fixtures
# ============================================================================


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_model_path(temp_output_dir):
    """Create a temporary model path."""
    model_dir = temp_output_dir / "model"
    model_dir.mkdir()
    return model_dir
