# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Synthetic datasets for diffusion model training and testing.

Provides mock datasets that generate random tensors with correct shapes
for development, testing, and benchmarking without requiring real data.

The module provides a flexible preset-based system for creating mock datasets
for different diffusion model architectures.

Quick Start
-----------

Using a preset (easiest):
    >>> from primus.backends.megatron.data.synthetic import MockFluxDataset
    >>> dataset = MockFluxDataset(num_samples=100, image_size=1024)
    >>> sample = dataset[0]
    >>> print(sample.keys())  # latents, t5_text_embeddings, clip_pooled_embeddings, etc.

Using the preset system directly:
    >>> from primus.backends.megatron.data.synthetic import MockDiffusionDataset
    >>> dataset = MockDiffusionDataset(num_samples=100, model_preset='flux')

Custom configuration for other models:
    >>> from primus.backends.megatron.data.synthetic import (
    ...     MockDiffusionDataset,
    ...     LatentConfig,
    ...     TextEmbeddingConfig,
    ... )
    >>> dataset = MockDiffusionDataset(
    ...     num_samples=100,
    ...     latent_config=LatentConfig(channels=4, downsample_factor=8),
    ...     text_embedding_configs=[
    ...         TextEmbeddingConfig(key='text_embeddings', shape=(77, 768)),
    ...         TextEmbeddingConfig(key='pooled_embeddings', shape=(1280,)),
    ...     ],
    ... )

Adding a New Model
------------------

To add support for a new diffusion model (e.g., SDXL):

1. Define a position ID generator function (if needed):
    >>> def sdxl_position_id_generator(image_size, latent_size, **kwargs):
    ...     # SDXL doesn't use position IDs
    ...     return {}

2. Register the model preset:
    >>> from primus.backends.megatron.data.synthetic import (
    ...     MockDiffusionDataset,
    ...     ModelPreset,
    ...     LatentConfig,
    ...     TextEmbeddingConfig,
    ... )
    >>> MockDiffusionDataset.register_model('sdxl', ModelPreset(
    ...     latent_config=LatentConfig(channels=4, downsample_factor=8),
    ...     text_embedding_configs=[
    ...         TextEmbeddingConfig(key='text_embeddings', shape=(77, 2048)),
    ...         TextEmbeddingConfig(key='pooled_embeddings', shape=(1280,)),
    ...     ],
    ...     position_id_generator=sdxl_position_id_generator,
    ... ))

3. Create a convenience class (optional):
    >>> class MockSDXLDataset(MockDiffusionDataset):
    ...     def __init__(self, num_samples=100, image_size=1024, seed=None):
    ...         super().__init__(
    ...             num_samples=num_samples,
    ...             image_size=image_size,
    ...             model_preset='sdxl',
    ...             seed=seed,
    ...         )

Architecture
------------

The module uses a hybrid registry + subclass approach:

- MockDiffusionDataset: Generic base class with preset system
- MockFluxDataset: Convenience class for Flux models
- Model presets: Registered configurations for specific model architectures
- Configuration dataclasses: LatentConfig, TextEmbeddingConfig, ModelPreset

This design provides flexibility for testing various diffusion models while
maintaining a simple API for common use cases.
"""

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from primus.backends.megatron.core.models.diffusion.flux.utils import (
    generate_image_position_ids,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Data Structures
# ============================================================================


@dataclass
class LatentConfig:
    """Configuration for latent generation."""

    channels: int  # Number of latent channels
    downsample_factor: int = 8  # VAE spatial downsampling factor


@dataclass
class TextEmbeddingConfig:
    """Configuration for a single text embedding."""

    key: str  # Output dictionary key (e.g., 't5_text_embeddings')
    shape: Tuple[int, ...]  # Shape without batch dimension
    dtype: str = "float32"  # Data type


@dataclass
class ModelPreset:
    """Complete configuration preset for a diffusion model."""

    latent_config: LatentConfig
    text_embedding_configs: List[TextEmbeddingConfig]
    position_id_generator: Optional[Callable] = None


# ============================================================================
# Position ID Generator Functions
# ============================================================================


def flux_position_id_generator(
    image_size: int,
    latent_size: int,
    t5_seq_len: int,
    seed: int,
    idx: int,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, torch.Tensor]:
    """
    Generate Flux-specific position IDs.

    Args:
        image_size: Original image size
        latent_size: Latent space size (after VAE downsampling)
        t5_seq_len: T5 sequence length
        seed: Random seed (unused but kept for consistency)
        idx: Sample index (unused but kept for consistency)
        device: Device to create tensors on (default: cpu)

    Returns:
        Dictionary with 'img_ids' and 'txt_ids' tensors
    """
    # Generate image position IDs using Flux's standard function
    img_ids = generate_image_position_ids(
        batch_size=1,
        height=latent_size,
        width=latent_size,
        device=device,
        dtype=torch.float32,
    ).squeeze(
        0
    )  # Remove batch dimension: (1, seq_len, 3) -> (seq_len, 3)

    # Text position IDs (all zeros for text in Flux)
    txt_ids = torch.zeros(t5_seq_len, 3, device=device)

    return {"img_ids": img_ids, "txt_ids": txt_ids}


class MockDiffusionDataset(Dataset):
    """
    Generic mock dataset for diffusion models.

    Supports both preset-based initialization (via model_preset) and
    custom configuration (via explicit configs). This allows the dataset
    to generate synthetic data for various diffusion model architectures.

    Usage:
        # Using a preset (e.g., 'flux')
        dataset = MockDiffusionDataset(num_samples=100, model_preset='flux')

        # Using custom configuration
        dataset = MockDiffusionDataset(
            num_samples=100,
            latent_config=LatentConfig(channels=4, downsample_factor=8),
            text_embedding_configs=[
                TextEmbeddingConfig(key='text_embeddings', shape=(77, 768)),
            ]
        )

        # Using legacy parameters (backward compatible)
        dataset = MockDiffusionDataset(
            num_samples=100,
            latent_channels=16,
            t5_seq_len=512,
            t5_hidden_dim=4096,
            clip_hidden_dim=768,
        )
    """

    # Registry of model presets
    MODEL_PRESETS: Dict[str, ModelPreset] = {}

    @classmethod
    def register_model(cls, name: str, preset: ModelPreset):
        """
        Register a model preset for easy reuse.

        Args:
            name: Name of the preset (e.g., 'flux', 'sdxl')
            preset: ModelPreset configuration object
        """
        cls.MODEL_PRESETS[name] = preset
        logger.info(f"Registered model preset: {name}")

    def __init__(
        self,
        num_samples: int = 100,
        image_size: int = 1024,
        # Option 1: Use preset
        model_preset: Optional[str] = None,
        # Option 2: Custom configuration
        latent_config: Optional[LatentConfig] = None,
        text_embedding_configs: Optional[List[TextEmbeddingConfig]] = None,
        position_id_generator: Optional[Callable] = None,
        seed: Optional[int] = None,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cpu",
        vae_latent_mode: str = "presampled",
        is_validation: bool = False,
    ):
        """
        Initialize mock diffusion dataset.

        Note: Always returns preencoded format (latents + embeddings + position IDs).
        This matches the output of EncodedDiffusionTaskEncoder.

        Args:
            num_samples: Number of samples in dataset
            image_size: Image size (assumes square images)
            model_preset: Name of registered model preset to use
            latent_config: Custom latent configuration (ignored if model_preset set)
            text_embedding_configs: Custom text embedding configs (ignored if model_preset set)
            position_id_generator: Custom position ID generator function (ignored if model_preset set)
            seed: Random seed for reproducibility
            dtype: Data type for tensors (default: torch.bfloat16)
            device: Device to create tensors on ('cpu' or 'cuda', default: 'cpu')
        """
        super().__init__()

        # Load from preset if specified
        if model_preset:
            if model_preset not in self.MODEL_PRESETS:
                raise ValueError(
                    f"Unknown model preset: {model_preset}. "
                    f"Available presets: {list(self.MODEL_PRESETS.keys())}"
                )
            preset = self.MODEL_PRESETS[model_preset]
            self.latent_config = preset.latent_config
            self.text_embedding_configs = preset.text_embedding_configs
            self.position_id_generator = preset.position_id_generator
            logger.info(f"Using model preset: {model_preset}")
        # Otherwise use custom configuration
        else:
            if latent_config is None:
                raise ValueError("Must provide either model_preset or latent_config")
            self.latent_config = latent_config
            self.text_embedding_configs = text_embedding_configs or []
            self.position_id_generator = position_id_generator

        # Common initialization
        self.num_samples = num_samples
        self.image_size = image_size
        self.seed = seed if seed is not None else 0
        self.dtype = dtype
        self.device = torch.device(device)
        self.vae_latent_mode = vae_latent_mode
        self.is_validation = is_validation
        self.latent_size = image_size // self.latent_config.downsample_factor

        # Store legacy attributes for backward compatibility
        self.latent_channels = self.latent_config.channels

        # Extract T5 and CLIP dims if present (for backward compatibility)
        self.t5_seq_len = None
        self.t5_hidden_dim = None
        self.clip_hidden_dim = None
        for emb_config in self.text_embedding_configs:
            if emb_config.key in ("t5_text_embeddings", "prompt_embeds") and len(emb_config.shape) == 2:
                self.t5_seq_len, self.t5_hidden_dim = emb_config.shape
            elif (
                emb_config.key in ("clip_pooled_embeddings", "pooled_prompt_embeds")
                and len(emb_config.shape) == 1
            ):
                self.clip_hidden_dim = emb_config.shape[0]

        logger.info(
            f"Initialized MockDiffusionDataset: {num_samples} samples, "
            f"image_size={image_size}, latent_size={self.latent_size}"
        )

    def __len__(self) -> int:
        """Return number of samples."""
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a mock sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with mock data matching DiffusionSample format

        Raises:
            IndexError: If idx is out of bounds
        """
        if idx >= self.num_samples or idx < 0:
            raise IndexError(f"Index {idx} out of range for dataset with {self.num_samples} samples")

        # Set seed based on instance seed and idx for reproducibility
        gen = torch.Generator(device=self.device)
        gen.manual_seed(self.seed + idx)

        latent_shape = (
            self.latent_config.channels,
            self.latent_size,
            self.latent_size,
        )

        if self.vae_latent_mode == "resample":
            mean = torch.randn(*latent_shape, generator=gen, dtype=self.dtype, device=self.device)
            logvar = torch.randn(*latent_shape, generator=gen, dtype=self.dtype, device=self.device)
            sample = {"mean": mean, "logvar": logvar}
        else:
            latents = torch.randn(*latent_shape, generator=gen, dtype=self.dtype, device=self.device)
            sample = {"latents": latents}

        # Generate text embeddings based on configs
        for emb_config in self.text_embedding_configs:
            # Get dtype
            dtype = getattr(torch, emb_config.dtype, torch.float32)

            # Generate tensor with the specified shape
            embedding = torch.randn(
                *emb_config.shape,
                generator=gen,
                dtype=dtype,
                device=self.device,
            )

            sample[emb_config.key] = embedding

        # Generate position IDs if generator provided
        if self.position_id_generator is not None:
            position_ids = self.position_id_generator(
                image_size=self.image_size,
                latent_size=self.latent_size,
                t5_seq_len=self.t5_seq_len if self.t5_seq_len else 512,
                seed=self.seed,
                idx=idx,
                device=self.device,
            )
            sample.update(position_ids)

        # Generate mock caption
        sample["caption"] = f"Mock caption {idx}"

        if self.is_validation:
            sample["timestep"] = torch.tensor(idx % 8)

        return sample


# ============================================================================
# Register Model Presets
# ============================================================================

# Register Flux model preset (FLUX.1-dev: T5 max_sequence_length=512)
MockDiffusionDataset.register_model(
    "flux",
    ModelPreset(
        latent_config=LatentConfig(channels=16, downsample_factor=8),
        text_embedding_configs=[
            TextEmbeddingConfig(
                key="prompt_embeds", shape=(512, 4096), dtype="bfloat16"
            ),  # T5 text embeddings
            TextEmbeddingConfig(
                key="pooled_prompt_embeds", shape=(768,), dtype="bfloat16"
            ),  # CLIP pooled embeddings
        ],
        position_id_generator=flux_position_id_generator,
    ),
)

# Register Flux Schnell preset (FLUX.1-schnell: T5 max_sequence_length=256)
# Matches the MLPerf Training v5.1 benchmark specification and NVIDIA reference.
MockDiffusionDataset.register_model(
    "flux_schnell",
    ModelPreset(
        latent_config=LatentConfig(channels=16, downsample_factor=8),
        text_embedding_configs=[
            TextEmbeddingConfig(
                key="prompt_embeds", shape=(256, 4096), dtype="bfloat16"
            ),  # T5 text embeddings
            TextEmbeddingConfig(
                key="pooled_prompt_embeds", shape=(768,), dtype="bfloat16"
            ),  # CLIP pooled embeddings
        ],
        position_id_generator=flux_position_id_generator,
    ),
)


# ============================================================================
# Model-Specific Convenience Classes
# ============================================================================


class MockFluxDataset(MockDiffusionDataset):
    """
    Mock dataset specifically for Flux diffusion model.

    Convenience wrapper that uses the 'flux' preset with Flux-specific
    default parameters:
        - image_size=1024
        - latent_channels=16
        - t5_seq_len=512
        - t5_hidden_dim=4096
        - clip_hidden_dim=768

    This class provides a simplified API for the common case of testing
    Flux models without needing to understand the preset system.
    """

    def __init__(
        self,
        num_samples: int = 100,
        image_size: int = 1024,
        seed: Optional[int] = None,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        **kwargs,
    ):
        """
        Initialize mock Flux dataset.

        Note: Always returns preencoded format (matches TaskEncoder output).

        Args:
            num_samples: Number of samples in dataset
            image_size: Image size (default 1024 for Flux)
            seed: Random seed for reproducibility
            dtype: Data type for tensors (default: torch.bfloat16)
            device: Device to create tensors on ('cpu' or 'cuda', default: 'cuda')
        """
        super().__init__(
            num_samples=num_samples,
            image_size=image_size,
            model_preset="flux",
            seed=seed,
            dtype=dtype,
            device=device,
            **kwargs,
        )

        logger.info(f"Initialized MockFluxDataset with Flux preset on device={device}")


class PreGeneratedMockFluxDataset(MockFluxDataset):
    """
    Pre-generated mock Flux dataset for maximum training throughput.

    This dataset generates all samples once during initialization and caches
    them in memory. This eliminates on-the-fly random number generation overhead
    and provides benchmark-quality performance matching latent caching approaches.

    Memory usage: For 1000 samples at 512x512 resolution with bf16:
        - Latents: 1000 * 16 * 64 * 64 * 2 bytes = ~128 MB
        - Text embeddings: 1000 * (512 * 4096 + 768) * 2 bytes = ~4 GB
        - Total: ~4.2 GB per GPU (acceptable for 192GB MI300X)

    Use this instead of MockFluxDataset for:
        - Performance benchmarking
        - Maximum training throughput
        - Consistent timing measurements
    """

    def __init__(self, *args, **kwargs):
        """Initialize and pre-generate all samples."""
        from primus.core.utils.module_utils import log_rank_0

        # Initialize parent class
        super().__init__(*args, **kwargs)

        log_rank_0(f"Pre-generating {self.num_samples} mock Flux samples...")
        log_rank_0(f"  Image size: {self.image_size}x{self.image_size}")
        log_rank_0(f"  Latent size: {self.latent_size}x{self.latent_size}")

        # Pre-generate all samples using parent's __getitem__
        self._samples = [MockFluxDataset.__getitem__(self, i) for i in range(self.num_samples)]

        # Calculate memory usage (only count tensors)
        sample_size = sum(
            tensor.element_size() * tensor.numel()
            for tensor in self._samples[0].values()
            if hasattr(tensor, "element_size")  # Only count torch tensors
        )
        total_mb = (sample_size * self.num_samples) / (1024 * 1024)

        log_rank_0(f"Pre-generation complete!")
        log_rank_0(f"  Memory usage: {total_mb:.1f} MB")
        log_rank_0(f"  Data loading overhead: eliminated")

    def __getitem__(self, idx: int):
        """
        Return pre-generated sample (no computation).

        This is 10-100x faster than on-the-fly generation as it's just
        a memory lookup with no random number generation or tensor creation.
        """
        if idx >= self.num_samples or idx < 0:
            raise IndexError(f"Index {idx} out of range for dataset with {self.num_samples} samples")
        return self._samples[idx]


class MockFluxSchnellDataset(MockDiffusionDataset):
    """
    Mock dataset for FLUX.1-schnell (MLPerf Training v5.1 benchmark).

    Uses the 'flux_schnell' preset which matches the NVIDIA MLPerf reference:
        - T5 max_sequence_length=256 (vs 512 for FLUX.1-dev)
        - image_size=256 (benchmark default)
        - latent_channels=16, context_dim=4096, vec_in_dim=768
    """

    def __init__(
        self,
        num_samples: int = 100,
        image_size: int = 256,
        seed: Optional[int] = None,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__(
            num_samples=num_samples,
            image_size=image_size,
            model_preset="flux_schnell",
            seed=seed,
            dtype=dtype,
            device=device,
            **kwargs,
        )
        logger.info(f"Initialized MockFluxSchnellDataset on device={device}")


class PreGeneratedMockFluxSchnellDataset(MockFluxSchnellDataset):
    """
    Pre-generated mock FLUX.1-schnell dataset for benchmark throughput.

    Generates all samples once at init and caches them in memory.
    Matches the MLPerf Training v5.1 data shapes (T5 seq_len=256).
    """

    def __init__(self, *args, **kwargs):
        from primus.core.utils.module_utils import log_rank_0

        super().__init__(*args, **kwargs)

        log_rank_0(f"Pre-generating {self.num_samples} mock Flux-schnell samples...")
        log_rank_0(f"  Image size: {self.image_size}x{self.image_size}")
        log_rank_0(f"  Latent size: {self.latent_size}x{self.latent_size}")

        self._samples = [MockFluxSchnellDataset.__getitem__(self, i) for i in range(self.num_samples)]

        sample_size = sum(
            tensor.element_size() * tensor.numel()
            for tensor in self._samples[0].values()
            if hasattr(tensor, "element_size")
        )
        total_mb = (sample_size * self.num_samples) / (1024 * 1024)

        log_rank_0(f"Pre-generation complete!")
        log_rank_0(f"  Memory usage: {total_mb:.1f} MB")
        log_rank_0(f"  Data loading overhead: eliminated")

    def __getitem__(self, idx: int):
        if idx >= self.num_samples or idx < 0:
            raise IndexError(f"Index {idx} out of range for dataset with {self.num_samples} samples")
        return self._samples[idx]


__all__ = [
    "LatentConfig",
    "TextEmbeddingConfig",
    "ModelPreset",
    "MockDiffusionDataset",
    "MockFluxDataset",
    "PreGeneratedMockFluxDataset",
    "MockFluxSchnellDataset",
    "PreGeneratedMockFluxSchnellDataset",
]
