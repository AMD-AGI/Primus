# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Synthetic data generation for diffusion models.

This module provides mock datasets that generate synthetic data
with correct tensor shapes and formats for training and testing
without requiring real datasets or model weights.
"""

from .mock_datasets import (
    LatentConfig,
    MockDiffusionDataset,
    MockFluxDataset,
    MockFluxSchnellDataset,
    ModelPreset,
    PreGeneratedMockFluxDataset,
    PreGeneratedMockFluxSchnellDataset,
    TextEmbeddingConfig,
)

__all__ = [
    "MockDiffusionDataset",
    "MockFluxDataset",
    "PreGeneratedMockFluxDataset",
    "MockFluxSchnellDataset",
    "PreGeneratedMockFluxSchnellDataset",
    "LatentConfig",
    "TextEmbeddingConfig",
    "ModelPreset",
]
