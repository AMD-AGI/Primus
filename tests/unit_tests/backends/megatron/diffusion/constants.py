# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Constants for Flux diffusion model testing.

This module centralizes common dimensional and architectural constants
used across diffusion tests to reduce magic numbers and improve maintainability.
"""

# Model Architecture Dimensions
HIDDEN_DIM_FLUX = 3072  # Flux model hidden dimension
NUM_ATTENTION_HEADS_FLUX = 24  # Number of attention heads in Flux
HEAD_DIM_FLUX = 128  # Hidden dimension per attention head (3072 / 24)

# Encoder Dimensions
T5_XXL_EMBEDDING_DIM = 4096  # T5-XXL text encoder output dimension
CLIP_L_EMBEDDING_DIM = 768  # CLIP-L pooled embedding dimension
VAE_LATENT_CHANNELS = 16  # VAE encoder output channels for Flux

# Position Encoding
ROPE_THETA_DEFAULT = 10000  # Default theta value for RoPE (Rotary Position Embedding)
FLUX_AXES_DIM = (16, 56, 56)  # Default axes dimensions for Flux 3D RoPE

# Embedding Dimensions
TIMESTEP_EMBEDDING_DIM = 256  # Standard timestep embedding dimension

# Batch Sizes (count-based naming)
BATCH_SIZE_SINGLE = 1  # Single sample tests
BATCH_SIZE_PAIR = 2  # Paired sample tests
BATCH_SIZE_QUAD = 4  # Quad sample tests (most common)
BATCH_SIZE_OCTO = 8  # Octuple sample tests
BATCH_SIZE_HEX = 16  # Hexadecuple sample tests

# Image Dimensions (size-based naming)
IMG_SIZE_MICRO = 4  # Micro size for unit tests
IMG_SIZE_MINI = 8  # Mini size for quick tests
IMG_SIZE_TINY = 16  # Tiny size for fast tests
IMG_SIZE_SMALL = 32  # Small size for standard tests
IMG_SIZE_MEDIUM = 64  # Medium size
IMG_SIZE_LARGE = 128  # Large size

# Text Sequence Lengths
TEXT_SEQ_LEN_SHORT = 77  # Standard CLIP text length
TEXT_SEQ_LEN_MEDIUM = 128  # Medium sequence length
TEXT_SEQ_LEN_LONG = 256  # Long sequence length
TEXT_SEQ_LEN_XLARGE = 512  # Extra long sequence length

# Additional Sequence Lengths
SEQ_LEN_TINY = 100  # Small sequence for basic tests
ATTENTION_SEQ_LEN = 256  # Standard attention sequence length

# Position Encoding Dimensions
POS_GRID_SMALL = 4  # Small grid for position tests (4x3, 4x4)
CHANNEL_GROUPS = 16  # Channel groups for position IDs

# Training Hyperparameters (common test values)
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_GRADIENT_CLIP_NORM = 1.0
DEFAULT_GUIDANCE_SCALE = 3.5  # Typical classifier-free guidance scale

# Scheduler Parameters
DEFAULT_NUM_TRAIN_TIMESTEPS = 1000
DEFAULT_SHIFT = 1.0  # Default shift for flow matching scheduler

# Tensor Dimensions
TENSOR_CHANNELS_RGB = 3  # RGB channels for scheduler tests

# Iteration Counts
TRAINING_STEPS_FEW = 5  # Few training steps
TRAINING_STEPS_MODERATE = 10  # Moderate training steps
ACCUMULATION_STEPS = 4  # Gradient accumulation steps
