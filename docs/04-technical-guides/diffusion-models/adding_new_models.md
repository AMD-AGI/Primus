# Adding New Diffusion Models

This guide explains how to add new diffusion models to Primus, following the established patterns and architecture.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Step-by-Step Guide](#step-by-step-guide)
4. [Example: Adding DiT](#example-adding-dit)
5. [Testing Your Model](#testing-your-model)
6. [Best Practices](#best-practices)

---

## Overview

Adding a new diffusion model involves:
1. Creating model configuration
2. Implementing model class
3. Adding necessary layers
4. Creating data pipeline components
5. Writing tests
6. Updating documentation

**Time Estimate**: 5-10 days depending on model complexity

---

## Prerequisites

Before adding a new model, ensure you have:
- ✅ Understanding of the model architecture (paper, reference implementation)
- ✅ Access to pretrained weights (if applicable)
- ✅ Sample dataset for testing
- ✅ Familiarity with Primus diffusion architecture
- ✅ Development environment setup

**Required Reading**:
- [Architecture Overview](architecture_overview.md)
- [Data Preprocessing Guide](data_preprocessing.md)
- [Energon Integration](energon_integration.md)

---

## Step-by-Step Guide

### Step 1: Create Model Directory

Create a directory for your model under `core/models/diffusion/`:

```bash
mkdir -p primus/backends/megatron/core/models/diffusion/dit
cd primus/backends/megatron/core/models/diffusion/dit
```

Create files:
```bash
touch __init__.py
touch config.py
touch model.py
touch layers.py  # If model-specific layers needed
```

### Step 2: Implement Configuration

**File**: `config.py`

```python
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Configuration for DiT (Diffusion Transformer) model."""

from dataclasses import dataclass
from typing import Optional
from ..common.config import BaseDiffusionConfig


@dataclass
class DiTConfig(BaseDiffusionConfig):
    """
    DiT-specific configuration.

    DiT uses a standard transformer architecture for diffusion.
    """

    # Model identification
    model_type: str = "dit"

    # Architecture: Number of layers
    num_layers: int = 28  # DiT-XL/2 default

    # Architecture: Dimensions
    hidden_size: int = 1152
    num_attention_heads: int = 16

    # Input dimensions
    in_channels: int = 4  # VAE latent channels (standard SD VAE)

    # Context dimensions
    context_dim: int = 768  # CLIP text embedding dimension

    # Patchification
    patch_size: int = 2  # DiT uses 2x2 patches

    # Class conditioning (for conditional generation)
    num_classes: int = 1000  # ImageNet classes
    class_dropout_prob: float = 0.1

    # Adaptive LayerNorm (DiT-specific)
    use_adaptive_layernorm: bool = True

    def validate(self):
        """Validate DiT-specific configuration."""
        # Call parent validation
        super().validate()

        # DiT-specific validations
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")

        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {self.patch_size}")

        if self.num_classes < 0:
            raise ValueError(f"num_classes must be non-negative, got {self.num_classes}")

    @classmethod
    def dit_xl_2(cls, **kwargs):
        """
        Create configuration for DiT-XL/2.

        Args:
            **kwargs: Override default parameters

        Returns:
            DiTConfig instance
        """
        defaults = {
            'num_layers': 28,
            'hidden_size': 1152,
            'num_attention_heads': 16,
            'patch_size': 2,
        }
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def dit_l_2(cls, **kwargs):
        """Create configuration for DiT-L/2."""
        defaults = {
            'num_layers': 24,
            'hidden_size': 1024,
            'num_attention_heads': 16,
            'patch_size': 2,
        }
        defaults.update(kwargs)
        return cls(**defaults)
```

### Step 3: Implement Model Class

**File**: `model.py`

```python
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""DiT model implementation."""

import torch
import torch.nn as nn
from primus.backends.megatron.core.models.common.diffusion_module.diffusion_module import DiffusionModule
from megatron.core.process_groups_config import ProcessGroupCollection
from ..common.layers import MMDiTLayer  # Reuse shared components if applicable
from .config import DiTConfig


class DiT(DiffusionModule):
    """
    DiT (Diffusion Transformer) model.

    Reference: "Scalable Diffusion Models with Transformers" (Peebles & Xie, 2023)

    Note: Inherits from DiffusionModule for Megatron-Core integration
    (process groups, distributed checkpointing, attention backend config)
    """

    def __init__(
        self,
        config: DiTConfig,
        encoder_configs: Optional[Dict[str, Any]] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        """
        Initialize DiT model.

        Args:
            config: DiT configuration
            encoder_configs: Optional encoder configurations (VAE, T5, CLIP)
            pg_collection: Process group collection for distributed training
        """
        super().__init__(config, pg_collection=pg_collection, encoder_configs=encoder_configs)

        self.config = config

        # Input projection (patchify)
        self.input_proj = nn.Conv2d(
            config.in_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, (config.seq_length // config.patch_size) ** 2, config.hidden_size)
        )

        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.SiLU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size),
        )

        # Class embedding (for conditional generation)
        if config.num_classes > 0:
            self.class_embed = nn.Embedding(config.num_classes, config.hidden_size)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(config) for _ in range(config.num_layers)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.patch_size ** 2 * config.out_channels),
        )

        # Initialize weights
        self._init_weights()

    def forward(self, x, timesteps, context=None, class_labels=None, **kwargs):
        """
        Forward pass through DiT.

        Args:
            x: Noisy latents [B, C, H, W]
            timesteps: Diffusion timesteps [B]
            context: Text conditioning [B, S, D] (optional)
            class_labels: Class labels [B] (optional)

        Returns:
            Model prediction [B, C, H, W]
        """
        B, C, H, W = x.shape

        # Patchify input
        x = self.input_proj(x)  # [B, hidden_size, H/p, W/p]
        x = x.flatten(2).transpose(1, 2)  # [B, N, hidden_size]

        # Add positional embedding
        x = x + self.pos_embed

        # Embed timesteps
        t_emb = self.time_embed(self._timestep_embedding(timesteps))  # [B, hidden_size]

        # Embed class labels (if provided)
        if class_labels is not None and self.config.num_classes > 0:
            c_emb = self.class_embed(class_labels)  # [B, hidden_size]
            # Combine with timestep embedding
            cond = t_emb + c_emb
        else:
            cond = t_emb

        # Transformer blocks
        for block in self.blocks:
            x = block(x, cond, context)

        # Output projection
        x = self.output_proj(x)  # [B, N, p^2 * C]

        # Unpatchify
        x = self._unpatchify(x, H, W)  # [B, C, H, W]

        return x

            target: Ground truth target [B, C, H, W]

        Returns:
            Loss scalar
        """
        # Simple MSE loss (can be extended)
        loss = nn.functional.mse_loss(model_output, target)
        return loss

    def _timestep_embedding(self, timesteps):
        """Create sinusoidal timestep embeddings."""
        # Standard sinusoidal embedding
        half_dim = self.config.hidden_size // 2
        emb = torch.exp(
            -torch.arange(half_dim, device=timesteps.device) *
            (torch.log(torch.tensor(10000.0)) / (half_dim - 1))
        )
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

    def _unpatchify(self, x, H, W):
        """Convert patched tensor back to image."""
        p = self.config.patch_size
        h = H // p
        w = W // p
        x = x.reshape(x.shape[0], h, w, p, p, self.config.out_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.reshape(x.shape[0], self.config.out_channels, H, W)
        return x

    def _init_weights(self):
        """Initialize model weights."""
        # Standard initialization (customize as needed)
        pass


class DiTBlock(nn.Module):
    """DiT transformer block with adaptive LayerNorm."""

    def __init__(self, config):
        super().__init__()
        # Implementation details...
        pass

    def forward(self, x, cond, context=None):
        # Block forward pass
        pass
```

### Step 4: Add Model-Specific Layers (If Needed)

If your model has unique layers not shared with other models, add them to `layers.py`.

### Step 5: Export Model

**File**: `__init__.py`

```python
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""DiT model implementation."""

from .config import DiTConfig
from .model import DiT

__all__ = [
    'DiTConfig',
    'DiT',
]
```

### Step 6: Create Configuration Files

**File**: `primus/configs/models/megatron/diffusion/dit_xl_2.yaml`

```yaml
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

# DiT-XL/2 Configuration

model_type: dit

# Architecture: Layers
num_layers: 28

# Architecture: Dimensions
hidden_size: 1152
num_attention_heads: 16

# Input/Output Channels
in_channels: 4        # VAE latent channels
out_channels: 4

# Patchification
patch_size: 2

# Class Conditioning
num_classes: 1000     # ImageNet classes
class_dropout_prob: 0.1

# Adaptive LayerNorm
use_adaptive_layernorm: true

# Precision Settings
bf16: true
fp16: false

# Training
seq_length: 4096
micro_batch_size: 8
global_batch_size: 256
learning_rate: 1.0e-4
```

### Step 7: Write Tests

**File**: `tests/unit_tests/backends/megatron/diffusion/test_dit_model.py`

```python
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for DiT model."""

import pytest
import torch
from primus.backends.megatron.core.models.diffusion.dit import DiT, DiTConfig


class TestDiTConfig:
    """Tests for DiT configuration."""

    def test_dit_xl_2_factory(self):
        """Test DiT-XL/2 configuration factory method."""
        config = DiTConfig.dit_xl_2()

        assert config.model_type == "dit"
        assert config.num_layers == 28
        assert config.hidden_size == 1152
        assert config.num_attention_heads == 16
        assert config.patch_size == 2

    def test_dit_config_validation(self):
        """Test configuration validation."""
        config = DiTConfig.dit_xl_2()
        config.validate()  # Should not raise

        # Invalid configuration
        with pytest.raises(ValueError):
            config = DiTConfig(num_layers=-1)
            config.validate()


class TestDiTModel:
    """Tests for DiT model."""

    def test_dit_initialization(self):
        """Test DiT model initialization."""
        config = DiTConfig.dit_xl_2()
        model = DiT(config)

        assert model is not None
        assert isinstance(model, DiffusionModule)

    def test_dit_forward_shapes(self):
        """Test forward pass produces correct output shapes."""
        config = DiTConfig.dit_xl_2()
        model = DiT(config)

        batch_size = 2
        latents = torch.randn(batch_size, 4, 32, 32)
        timesteps = torch.rand(batch_size)
        class_labels = torch.randint(0, 1000, (batch_size,))

        output = model(latents, timesteps, class_labels=class_labels)

        assert output.shape == latents.shape

### Loss Computation

Use standalone loss functions from `loss_computation.py` instead of implementing loss as a method:

```python
from primus.backends.megatron.training.diffusion.loss_computation import compute_flow_matching_loss

# In your forward_step_func
target = noise - clean_latents
loss = compute_flow_matching_loss(prediction, clean_latents, noise)
```

Models do NOT implement loss as a method. Loss computation is:
- Separate from model architecture
- Reusable across models
- Testable independently

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### Step 8: Update Documentation

1. Add model to `README.md` supported models list
2. Update `architecture_overview.md` with model-specific details
3. Create model-specific training guide (e.g., `dit_training.md`)

### Step 9: Add Example Scripts

**File**: Use `examples/run_pretrain.sh` with appropriate config

```python
#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Training script for DiT model."""

import argparse
import yaml

from primus.backends.megatron.core.models.diffusion.dit import DiT, DiTConfig
from primus.backends.megatron.training.diffusion.schedulers import DDPMScheduler
from primus.backends.megatron.data.dataloader import MegatronDataloaderWrapper
# ... other imports


def main():
    # Use examples/run_pretrain.sh with config from examples/megatron/configs/MI300X/diffusion/
    # MegatronDataloaderWrapper wraps an existing iterable (from dataset provider):
    #   dataloader = MegatronDataloaderWrapper(energon_loader_or_pytorch_loader)
    # ...


if __name__ == "__main__":
    main()
```

---

## Example: Adding DiT

See the complete example in the step-by-step guide above.

**Key Files Created**:
1. `core/models/diffusion/dit/config.py` - DiTConfig
2. `core/models/diffusion/dit/model.py` - DiT model
3. `configs/models/megatron/diffusion/dit_xl_2.yaml` - Config file
4. `tests/unit_tests/backends/megatron/diffusion/test_dit_model.py` - Tests
5. `examples/run_pretrain.sh` - Use with config from `examples/megatron/configs/MI300X/diffusion/`

---

## Testing Your Model

### Unit Tests

Run tests to verify implementation:

```bash
# Run all DiT tests
pytest tests/unit_tests/backends/megatron/diffusion/test_dit_model.py -v

# Run specific test
pytest tests/unit_tests/backends/megatron/diffusion/test_dit_model.py::TestDiTModel::test_dit_forward_shapes -v
```

### Integration Tests

Test with actual data:

```bash
# Small dataset test
./examples/run_pretrain.sh --config examples/megatron/configs/MI300X/diffusion/flux_535m_pretrain.yaml
```

### Validation

Compare with reference implementation:
1. Load reference weights
2. Run same inputs through both models
3. Compare outputs (should match within tolerance)

---

## Best Practices

### 1. Code Organization
- ✅ Separate configuration from model code
- ✅ Reuse shared components from `common/`
- ✅ Keep model-specific code minimal
- ✅ Follow existing naming conventions

### 2. Configuration
- ✅ Extend `BaseDiffusionConfig`
- ✅ Add factory methods for common sizes
- ✅ Implement validation
- ✅ Document all parameters

### 3. Model Implementation
- ✅ Extend `DiffusionModule` from `primus.backends.megatron.core.models.common.diffusion_module.diffusion_module`
- ✅ Implement required method: `forward()`
- ✅ Use standalone loss functions from `loss_computation.py`
- ✅ Add comprehensive docstrings
- ✅ Use type hints
- ✅ Include `pg_collection` parameter in `__init__` for distributed training

### 4. Testing
- ✅ Test configuration validation
- ✅ Test model initialization
- ✅ Test forward pass shapes
- ✅ Test loss computation
- ✅ Test with mock data first

### 5. Documentation
- ✅ Update README with new model
- ✅ Document architecture specifics
- ✅ Provide usage examples
- ✅ Reference original paper

### 6. Performance
- ✅ Profile memory usage
- ✅ Optimize critical paths
- ✅ Support mixed precision
- ✅ Enable gradient checkpointing

---

## Common Pitfalls

### 1. Import Errors
❌ **Wrong**: Absolute imports
```python
from primus.backends.megatron.core.models.common.diffusion_module.diffusion_module import DiffusionModule
```

✅ **Right**: Relative imports
```python
from ...common.diffusion_module.diffusion_module import DiffusionModule
```

### 2. Configuration Validation
❌ **Wrong**: No validation
```python
class DiTConfig(BaseDiffusionConfig):
    pass  # No validation
```

✅ **Right**: Validate parameters
```python
def validate(self):
    super().validate()
    if self.num_layers <= 0:
        raise ValueError(f"num_layers must be positive")
```

### 3. Shape Mismatches
❌ **Wrong**: Assuming fixed shapes
```python
def forward(self, x):
    # Assumes x is always [B, 4, 32, 32]
    pass
```

✅ **Right**: Handle variable shapes
```python
def forward(self, x):
    B, C, H, W = x.shape
    # Handle any valid shape
    pass
```

### 4. Testing
❌ **Wrong**: No tests
```python
# Just implement and hope it works
```

✅ **Right**: Comprehensive tests
```python
def test_forward_shapes(self):
    # Test various input shapes
    pass
```

---

## Checklist

Before submitting your new model:

- [ ] Configuration class implemented and validated
- [ ] Model class extends `DiffusionModule`
- [ ] Required method implemented: `forward()`
- [ ] Loss computation uses standalone functions from `loss_computation.py`
- [ ] Process group support added (`pg_collection` parameter)
- [ ] YAML configuration files created
- [ ] Unit tests written and passing
- [ ] Integration tests run successfully
- [ ] Documentation updated
- [ ] Example training script provided
- [ ] Code follows Primus style guide
- [ ] No linter errors
- [ ] PR description includes architecture details

---

## Getting Help

If you encounter issues:
1. Review existing models (Flux) for patterns
2. Check documentation in `docs/04-technical-guides/diffusion-models/`
3. Run tests in debug mode: `pytest --pdb`
4. Consult architecture overview for design principles

---

**Last Updated**: December 2025
