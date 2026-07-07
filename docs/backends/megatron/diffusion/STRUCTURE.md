# Flux Diffusion Infrastructure - Directory Structure

**Created**: December 5, 2025
**Status**: ✓ Implementation Complete

## Overview

This document describes the directory structure created for Flux diffusion model support in Primus, following Megatron-Core conventions with production-ready enhancements.

---

## Directory Tree

```
Primus/
├── primus/backends/megatron/
│   ├── core/models/
│   │   ├── common/diffusion_module/         # DiffusionModule base class
│   │   │   └── diffusion_module.py
│   │   └── diffusion/                      # Diffusion models (Megatron-Core convention)
│   │       ├── common/                     # Shared components (MMDiT layers, attention)
│   │       │   ├── __init__.py
│   │       │   ├── config.py               # ✓ BaseDiffusionConfig
│   │       │   ├── embeddings.py           # ✓ TimeStepEmbedder, MLPEmbedder
│   │       │   └── normalization.py        # ✓ AdaLN, AdaLNContinuous, RMSNorm
│   │       ├── flux/                       # Flux-specific components
│   │       │   ├── __init__.py
│   │       │   ├── config.py               # ✓ FluxConfig (with factory methods)
│   │       │   ├── model.py                # ✓ Flux model
│   │       │   ├── layers.py               # ✓ EmbedND, embedders
│   │       │   ├── layer_spec.py           # ✓ get_flux_layer_spec, get_flux_*_spec_for_backend, MMDiTLayer
│   │       │   ├── attention.py            # ✓ JointSelfAttention, FluxSingleAttention
│   │       │   ├── utils.py                # ✓ generate_image_position_ids
│   │       │   ├── checkpoint_utils.py     # ✓ Checkpoint utilities
│   │       │   └── checkpoint_converter.py # ✓ HF <-> Megatron conversion
│   │       └── __init__.py
│   │
│   ├── training/diffusion/                 # Training utilities
│   │   ├── schedulers/
│   │   │   ├── __init__.py
│   │   │   ├── base.py                     # ✓ BaseScheduler
│   │   │   └── flow_matching.py            # ✓ FlowMatchEulerDiscreteScheduler
│   │   ├── noise_utils.py                  # ✓ apply_flow_matching_noise, apply_ddpm_noise
│   │   ├── loss_computation.py             # ✓ compute_flow_matching_loss, etc.
│   │   ├── timestep_sampling.py             # ✓ LogitNormalSampler, UniformSampler
│   │   └── __init__.py
│   │
│   └── data/
│       ├── energon/                        # Shared Energon infrastructure
│       │   └── __init__.py                 # ✓ Energon wrappers
│       │
│       └── diffusion/                      # Diffusion-specific data
│           ├── encoders/                   # Hierarchical encoder registry
│           │   ├── image/
│           │   │   ├── vae/                # VAE variants
│           │   │   │   └── __init__.py     # ✓ AutoencoderKL, VQVAE, etc.
│           │   │   └── __init__.py
│           │   ├── text/
│           │   │   ├── t5/                 # T5 variants
│           │   │   │   └── __init__.py     # ✓ T5-XXL, T5-Large, etc.
│           │   │   ├── clip/               # CLIP variants
│           │   │   │   └── __init__.py     # ✓ CLIP-L, CLIP-H, etc.
│           │   │   └── __init__.py
│           │   └── __init__.py             # ✓ EncoderRegistry
│           │
│           ├── preprocessing/
│           │   ├── image/
│           │   │   └── __init__.py         # ✓ Resizing, augmentation
│           │   └── __init__.py
│           │
│           ├── task_encoders/              # Energon TaskEncoders
│           │   ├── __init__.py
│           │   └── image.py                # ✓ EncodedDiffusionTaskEncoder, RawDiffusionTaskEncoder
│           │
│           └── __init__.py
│
├── primus/modules/trainer/megatron/
│   └── diffusion/                          # Diffusion trainer
│       └── __init__.py                     # ✓ DiffusionTrainer
│
├── primus/configs/models/megatron/
│   └── diffusion/                          # YAML configs
│       ├── __init__.py
│       ├── flux_535m.yaml                  # ✓ Flux 535M config
│       ├── flux_12b.yaml                   # ✓ Flux 12B config
│       └── encoders.yaml                   # ✓ Encoder configs
│
├── examples/megatron/
│   ├── diffusion/
│   │   └── README.md                       # ✓ Training guide (consolidated)
│   ├── configs/MI300X/diffusion/           # MI300X training configs
│   │   ├── flux_535m_pretrain.yaml
│   │   ├── flux_12b_fsdp2_energon_schnell_resample_local_spec.yaml
│   │   ├── flux_12b_ddp_energon_schnell_resample_te_spec_fp8.yaml
│   │   └── ...
│   ├── configs/MI355X/diffusion/           # MI355X training configs (mirrors MI300X + MXFP4/MLPerf)
│   │   ├── flux_12b_ddp_energon_schnell_resample_*.yaml
│   │   ├── flux_12b_fsdp2_energon_schnell_resample_*.yaml
│   │   └── ...
│   └── prepare.py
│
├── examples/run_pretrain.sh                # Main training script
│
├── tests/
│   ├── unit_tests/backends/megatron/diffusion/   # Unit test suite
│   │   ├── test_flux_model.py
│   │   ├── test_flux_config.py
│   │   ├── test_flux_layers.py
│   │   ├── test_flux_embeddings.py
│   │   ├── test_flux_normalization.py
│   │   ├── test_flux_utils.py
│   │   ├── test_flux_checkpoint_converter.py
│   │   ├── test_flux_checkpoint_utils.py
│   │   ├── test_flux_layer_spec_backend_selection.py
│   │   ├── test_flux_compile_checkpoint_keys.py
│   │   ├── training/
│   │   ├── data/
│   │   └── distributed/
│   └── integration_tests/backends/megatron/diffusion/
│       ├── data/
│       └── distributed/
│
└── docs/backends/megatron/
    └── diffusion/                          # Documentation
        ├── README.md                       # ✓ Overview
        ├── STRUCTURE.md                    # ✓ This file
        ├── architecture_overview.md        # ✓ Design details
        ├── data_preprocessing.md           # ✓ Data guide (includes Flux-specific section)
        ├── energon_integration.md          # ✓ Energon patterns
        ├── flux_architecture.md            # ✓ Flux deep dive
        ├── fp8_training.md                 # ✓ FP8 training guide
        ├── api_reference.md                # ✓ API documentation
        └── adding_new_models.md            # ✓ Extension guide
```

---

## Completed Components

### ✓ Base Classes

1. **DiffusionModule** (`core/models/common/diffusion_module/diffusion_module.py`)
   - Base class for all diffusion models (extends MegatronModule)
   - Provides Megatron-Core integration
   - Required methods: `forward()`
   - Loss computation: Use standalone functions from `loss_computation.py`
   - Utility methods: `get_num_params()`, `set_requires_grad()`

2. **BaseDiffusionConfig** (`common/config.py`)
   - Extends `megatron.core.transformer.transformer_config.TransformerConfig`
   - Common parameters: `in_channels`, `out_channels`, `patch_size`
   - Validation method for configuration integrity

3. **FluxConfig** (`flux/config.py`)
   - Flux-specific configuration
   - Parameters: `num_joint_layers`, `num_single_layers`, `context_dim`, `vec_in_dim`
   - Factory methods: `flux_535m()`, `flux_12b()`
   - 3D RoPE configuration: `axes_dim`, `theta`

3. **BaseScheduler** (`schedulers/base.py`)
   - Abstract base for diffusion schedulers
   - Required: `add_noise()`, `get_velocity_target()`, `sample_timesteps()`
   - Optional: `scale_model_input()`, `get_snr()`, `get_alpha()`, `get_sigma()`

4. **FlowMatchEulerDiscreteScheduler** (`schedulers/flow_matching.py`)
   - Concrete implementation for Flux
   - Linear interpolation: `x_t = (1-t)*noise + t*data`
   - Velocity target: `v = data - noise`

### ✓ Directory Structure

- **25 `__init__.py` files** with comprehensive docstrings
- **Multiple implementation files** (models, configs, schedulers, data pipeline)
- **Complete test suite** with fixtures and helpers

---

## Architectural Decisions

### 1. Models under `core/models/`
- Follows Megatron-Core convention (`megatron/core/models/gpt/`, etc.)
- Easier upstream tracking when Megatron-Core adds diffusion support

### 2. Shared Components in `common/`
- Standard approach stores shared code in model-specific directories
- Primus: `common/` for MMDiT layers, attention, shared utilities
- Flux-specific: Only `EmbedND` and Flux model class

### 3. Hierarchical Encoder Structure
- `encoders/image/vae/`, `encoders/text/t5/`, `encoders/text/clip/`
- Registry pattern for config-driven selection
- Easy to add new encoder variants (5+ planned per modality)

### 4. Shared Energon Infrastructure
- `data/energon/` for cross-model utilities (VLM, diffusion, future)
- `data/diffusion/task_encoders/` for diffusion-specific TaskEncoders
- Traditional approach nests Energon under model-specific directories

### 5. Mock Data in Tests
- `tests/fixtures/diffusion/` (not production code)
- Traditional approach mixes test utilities with production code
- Follows pytest best practices

### 6. No PyTorch Lightning
- Pure Megatron patterns (no PTL DataModules)
- Better integration with Megatron training loop

---

## Import Examples

```python
# Base classes
from primus.backends.megatron.core.models.diffusion.common import (
    BaseDiffusionConfig,
)

# Flux configuration
from primus.backends.megatron.core.models.diffusion.flux import FluxConfig

# Create configs
config_535m = FluxConfig.flux_535m()
config_12b = FluxConfig.flux_12b()

# Schedulers
from primus.backends.megatron.training.diffusion.schedulers import (
    BaseScheduler,
    FlowMatchEulerDiscreteScheduler,
)

# Create scheduler
scheduler = FlowMatchEulerDiscreteScheduler()
timesteps = scheduler.sample_timesteps(batch_size=8, device='cuda')
```

---

## Validation Status

✓ All Python files syntactically correct
✓ No linter errors detected
✓ All imports properly structured
✓ Comprehensive docstrings
✓ Copyright headers applied (AMD 2025, Apache 2.0)

---

## Files Summary

All infrastructure files, model implementations, data pipeline components, tests, and documentation are complete and ready for production use.

---

**End of Structure Document**
