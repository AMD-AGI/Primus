# Architecture overview

This document provides a detailed overview of the diffusion model architecture in Primus, including design decisions, directory structure, and implementation patterns.

---

## Table of contents

1. [Design philosophy](#design-philosophy)
2. [Directory structure](#directory-structure)
3. [Architectural decisions](#architectural-decisions)
4. [Component hierarchy](#component-hierarchy)
5. [Data flow](#data-flow)
6. [Comparison with alternative implementations](#comparison-with-alternative-implementations)

---

## Design philosophy

### Core principles

1. **Megatron-Core Native**: Built on Megatron-Core patterns and conventions
2. **Extensibility**: Easy to add new models (DiT, MovieGen, custom)
3. **Clarity**: Clear separation between shared and model-specific code
4. **Reusability**: Shared components across multiple models
5. **Performance**: Support for precalculated data and multi-GPU training

### Architectural advantages

| Aspect | Primus Design Choice |
|--------|---------------------|
| Model Location | `core/models/diffusion/` (Megatron-Core convention) |
| Layer Organization | Unified `TransformerBlock` with heterogeneous specs |
| Shared Code | Dedicated `common/` directory |
| Encoders | Hierarchical `encoders/{type}/{variant}/` |
| Energon | Shared `data/energon/` for all models |
| Mock Data | Synthetic providers under `data/synthetic/` |
| Framework | Pure Megatron (no PyTorch Lightning dependency) |

---

## Directory structure

### High-level organization

```
primus/backends/megatron/
├── core/models/diffusion/          # Core model implementations
├── training/diffusion/             # Training utilities (schedulers, etc.)
└── data/
    ├── energon/                    # Shared Energon infrastructure
    └── diffusion/                  # Diffusion-specific data
```

### Detailed breakdown

#### 1. Core models (`core/models/diffusion/`)

Following Megatron-Core convention (`megatron/core/models/gpt/`, `megatron/core/models/multimodal/`):

```
core/models/diffusion/
├── common/                         # Shared building blocks
│   ├── __init__.py
│   ├── config.py                   # BaseDiffusionConfig
│   ├── embeddings.py               # TimeStepEmbedder, Timesteps, MLPEmbedder
│   └── normalization.py            # RMSNorm, AdaLN, AdaLNContinuous
│
└── flux/                           # Flux-specific components
    ├── __init__.py
    ├── config.py                   # FluxConfig
    ├── model.py                    # Flux (main model class)
    ├── layer_spec.py               # MMDiTLayer, FluxSingleTransformerBlock
    ├── attention.py                # JointSelfAttention, FluxSingleAttention
    ├── layers.py                   # EmbedND (RoPE), embedders
    ├── checkpoint_converter.py     # HF <-> Megatron conversion
    └── utils.py                    # image position ids, helpers
```

**Rationale**:
- `common/`: shared building blocks (base config, timestep/positional embeddings, normalization) reusable across diffusion models
- `flux/`: Flux-specific code (model class, MMDiT and single-block layer specs, joint attention, RoPE embedders)
- Clear separation keeps model-specific code isolated and shared code reusable for future models

#### 2. Training (`training/diffusion/`)

```
training/diffusion/
├── __init__.py
└── schedulers/
    ├── __init__.py
    ├── base.py                     # BaseScheduler
    ├── flow_matching.py            # FlowMatchEulerDiscreteScheduler
    ├── ddpm.py                     # [Future] DDPMScheduler
    ├── edm.py                      # [Future] EDMScheduler
    └── euler.py                    # [Future] EulerDiscreteScheduler
```

**Rationale**:
- Schedulers define noise schedules and training targets
- Separate from model code for clarity
- Easy to add new schedulers (DDPM, EDM, etc.)

#### 3. Data pipeline (`data/`)

```
data/
├── energon/                        # Shared Energon utilities
│   └── __init__.py
├── dataloader.py                   # MegatronDataloaderWrapper (wraps any iterable)
│
└── diffusion/
    ├── __init__.py
    ├── encoders/                   # Hierarchical encoder registry
    │   ├── __init__.py
    │   ├── registry.py             # EncoderRegistry
    │   ├── image/
    │   │   └── vae/
    │   │       ├── __init__.py
    │   │       ├── autoencoder_kl.py   # AutoencoderKL
    │   │       └── vqvae.py            # VQVAE
    │   └── text/
    │       ├── t5/
    │       │   ├── __init__.py
    │       │   ├── t5_xxl.py           # T5-XXL
    │       │   └── t5_large.py         # T5-Large
    │       └── clip/
    │           ├── __init__.py
    │           ├── clip_l.py           # CLIP-L
    │           └── clip_h.py           # CLIP-H
    │
    ├── preprocessing/
    │   ├── __init__.py
    │   └── image/
    │       ├── __init__.py
    │       ├── transforms.py           # Resizing, normalization
    │       └── augmentation.py         # Data augmentation
    │
    └── task_encoders/
        ├── __init__.py
        └── image.py  # EncodedDiffusionTaskEncoder
```

**Rationale**:
- **Energon shared**: VLM and other models can reuse Energon utilities
- **Hierarchical encoders**: Organized by modality and variant type
- **Registry pattern**: Config-driven encoder selection
- **Preprocessing separated**: Clear pipeline stages

---

## Architectural decisions

### Decision 1: Models under `core/models/`

**Choice**: `primus/backends/megatron/core/models/diffusion/`
**Not**: `primus/backends/megatron/models/diffusion/`

**Reasoning**:
- Aligns with Megatron-Core structure
- Easier upstream tracking
- Clear that these are Megatron-Core compatible
- Consistent with existing Primus structure (`core/models/gpt/`)

### Decision 2: Separate `common/` directory

**Choice**: Shared components in `common/`
**Not**: Everything in `flux/` or flat structure

**Reasoning**:
- Standard approach puts shared code in model-specific directories
- Primus: `common/` makes it explicit what's shared
- Future DiT implementation trivial (reuse from `common/`)
- Clear contract: if in `common/`, must work for all models

**Shared Components**:
- `JointSelfAttention`: Used by DiT and Flux joint layers
- `FluxSingleAttention`: Used by DiT and Flux single layers
- `MMDiTLayer`: Joint (multimodal) transformer block
- `FluxSingleTransformerBlock`: Single-modality transformer block

**Flux-Only Components**:
- `EmbedND`: 3D RoPE position embedding (Flux-specific)
- `Flux` model class

### Decision 3: Hierarchical encoder structure

**Choice**: `encoders/image/vae/`, `encoders/text/t5/`, `encoders/text/clip/`
**Not**: Flat `encoders/conditioner.py`

**Reasoning**:
- Traditional approach uses flat structure with all encoders in one file
- User requirement: support 5+ variants per modality
- Registry pattern enables config-driven selection
- Easy to add new encoders without modifying existing files

**Registry Pattern**:
```python
from primus.backends.megatron.data.diffusion.encoders import get_encoder

# Config-driven selection
vae = get_encoder('autoencoder_kl', config=vae_config)
t5 = get_encoder('t5_xxl', config=t5_config)
clip = get_encoder('clip_l', config=clip_config)
```

### Decision 4: Shared Energon infrastructure

**Choice**: `data/energon/` for shared utilities
**Not**: Nested under `data/diffusion/`

**Reasoning**:
- Energon is general-purpose (VLM, diffusion, future models)
- Traditional approach nests under model-specific directories
- Megatron-LM has Energon at example level (not in core)
- Primus approach: shared infra + model-specific TaskEncoders

**Pattern**:
```python
# Shared: data/dataloader.py
# MegatronDataloaderWrapper wraps any iterable (Energon loader, PyTorch DataLoader, etc.)
wrapper = MegatronDataloaderWrapper(dataloader)

# Model-specific: data/diffusion/task_encoders/image.py
class EncodedDiffusionTaskEncoder:
    """Diffusion-specific encoding logic"""
    pass
```

### Decision 5: Model provider at adapter level

**Choice**: construct the model through provider functions in the Megatron trainer/adapter layer under `primus/backends/megatron/`, above `core/models/`.

**Reasoning**:
- Model providers are adapter/wrapper functions rather than a standalone module; they live alongside the trainers (for example `primus/backends/megatron/megatron_pretrain_trainer.py`)
- They sit above the core models to add Primus-specific functionality
- Example: wrap the model with a custom loss, precision handling, or checkpoint logic

### Decision 6: No PyTorch lightning

**Choice**: Pure Megatron patterns
**Not**: PyTorch Lightning DataModules

**Reasoning**:
- Primus doesn't use PyTorch Lightning
- Framework-specific implementations reduce flexibility
- Better integration with Megatron training loop
- Follows Megatron-LM's `MegatronDataloaderWrapper` wrapper pattern

---

## Component hierarchy

### 1. Model hierarchy

```
nn.Module (PyTorch)
└── MegatronModule
    └── DiffusionModule (abstract)
        └── Flux (concrete)
        ├── Joint layers: MMDiTLayer × num_joint_layers
        │   └── JointSelfAttention (shared)
        ├── Single layers: FluxSingleTransformerBlock × num_single_layers
        │   └── FluxSingleAttention (shared)
        └── Embeddings:
            ├── TimeStepEmbedder (shared)
            ├── MLPEmbedder (shared)
            └── EmbedND (Flux-specific, 3D RoPE)
```

### 2. Configuration hierarchy

```
TransformerConfig (Megatron-Core)
└── BaseDiffusionConfig
    └── FluxConfig
        ├── flux_535m() factory
        └── flux_12b() factory
```

### 3. Scheduler hierarchy

```
BaseScheduler (abstract)
├── FlowMatchEulerDiscreteScheduler (Flux)
├── DDPMScheduler (future)
├── EDMScheduler (future)
└── EulerDiscreteScheduler (future)
```

### 4. Encoder hierarchy

```
BaseEncoder (abstract)
├── ImageEncoder
│   └── VAE
│       ├── AutoencoderKL (Flux)
│       └── VQVAE (future)
└── TextEncoder
    ├── T5
    │   ├── T5-XXL (Flux)
    │   └── T5-Large (future)
    └── CLIP
        ├── CLIP-L (Flux)
        └── CLIP-H (future)
```

---

## Data flow

### Training pipeline

```
Raw Data (images + captions)
    ↓
[Optional] Precalculation
    ├─ VAE → latents [B, 64, H/8, W/8]
    ├─ T5-XXL → embeddings [B, 512, 4096]
    └─ CLIP-L → pooled [B, 768]
    ↓
WebDataset/Energon Format (.tar files)
    ↓
MegatronDataloaderWrapper (from data/dataloader.py)
    ↓
EncodedDiffusionTaskEncoder (from data/diffusion/task_encoders/)
    ├─ Load precalculated data, OR
    └─ Encode on-the-fly (slower)
    ↓
Training Batch:
    ├─ latents: [B, 64, H, W]
    ├─ t5_embeddings: [B, S, 4096]
    └─ clip_pooled: [B, 768]
    ↓
FlowMatchEulerDiscreteScheduler
    ├─ Sample timesteps: t ~ U(0, 1)
    ├─ Sample noise: ε ~ N(0, I)
    ├─ Add noise: x_t = (1-t)*ε + t*x_0
    └─ Compute target: v = x_0 - ε
    ↓
Flux Model Forward Pass
    ├─ Embed timesteps
    ├─ Embed pooled text (CLIP)
    ├─ Joint layers (process latents + T5 embeddings)
    ├─ Single layers (process latents only)
    └─ Output: v_pred [B, 64, H, W]
    ↓
Loss Computation: MSE(v_pred, v_target)
    ↓
Backward Pass & Optimizer Step
```

### Inference pipeline

```
Text Prompt
    ↓
Text Encoders
    ├─ T5-XXL → embeddings [1, S, 4096]
    └─ CLIP-L → pooled [1, 768]
    ↓
Initialize Noise: x_0 ~ N(0, I)
    ↓
Sampling Loop (t = 1.0 → 0.0)
    ├─ Model forward: v_t = Flux(x_t, t, embeddings)
    ├─ Update: x_{t-dt} = x_t + v_t * dt
    └─ Repeat until t = 0
    ↓
Latents: x_0 [1, 64, H, W]
    ↓
VAE Decoder
    ↓
Generated Image [1, 3, H*8, W*8]
```

---

## Comparison with alternative implementations

### Primus architectural advantages

#### 1. TransformerBlock architecture (Primus innovation)
- **Primus**: Unified `TransformerBlock` with heterogeneous layer specs
- **Others**: Separate `nn.ModuleList` containers for double/single blocks
- **Benefit**: Better PP slicing, unified checkpointing, future-proof

#### 2. Megatron-core native
- **Primus**: Pure Megatron-Core, no framework dependencies
- **Others**: Often integrated with PyTorch Lightning or other frameworks
- **Benefit**: Tighter integration, simpler training loops

#### 3. Checkpoint format
- **Primus**: Unified `transformer.layers.{0-56}` structure
- **Others**: Separate `double_blocks.{i}` and `single_blocks.{j}`
- **Benefit**: Simpler distributed checkpointing

#### 4. Encoder architecture
- **Primus**: Registry-based, hierarchical organization
- **Others**: Direct imports from monolithic files
- **Benefit**: Easy extensibility for new encoder variants

### File organization comparison

| Component | Standard Location | Primus Location | Improvement |
|-----------|------------------|-----------------|-------------|
| Model | `models/diffusion/flux/` | `core/models/diffusion/flux/` | Megatron-Core convention |
| Layers | Mixed locations | `common/` for shared, `flux/` for specific | Clear boundaries |
| Encoders | Single file | `data/diffusion/encoders/{type}/{variant}/` | Hierarchical, extensible |
| Tests | Mixed with code | `tests/unit_tests/backends/megatron/diffusion/` | Proper separation |

---

## Implementation status

### Core infrastructure ✅
- ✅ Directory structure
- ✅ Base classes (DiffusionModule, BaseDiffusionConfig, BaseScheduler)
- ✅ DiffusionModule with Megatron-Core integration
- ✅ FluxConfig with factory methods
- ✅ FlowMatchEulerDiscreteScheduler implementation
- ✅ Configuration files (YAML)
- ✅ Testing framework (290+ tests)
- ✅ Documentation structure

### Flux model implementation ✅
- ✅ Flux model architecture
- ✅ MMDiT layers and attention
- ✅ Embeddings (RoPE, timestep, vector)
- ✅ Encoder registry and loaders
- ✅ TaskEncoder for Energon
- ✅ Data pipeline

---

## Extension points

### Adding a new model (e.g., DiT)

1. **Create model directory**: `core/models/diffusion/dit/`
2. **Add config**: Extend `BaseDiffusionConfig`
3. **Implement model**: Extend `DiffusionModule` (which extends MegatronModule)
   - Inherit process group management
   - Get distributed checkpointing support
   - Access attention backend configuration
4. **Reuse shared components**: Import from `common/`
5. **Add tests**: `tests/unit_tests/backends/megatron/diffusion/test_dit_model.py`
6. **Update configs**: Add `dit_config.yaml`

### Adding a new encoder variant

1. **Create encoder file**: e.g., `data/diffusion/encoders/text/t5/t5_large.py`
2. **Implement encoder class**: Extend `BaseEncoder`
3. **Register**: Add to `ENCODER_REGISTRY`
4. **Add config**: Update `encoders.yaml`
5. **Add tests**: Test in `tests/unit_tests/backends/megatron/diffusion/data/encoders/`

### Adding a new scheduler

1. **Create scheduler file**: `training/diffusion/schedulers/ddpm.py`
2. **Implement**: Extend `BaseScheduler`
3. **Export**: Add to `__init__.py`
4. **Add tests**: `tests/unit_tests/backends/megatron/diffusion/training/test_scheduler.py`
5. **Document**: Update this file and README

---

## Performance considerations

### Memory optimization
- **Precalculated data**: 5-10x faster, lower memory
- **Frozen encoders**: Only train diffusion model
- **Gradient checkpointing**: Trade compute for memory
- **Mixed precision**: bf16 on MI300X (compatible with H100/A100)

### Multi-GPU scaling
- **Tensor Parallelism**: Split model across GPUs
- **Pipeline Parallelism**: Split layers across GPUs
- **Data Parallelism**: Replicate model, split data
- **Sequence Parallelism**: For very long sequences

### Best practices
1. Use precalculated mode for training
2. Freeze encoders (standard practice)
3. Use bf16 on modern hardware
4. Start with TP=1, PP=1, scale as needed
5. Profile before optimizing

---

## References

- **Megatron-Core**: [nvidia/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) transformer patterns
- **Flux**: [black-forest-labs/FLUX.1](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- **Flow Matching**: Rectified flow and flow matching papers
- **NeMo**: [nvidia/NeMo](https://github.com/NVIDIA/NeMo) - Alternative diffusion implementation

---

**Last Updated**: December 2025
