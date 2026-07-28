# Flux model API reference

## Overview

This document provides comprehensive API reference for the Flux diffusion model implementation in Primus. Flux is a flow-based diffusion model that uses MMDiT (Multimodal Diffusion Transformer) architecture for high-quality text-to-image generation.

---

## Base classes

### DiffusionModule

**Location**: `primus/backends/megatron/core/models/common/diffusion_module/diffusion_module.py`

Base class for all diffusion models, providing Megatron-Core integration.

```python
from primus.backends.megatron.core.models.common.diffusion_module import DiffusionModule
```

**Key Features**:
- Process group management (TP, PP, CP, DP)
- Attention backend configuration
- Distributed checkpointing support
- Common loss computation utilities

**Inherited Methods** (available to all diffusion models):
- `get_num_params()` - Count trainable/total parameters
- `set_requires_grad()` - Freeze/unfreeze model
- `compute_diffusion_loss()` - Common loss helper (MSE, MAE, Huber)
- `sharded_state_dict()` - Distributed checkpointing

> Note: in-model encoder loading (`load_encoders()`, `get_encoder_by_type()`,
> `get_encoders_by_type()`) is not implemented and raises `NotImplementedError`.
> Encode VAE/T5/CLIP inputs via the offline diffusion preprocessing pipeline
> (`primus.backends.megatron.data.diffusion.preprocessing`) instead.

---

## Model architecture

### Flux class

**Location**: `primus/backends/megatron/core/models/diffusion/flux/model.py`

```python
from primus.backends.megatron.core.models.diffusion.flux.model import Flux
from primus.backends.megatron.core.models.diffusion.flux.config import FluxConfig

# Create model
config = FluxConfig.flux_535m()
model = Flux(config)
```

#### Architecture diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                          Flux Model                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Input Processing:                                               │
│  ┌────────────┐        ┌────────────┐                           │
│  │ Image      │──┐  ┌──│ Text       │                           │
│  │ Latents    │  │  │  │ Embeddings │                           │
│  │ [B,64,H,W] │  │  │  │ [B,S,4096] │                           │
│  └────────────┘  │  │  └────────────┘                           │
│                  ▼  ▼                                            │
│            ┌──────────────┐                                      │
│            │ Linear Embed │                                      │
│            └──────┬───────┘                                      │
│                   │ [B,seq,3072]                                 │
│                   ▼                                              │
│            ┌──────────────┐                                      │
│            │ 3D RoPE      │                                      │
│            │ Position Emb │                                      │
│            └──────┬───────┘                                      │
│                   │                                              │
│  Conditioning:    │                                              │
│  ┌────────────┐  │                                              │
│  │ Timestep   │──┼──┐                                           │
│  │ Embedding  │  │  │                                           │
│  └────────────┘  │  │                                           │
│  ┌────────────┐  │  │                                           │
│  │ CLIP       │──┼──┤                                           │
│  │ Pooled     │  │  │ vec_emb [B,3072]                         │
│  └────────────┘  │  │                                           │
│  ┌────────────┐  │  │                                           │
│  │ Guidance   │──┼──┘                                           │
│  │ (optional) │  │                                              │
│  └────────────┘  │                                              │
│                   │                                              │
│  Double Blocks    │                                              │
│  (Joint):         │                                              │
│  ┌────────────────▼──────────┐                                  │
│  │   MMDiTLayer x N_joint    │ N_joint = 1 (535M), 19 (12B)    │
│  │  ┌──────────────────────┐ │                                  │
│  │  │ Joint Self-Attention │ │ (image + text together)         │
│  │  └──────────────────────┘ │                                  │
│  │  ┌──────────────────────┐ │                                  │
│  │  │ Image MLP            │ │                                  │
│  │  └──────────────────────┘ │                                  │
│  │  ┌──────────────────────┐ │                                  │
│  │  │ Text MLP             │ │                                  │
│  │  └──────────────────────┘ │                                  │
│  └────────────┬──────────────┘                                  │
│               │                                                  │
│  Single Blocks│                                                  │
│  (Combined):  │                                                  │
│  ┌────────────▼──────────────┐                                  │
│  │ FluxSingleTransformer     │ N_single = 1 (535M), 38 (12B)   │
│  │       Block x N_single    │                                  │
│  │  ┌──────────────────────┐ │                                  │
│  │  │ Self-Attention       │ │ (image + text concatenated)     │
│  │  └──────────────────────┘ │                                  │
│  │  ┌──────────────────────┐ │                                  │
│  │  │ MLP                  │ │                                  │
│  │  └──────────────────────┘ │                                  │
│  └────────────┬──────────────┘                                  │
│               │ (extract image tokens)                           │
│               ▼                                                  │
│  Output Processing:                                              │
│  ┌────────────────────────┐                                     │
│  │ AdaLNContinuous        │                                     │
│  │ (timestep conditioned) │                                     │
│  └────────────┬───────────┘                                     │
│               ▼                                                  │
│  ┌────────────────────────┐                                     │
│  │ Linear Projection      │                                     │
│  └────────────┬───────────┘                                     │
│               ▼                                                  │
│         [B,64,H,W]                                               │
│       Predicted Velocity                                         │
└─────────────────────────────────────────────────────────────────┘
```

#### Parameter counts

| Variant | Joint Layers | Single Layers | Total Parameters | Use Case |
|---------|-------------|---------------|------------------|----------|
| Flux 535M | 1 | 1 | ~535 million | Testing, debugging, prototyping |
| Flux 12B | 19 | 38 | ~12 billion | Production training |

---

## Configuration

### FluxConfig

**Location**: `primus/backends/megatron/core/models/diffusion/flux/config.py`

Complete configuration class for Flux models, inheriting from `BaseDiffusionConfig`.

#### Key parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_joint_layers` | int | 19 | Number of joint (MMDiT) transformer layers |
| `num_single_layers` | int | 38 | Number of single transformer layers |
| `hidden_size` | int | 3072 | Hidden dimension size |
| `num_attention_heads` | int | 24 | Number of attention heads |
| `in_channels` | int | 64 | Input channels (VAE latent dimension) |
| `context_dim` | int | 4096 | Text context dimension (T5-XXL) |
| `vec_in_dim` | int | 768 | Vector input dimension (CLIP pooled) |
| `model_channels` | int | 256 | Channels for timestep embedding |
| `guidance_embed` | bool | False | Enable guidance embedding for CFG |
| `guidance_scale` | float | 3.5 | Guidance scale for classifier-free guidance |
| `theta` | int | 10000 | Base for RoPE frequency computation |
| `axes_dim` | tuple | (16, 56, 56) | Dimensions for 3D RoPE axes |
| `patch_size` | int | 1 | Patch size for image tokens |
| `add_qkv_bias` | bool | True | Add bias to QKV projections |
| `rotary_interleaved` | bool | True | Interleave RoPE dimensions |
| `layernorm_epsilon` | float | 1e-6 | Epsilon for layer normalization |
| `hidden_dropout` | float | 0.0 | Hidden layer dropout rate |
| `attention_dropout` | float | 0.0 | Attention dropout rate |

#### Configuration examples

**Flux 535M (Testing)**:
```python
config = FluxConfig.flux_535m()
# Equivalent to:
config = FluxConfig(
    num_joint_layers=1,
    num_single_layers=1,
    hidden_size=3072,
    num_attention_heads=24,
)
```

**Flux 12B (Production)**:
```python
config = FluxConfig.flux_12b()
# Equivalent to:
config = FluxConfig(
    num_joint_layers=19,
    num_single_layers=38,
    hidden_size=3072,
    num_attention_heads=24,
)
```

**Custom Configuration**:
```python
config = FluxConfig(
    num_joint_layers=4,
    num_single_layers=8,
    hidden_size=2048,
    num_attention_heads=16,
    guidance_embed=True,
    patch_size=2,
)
```

---

## Components API

### Embeddings

#### TimeStepEmbedder

**Location**: `primus/backends/megatron/core/models/diffusion/common/embeddings.py`

Converts scalar timesteps to high-dimensional embeddings using sinusoidal encoding.

```python
from primus.backends.megatron.core.models.diffusion.common.embeddings import TimeStepEmbedder

embedder = TimeStepEmbedder(embedding_dim=256, hidden_dim=3072)
timesteps = torch.tensor([0, 100, 500, 999])  # [B]
t_emb = embedder(timesteps)  # [B, 3072]
```

**Input**: Timesteps [B] in range [0, 1000]
**Output**: Embeddings [B, hidden_dim]

#### MLPEmbedder

Embeds vector conditioning (e.g., CLIP pooled embeddings) via 2-layer MLP.

```python
from primus.backends.megatron.core.models.diffusion.common.embeddings import MLPEmbedder

embedder = MLPEmbedder(in_dim=768, hidden_dim=3072)
clip_pooled = torch.randn(4, 768)  # [B, 768]
embedded = embedder(clip_pooled)  # [B, 3072]
```

**Input**: Vectors [B, in_dim]
**Output**: Embeddings [B, hidden_dim]

---

### Normalization

#### AdaLN (adaptive layer normalization)

**Location**: `primus/backends/megatron/core/models/diffusion/common/normalization.py`

Applies layer normalization conditioned on timestep embeddings.

```python
from megatron.core.transformer.transformer_config import TransformerConfig
from primus.backends.megatron.core.models.diffusion.common.normalization import AdaLN

config = TransformerConfig(hidden_size=3072)
adaln = AdaLN(config, n_adaln_chunks=6)

timestep_emb = torch.randn(4, 3072)
shift, scale, gate, shift_mlp, scale_mlp, gate_mlp = adaln(timestep_emb)
# Each output: [B, 3072]
```

**Methods**:
- `forward(timestep_emb)`: Generate modulation parameters
- `modulate(x, shift, scale)`: Apply adaptive modulation
- `scale_add(residual, x, gate)`: Gated residual addition

#### AdaLNContinuous

Continuous variant of AdaLN for Flux output normalization.

```python
from primus.backends.megatron.core.models.diffusion.common.normalization import AdaLNContinuous

adaln = AdaLNContinuous(config, conditioning_embedding_dim=3072)
x = torch.randn(4, 256, 3072)  # [B, seq, hidden]
cond = torch.randn(4, 3072)  # [B, cond_dim]
x_norm = adaln(x, cond)  # [B, seq, hidden]
```

#### RMSNorm

Root Mean Square Layer Normalization (simpler, faster than LayerNorm).

```python
from primus.backends.megatron.core.models.diffusion.common.normalization import RMSNorm

norm = RMSNorm(hidden_size=3072)
x = torch.randn(4, 256, 3072)
x_norm = norm(x)
```

---

### Position embeddings

#### EmbedND (3D RoPE)

**Location**: `primus/backends/megatron/core/models/diffusion/flux/layers.py`

Multi-dimensional Rotary Position Embedding for image patches.

```python
from primus.backends.megatron.core.models.diffusion.flux.layers import (
    EmbedND,
)
from primus.backends.megatron.core.models.diffusion.flux.utils import (
    generate_image_position_ids,
)

# Initialize
embed_nd = EmbedND(dim=3072, theta=10000, axes_dim=[16, 56, 56])

# Generate position IDs for 56x56 image patches
batch_size = 2
height, width = 112, 112  # Unpacked dimensions (56*2, 56*2)
img_ids = generate_image_position_ids(batch_size, height, width)
# img_ids: [B, H*W/4, 3] where dimension 0 is always 0

# Get RoPE frequencies
rope_freqs = embed_nd(img_ids)  # [3, B, H*W, 3072]
```

**Axes**:
- Axis 0: Channel groups (16 for 64 channels)
- Axis 1: Height positions (56 for 1024px image)
- Axis 2: Width positions (56 for 1024px image)

---

### Attention mechanisms

#### JointSelfAttention

**Location**: `primus/backends/megatron/core/models/diffusion/flux/attention.py`

Joint attention over image and text tokens (MMDiT architecture).

```python
from primus.backends.megatron.core.models.diffusion.flux.attention import (
    JointSelfAttention,
    JointSelfAttentionSubmodules,
)

submodules = JointSelfAttentionSubmodules(...)
joint_attn = JointSelfAttention(config, submodules, layer_number=0)

# Forward
img_tokens = torch.randn(3136, 2, 3072)  # [seq_img, B, hidden]
txt_tokens = torch.randn(512, 2, 3072)   # [seq_txt, B, hidden]
img_out, txt_out = joint_attn(
    img_tokens,
    attention_mask=None,
    additional_hidden_states=txt_tokens,
)
```

**Input**:
- `hidden_states`: Image tokens [seq_img, B, hidden]
- `additional_hidden_states`: Text tokens [seq_txt, B, hidden]

**Output**: Tuple of (img_output, txt_output)

#### FluxSingleAttention

Single-stream self-attention for image tokens only.

```python
from primus.backends.megatron.core.models.diffusion.flux.attention import FluxSingleAttention

single_attn = FluxSingleAttention(config, submodules, layer_number=0)

img_tokens = torch.randn(3136, 2, 3072)
output = single_attn(img_tokens, attention_mask=None)
```

---

### Layer specifications

#### MMDiTLayer

**Location**: `primus/backends/megatron/core/models/diffusion/flux/layer_spec.py`

Joint image-text transformer block.

```python
from primus.backends.megatron.core.models.diffusion.flux.layer_spec import (
    MMDiTLayer,
    get_flux_double_transformer_spec_for_backend,
)

# Using factory function (recommended)
spec = get_flux_double_transformer_spec_for_backend(backend)
mmdit_layer = MMDiTLayer(
    config=config,
    submodules=spec.submodules,
    layer_number=0,
)

# Forward
img_tokens = torch.randn(3136, 2, 3072)
txt_tokens = torch.randn(512, 2, 3072)
emb = torch.randn(2, 3072)
img_out, txt_out = mmdit_layer(img_tokens, txt_tokens, emb=emb)
```

#### FluxSingleTransformerBlock

Image-only transformer block.

```python
from primus.backends.megatron.core.models.diffusion.flux.layer_spec import (
    FluxSingleTransformerBlock,
    get_flux_single_transformer_spec_for_backend,
)

spec = get_flux_single_transformer_spec_for_backend(backend)
single_block = FluxSingleTransformerBlock(
    config=config,
    submodules=spec.submodules,
    layer_number=0,
)

# Forward
img_tokens = torch.randn(3136, 2, 3072)
emb = torch.randn(2, 3072)
output, _ = single_block(img_tokens, emb=emb)
```

---

## Training utilities

### Noise application

**Location**: `primus/backends/megatron/training/diffusion/noise_utils.py`

Pure functions for applying noise according to different diffusion forward processes.

#### apply_flow_matching_noise()

```python
from primus.backends.megatron.training.diffusion.noise_utils import apply_flow_matching_noise

clean_latents = torch.randn(2, 16, 64, 64)
noise = torch.randn(2, 16, 64, 64)
sigma = torch.tensor([0.3, 0.7]).reshape(2, 1, 1, 1)

noisy = apply_flow_matching_noise(clean_latents, noise, sigma)
# Formula: noisy = (1 - sigma) * clean + sigma * noise
```

**Parameters**:
- `clean_latents` (Tensor): Clean latents [any shape]
- `noise` (Tensor): Sampled noise [same shape as clean_latents]
- `sigma` (Tensor): Noise schedule values [broadcast compatible], range [0, 1]

**Returns**: Noisy latents [same shape as clean_latents]

**Reference**: [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)

#### apply_ddpm_noise()

```python
from primus.backends.megatron.training.diffusion.noise_utils import apply_ddpm_noise

clean = torch.randn(2, 3, 256, 256)
noise = torch.randn(2, 3, 256, 256)
alpha_bar = torch.tensor([0.9, 0.5]).reshape(2, 1, 1, 1)

noisy = apply_ddpm_noise(clean, noise, alpha_bar)
# Formula: noisy = sqrt(alpha_bar) * clean + sqrt(1 - alpha_bar) * noise
```

**Parameters**:
- `clean_latents` (Tensor): Clean latents
- `noise` (Tensor): Sampled noise (same shape)
- `alpha_bar` (Tensor): Cumulative product of alphas, range (0, 1]

**Returns**: Noisy latents

**Reference**: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)

---

### Loss computation

**Location**: `primus/backends/megatron/training/diffusion/loss_computation.py`

Reusable loss computation logic for different diffusion training objectives.

#### compute_flow_matching_loss()

```python
from primus.backends.megatron.training.diffusion.loss_computation import compute_flow_matching_loss

prediction = torch.randn(2, 16, 64, 64)  # Model output
clean = torch.randn(2, 16, 64, 64)
noise = torch.randn(2, 16, 64, 64)

loss = compute_flow_matching_loss(prediction, clean, noise)
# Formula: target = noise - clean
#          loss = MSE(prediction, target)
```

**Parameters**:
- `prediction` (Tensor): Model output (predicted velocity) [any shape]
- `clean_latents` (Tensor): Original clean latents [same shape]
- `noise` (Tensor): Sampled noise [same shape]

**Returns**: Scalar loss value (mean squared error)

**Used by**: Flux, SD3, video models

#### compute_epsilon_loss()

```python
from primus.backends.megatron.training.diffusion.loss_computation import compute_epsilon_loss

prediction = torch.randn(2, 3, 256, 256)
noise = torch.randn(2, 3, 256, 256)

loss = compute_epsilon_loss(prediction, noise)
# Formula: loss = MSE(prediction, noise)
```

**Parameters**:
- `prediction` (Tensor): Model output (predicted noise)
- `noise` (Tensor): Sampled noise (ground truth)

**Returns**: Scalar loss value

**Used by**: DDPM and older models

#### compute_v_prediction_loss()

```python
from primus.backends.megatron.training.diffusion.loss_computation import compute_v_prediction_loss

prediction = torch.randn(2, 16, 64, 64)
clean = torch.randn(2, 16, 64, 64)
noise = torch.randn(2, 16, 64, 64)
sigma = torch.tensor([0.3, 0.7]).reshape(2, 1, 1, 1)

loss = compute_v_prediction_loss(prediction, clean, noise, sigma)
# Formula: v = sigma * noise - (1 - sigma) * clean
#          loss = MSE(prediction, v)
```

**Parameters**:
- `prediction` (Tensor): Model output
- `clean_latents` (Tensor): Clean latents
- `noise` (Tensor): Sampled noise
- `sigma` (Tensor): Noise schedule values [broadcast compatible]

**Returns**: Scalar loss value

**Reference**: [Progressive Distillation for Fast Sampling](https://arxiv.org/abs/2202.00512)

---

### Timestep sampling

**Location**: `primus/backends/megatron/training/diffusion/timestep_sampling.py`

Sampling strategies for training timesteps (hyperparameter optimization, separate from inference).

#### LogitNormalSampler

```python
from primus.backends.megatron.training.diffusion.timestep_sampling import LogitNormalSampler

sampler = LogitNormalSampler(mean=0.0, std=1.0)
timesteps, sigmas = sampler.sample(
    batch_size=32,
    device='cuda',
    scheduler=flow_scheduler
)
```

**Description**: Logit-normal distribution sampling, emphasizes boundary timesteps (t≈0 and t≈1000).

**Parameters**:
- `mean` (float): Mean of normal distribution (default: 0.0)
- `std` (float): Standard deviation (default: 1.0)

**Returns**: Tuple of (timesteps [B], sigmas [B])

**Reference**: [Stable Diffusion 3](https://arxiv.org/abs/2403.03206v1), Section 3.1

**Used by**: Flux, SD3

#### UniformSampler

```python
from primus.backends.megatron.training.diffusion.timestep_sampling import UniformSampler

sampler = UniformSampler()
timesteps, sigmas = sampler.sample(batch_size=32, device='cuda', scheduler=flow_scheduler)
```

**Description**: Uniform timestep sampling (baseline approach for comparison).

**Returns**: Tuple of (timesteps [B], sigmas [B])

#### ModeSampler

```python
from primus.backends.megatron.training.diffusion.timestep_sampling import ModeSampler

sampler = ModeSampler(mode_scale=1.29)
timesteps, sigmas = sampler.sample(batch_size=32, device='cuda', scheduler=flow_scheduler)
```

**Description**: Mode-based sampling from SD3 paper (alternative to logit-normal).

**Parameters**:
- `mode_scale` (float): Scaling factor (default: 1.29 from SD3 paper)

**Returns**: Tuple of (timesteps [B], sigmas [B])

#### create_timestep_sampler()

```python
from primus.backends.megatron.training.diffusion.timestep_sampling import create_timestep_sampler

# Factory function for easy experimentation
sampler = create_timestep_sampler("logit_normal", mean=0.0, std=1.0)
sampler = create_timestep_sampler("uniform")
sampler = create_timestep_sampler("mode", mode_scale=1.5)
```

**Parameters**:
- `strategy` (str): Sampling strategy ("logit_normal", "uniform", "mode")
- `**kwargs`: Additional arguments for the sampler

**Returns**: TimestepSampler instance

---

### Training workflow example

Complete training step using the utilities:

```python
import torch
from torch.optim import AdamW
from primus.backends.megatron.core.models.diffusion.flux import Flux, FluxConfig
from primus.backends.megatron.training.diffusion.noise_utils import apply_flow_matching_noise
from primus.backends.megatron.training.diffusion.loss_computation import compute_flow_matching_loss
from primus.backends.megatron.training.diffusion.timestep_sampling import LogitNormalSampler
from primus.backends.megatron.training.diffusion.schedulers.flow_match_euler import (
    FlowMatchEulerDiscreteScheduler
)

# Setup
config = FluxConfig.flux_535m()
model = Flux(config).cuda()
optimizer = AdamW(model.parameters(), lr=1e-4)

# Initialize scheduler and timestep sampler
scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
timestep_sampler = LogitNormalSampler(mean=0.0, std=1.0)

# Training loop
for batch in dataloader:
    clean_latents = batch['latents'].cuda()  # [B, 16, 64, 64]
    txt_embeddings = batch['text'].cuda()     # [B, 512, 4096]
    clip_pooled = batch['clip'].cuda()        # [B, 768]

    batch_size = clean_latents.shape[0]

    # 1. Sample timesteps
    timesteps, sigmas = timestep_sampler.sample(
        batch_size=batch_size,
        device='cuda',
        scheduler=scheduler
    )

    # 2. Sample noise
    noise = torch.randn_like(clean_latents)

    # 3. Apply noise
    sigma_reshaped = sigmas.view(-1, 1, 1, 1)
    noisy_latents = apply_flow_matching_noise(clean_latents, noise, sigma_reshaped)

    # 4. Prepare position IDs
    img_ids = generate_image_position_ids(batch_size, 128, 128).cuda()
    txt_ids = torch.zeros(batch_size, 512, 3).cuda()

    # 5. Forward pass
    predicted_velocity = model(
        img=noisy_latents,
        txt=txt_embeddings,
        y=clip_pooled,
        timesteps=sigmas,
        img_ids=img_ids,
        txt_ids=txt_ids,
    )

    # 6. Compute loss
    loss = compute_flow_matching_loss(predicted_velocity, clean_latents, noise)

    # 7. Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Step {step}, Loss: {loss.item():.4f}")
```

---

## Usage examples

### Basic inference

```python
import torch
from primus.backends.megatron.core.models.diffusion.flux.model import Flux
from primus.backends.megatron.core.models.diffusion.flux.config import FluxConfig
from primus.backends.megatron.core.models.diffusion.flux.utils import generate_image_position_ids

# 1. Setup model
config = FluxConfig.flux_535m()
model = Flux(config)
model.eval()

# 2. Prepare inputs
batch_size = 1
img_latents = torch.randn(batch_size, 64, 128, 128)  # VAE latents
txt_embeddings = torch.randn(batch_size, 512, 4096)  # T5-XXL embeddings
clip_pooled = torch.randn(batch_size, 768)  # CLIP-L pooled
timesteps = torch.tensor([0.5])  # Diffusion timestep [0, 1]

# 3. Generate position IDs
img_ids = generate_image_position_ids(batch_size, 256, 256)
txt_ids = torch.zeros(batch_size, 512, 3)

# 4. Forward pass
with torch.no_grad():
    predicted_velocity = model(
        img=img_latents,
        txt=txt_embeddings,
        y=clip_pooled,
        timesteps=timesteps,
        img_ids=img_ids,
        txt_ids=txt_ids,
    )

print(f"Output shape: {predicted_velocity.shape}")  # [1, 64, 128, 128]
```

### Training step

```python
import torch
from torch.optim import AdamW

# Setup
config = FluxConfig.flux_535m()
model = Flux(config)
model.train()

optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

# Prepare batch
batch_size = 4
img = torch.randn(batch_size, 64, 128, 128)
txt = torch.randn(batch_size, 512, 4096)
y = torch.randn(batch_size, 768)
timesteps = torch.rand(batch_size)

img_ids = generate_image_position_ids(batch_size, 256, 256)
txt_ids = torch.zeros(batch_size, 512, 3)

# Add noise for flow matching
original = img.clone()
noise = torch.randn_like(img)
t = timesteps.view(-1, 1, 1, 1)
noisy_img = original + noise * t

# Velocity target for flow matching
velocity_target = noise - original

# Training step
optimizer.zero_grad()

# Forward pass
output = model(noisy_img, txt, y, timesteps, img_ids, txt_ids)

# Loss computation (using standalone function)
from primus.backends.megatron.training.diffusion.loss_computation import compute_flow_matching_loss
target = noise - clean_latents
loss = compute_flow_matching_loss(output, clean_latents, noise)

# Backward pass
loss.backward()

# Optimizer step
optimizer.step()

print(f"Loss: {loss.item():.4f}")
```

### With guidance (classifier-free guidance)

```python
# Enable guidance in config
config = FluxConfig.flux_535m(guidance_embed=True)
model = Flux(config)
model.eval()

# Prepare inputs with guidance
guidance_scale = torch.tensor([3.5])  # Typical guidance scale

with torch.no_grad():
    output = model(
        img=img_latents,
        txt=txt_embeddings,
        y=clip_pooled,
        timesteps=timesteps,
        img_ids=img_ids,
        txt_ids=txt_ids,
        guidance=guidance_scale,  # Add guidance
    )
```

### Different resolutions

```python
# Flux can handle different resolutions
resolutions = [
    (64, 64),    # 512x512 pixels
    (128, 128),  # 1024x1024 pixels
    (192, 192),  # 1536x1536 pixels
]

for height, width in resolutions:
    img = torch.randn(1, 64, height, width)
    img_ids = generate_image_position_ids(1, height, width)

    with torch.no_grad():
        output = model(img, txt, y, timesteps, img_ids, txt_ids)

    assert output.shape == img.shape
```

---

## Methods reference

### Flux.forward()

```python
def forward(
    self,
    img: Tensor,                    # [B, C, H, W] Image latents from VAE
    txt: Tensor,                    # [B, S_txt, D_txt] T5-XXL embeddings
    y: Tensor,                      # [B, D_pool] CLIP pooled embeddings
    timesteps: Tensor,              # [B] Timesteps in [0, 1]
    img_ids: Tensor,                # [B, H*W, 3] Image position IDs
    txt_ids: Tensor,                # [B, S_txt, 3] Text position IDs
    guidance: Optional[Tensor] = None,  # [B] Guidance scale
    controlnet_double_block_samples: Optional[Tensor] = None,
    controlnet_single_block_samples: Optional[Tensor] = None,
) -> Tensor:                        # Returns: [B, C, H, W] Predicted velocity
```

## Loss computation

Loss is computed using standalone functions from `loss_computation.py`:

### compute_flow_matching_loss()

```python
from primus.backends.megatron.training.diffusion.loss_computation import compute_flow_matching_loss

def compute_flow_matching_loss(
    prediction: Tensor,                 # [any shape] Model prediction
    clean_latents: Tensor,              # [same shape] Original clean latents
    noise: Tensor,                      # [same shape] Sampled noise
) -> Tensor:                            # Returns: Scalar loss
```

### Flux.get_num_params()

```python
def get_num_params(
    self,
    trainable_only: bool = True,
) -> int:                           # Returns: Number of parameters
```

---

## Testing

### Running tests

```bash
# Run all Flux tests
pytest tests/unit_tests/backends/megatron/diffusion/ -v

# Run specific test files
pytest tests/unit_tests/backends/megatron/diffusion/test_flux_embeddings.py -v
pytest tests/unit_tests/backends/megatron/diffusion/test_flux_normalization.py -v
pytest tests/unit_tests/backends/megatron/diffusion/test_flux_config.py -v
pytest tests/unit_tests/backends/megatron/diffusion/test_flux_layers.py -v
pytest tests/unit_tests/backends/megatron/diffusion/test_flux_model.py -v
pytest tests/unit_tests/backends/megatron/diffusion/test_flux_layer_spec_backend_selection.py -v

# Run with coverage
pytest tests/unit_tests/backends/megatron/diffusion/ --cov=primus.backends.megatron.core.models.diffusion --cov-report=html
```

### Test coverage

- **Component tests**: 80+ tests covering all components
- **Model tests**: 17+ tests for full model
- **Integration tests**: 11+ tests for complete workflows

---

## Performance considerations

### Memory usage

| Configuration | Model Weights | Training (bf16) | Training (fp32) |
|--------------|---------------|-----------------|-----------------|
| Flux 535M | ~2 GB | ~6-8 GB | ~10-12 GB |
| Flux 12B | ~24 GB | ~60-80 GB | ~100-120 GB |

### Throughput (estimated)

On MI300X (192GB):
- **Flux 535M**: ~5-10 samples/sec (depends on resolution)
- **Flux 12B**: ~0.5-1 samples/sec (requires multi-GPU)

### Optimization tips

1. **Use mixed precision**: `torch.autocast(device_type='cuda', dtype=torch.bfloat16)`
2. **Enable CUDA graphs**: Set `enable_cuda_graph=True` in config
3. **Use Transformer Engine**: Automatically used with factory functions
4. **Gradient checkpointing**: Can be enabled for memory savings

---

## Common issues

### Issue: Import errors

```python
# Old import style
from some_other_library import Flux

# Correct Primus import
from primus.backends.megatron.core.models.diffusion.flux.model import Flux
```

### Issue: Position ID shape mismatch

```python
# Position IDs must be [B, seq, 3] for 3D RoPE
img_ids = generate_image_position_ids(batch_size, height, width)
# Not: img_ids = torch.randn(batch_size, height * width, 2)  # Wrong!
```

### Issue: Timestep range

```python
# Flux expects timesteps in [0, 1]
timesteps = torch.rand(batch_size)  # Correct: [0, 1]
# Not: timesteps = torch.randint(0, 1000, (batch_size,))  # Wrong range!
```

---

## API compatibility

### Primus architecture features

| Aspect | Primus Implementation |
|--------|----------------------|
| Import path | `primus.backends.megatron.core.models.diffusion.flux` |
| Base class | `DiffusionModule` (extends MegatronModule) |
| Config parent | `BaseDiffusionConfig` (extends TransformerConfig) |
| Layer organization | Unified `TransformerBlock` with heterogeneous specs |
| Checkpoint format | `transformer.layers.{0-56}` unified namespace |
| Process groups | Via `pg_collection` parameter |

### Key design choices

**TransformerBlock Architecture**:
- Primus uses Megatron-Core's `TransformerBlock` with heterogeneous layer specifications
- Unified checkpoint format for simpler distributed training
- Note: pipeline parallelism is not supported for diffusion models (`pipeline_model_parallel_size` must be 1)

**Example Usage**:

```python
# Primus native approach
from primus.backends.megatron.core.models.diffusion.flux import Flux, FluxConfig

# Primus
from primus.backends.megatron.core.models.diffusion.flux.model import Flux
from primus.backends.megatron.core.models.diffusion.flux.config import FluxConfig
config = FluxConfig()
model = Flux(config)
```

For more advanced examples, see `examples/run_pretrain.sh`.

---

## References

### Papers
- **Flux**: "Flux: A Scalable Diffusion Model for High-Resolution Image Synthesis"
- **MMDiT**: "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis"
- **Flow Matching**: "Flow Matching for Generative Modeling"
- **RoPE**: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
- **DiT**: "Scalable Diffusion Models with Transformers" (Peebles & Xie, 2023)

### Source code
- **Primus Implementation**: `primus/backends/megatron/core/models/diffusion/flux/`
- **Megatron-Core**: `megatron/core/transformer/`
- **Official Flux**: Black Forest Labs (HuggingFace)

---

## Version information

- **Primus Version**: Current
- **Megatron-Core Version**: Latest
- **Transformer Engine**: Optional (recommended for performance)

---

## Support

For issues or questions:
1. Check this API reference
2. Read architecture guide: `docs/04-technical-guides/diffusion-models/flux_architecture.md`
3. See examples in docstrings
4. Check test files for usage patterns
