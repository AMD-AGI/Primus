# Energon Integration Guide

This guide explains how Megatron-Energon is integrated with Primus diffusion models, including the Cooker/TaskEncoder pattern, dataloader configuration, and dataset format.

---

## Table of Contents

1. [Overview](#overview)
2. [Energon Architecture](#energon-architecture)
3. [Dataset Format](#dataset-format)
4. [TaskEncoder and Cooker Pattern](#taskencoder-and-cooker-pattern)
5. [Dataloader Setup](#dataloader-setup)
6. [Dataset Configuration](#dataset-configuration)
7. [Implementation Examples](#implementation-examples)
8. [Best Practices](#best-practices)

---

## Overview

### What is Megatron-Energon?

Megatron-Energon is NVIDIA's data loading framework for large-scale multimodal training. It provides:
- **WebDataset integration**: Efficient streaming from .tar archives
- **Task encoding**: Flexible data transformation pipeline via Cookers
- **Multi-worker support**: Parallel data loading
- **Deterministic iteration**: Reproducible training
- **Checkpoint resumption**: Resume from any step

### Why Energon for Diffusion?

- **Proven at scale**: Used by NVIDIA for LLM and multimodal training
- **Flexible**: Supports various data formats and transformations
- **Efficient**: Optimized for multi-GPU training
- **Compatible**: Works with Megatron parallelism (TP, PP, DP)

### Primus Integration Strategy

```
Shared Infrastructure (data/)
  ├─ MegatronDataloaderWrapper
  ├─ Dataset configuration parsers
  └─ Common utilities
      ↓
Model-Specific TaskEncoders (data/diffusion/task_encoders/)
  ├─ EncodedDiffusionTaskEncoder (preencoded data)
  ├─ RawDiffusionTaskEncoder (raw images + text)
  └─ Custom task encoders
```

**Key Principle**: Share infrastructure, separate domain logic.

---

## Energon Architecture

### Component Stack

```
Training Loop
    ↓
MegatronDataloaderWrapper (Primus wrapper, cyclic iteration)
    ↓
Megatron-Energon Core
    ├─ WebDataset Reader (reads .tar shards)
    ├─ Cooker (transforms raw dict → typed Sample)
    ├─ TaskEncoder.batch() (stacks samples → batch)
    └─ Worker Pool
    ↓
.tar Shards (CrudeWebdataset format)
```

### Data Flow

```
1. Load from .tar shard
   → raw_dict = {'__key__': '0000000000',
                  'latents.pth': bytes,
                  'prompt_embeds.pth': bytes,
                  'pooled_prompt_embeds.pth': bytes,
                  'caption.txt': bytes}

2. Cooker function (e.g., cook_preencoded_diffusion)
   → Deserializes bytes to tensors
   → Returns typed DiffusionSample dataclass
   → DiffusionSample(latents=Tensor[64,128,128],
                      prompt_embeds=Tensor[512,4096],
                      pooled_prompt_embeds=Tensor[768],
                      caption="a photo of...")

3. TaskEncoder.batch()
   → Stacks list of DiffusionSample into batch dict
   → batch = {'latents': [B, 64, H, W],
              'prompt_embeds': [B, seq_len, 4096],
              'pooled_prompt_embeds': [B, 768]}

4. Return to training loop via MegatronDataloaderWrapper
   → Forward pass, loss, backprop
```

---

## Dataset Format

### CrudeWebdataset

Primus uses Energon's `CrudeWebdataset` format for preprocessed datasets. This is the simplest Energon format -- it stores raw key-value pairs in tar shards without requiring a strict schema.

The dataset type is configured in `.nv-meta/dataset.yaml`:

```yaml
__module__: megatron.energon
__class__: CrudeWebdataset
subflavors:
  encoding: preencoded    # or 'raw'
```

The `subflavors.encoding` field tells the TaskEncoder which Cooker to use:
- `preencoded`: Uses `cook_preencoded_diffusion` -- loads pre-encoded tensors
- `raw`: Uses `cook_raw_images` -- loads raw images and text

### Pre-encoded Shard Contents

Each tar shard contains samples with these keys:

| Key | Type | Shape | Description |
|-----|------|-------|-------------|
| `latents.pth` | Tensor | `[64, H/8, W/8]` | VAE-encoded image latents |
| `prompt_embeds.pth` | Tensor | `[seq_len, 4096]` | T5-XXL text embeddings |
| `pooled_prompt_embeds.pth` | Tensor | `[768]` | CLIP-L pooled embeddings |
| `caption.txt` | str | N/A | Original caption text |

### Raw Shard Contents

| Key | Type | Description |
|-----|------|-------------|
| `jpg` / `png` / `webp` | bytes | Preprocessed image |
| `txt` | str | Caption text |

### Known Energon CLI Limitations

The standard Energon CLI tools have issues with `CrudeWebdataset`:
- `energon info` raises `KeyError: 'sample_type'`
- `energon preview` raises `TypeError` (expects dataclass, `CrudeSample` is a dict)
- `energon lint` raises `AssertionError` (expects registered cookers)

Primus includes custom validation (`primus.backends.megatron.data.diffusion.preprocessing.validate`) as a replacement. See the [Data Preprocessing Guide](data_preprocessing.md#validation) for details.

---

## TaskEncoder and Cooker Pattern

Primus uses Energon's Cooker pattern rather than the older `encode_sample()` approach. Cookers are `@stateless` functions that transform raw sample dicts into typed dataclass instances, dispatched based on `subflavors`.

### DiffusionSample Dataclass

Location: `primus/backends/megatron/data/diffusion/task_encoders/image.py`

```python
# (imports like torch omitted for brevity)
from dataclasses import dataclass
from megatron.energon import Sample

@dataclass
class DiffusionSample(Sample):
    """
    Diffusion training sample with framework-standard field names.

    Inherits from megatron.energon.Sample to ensure __key__, __restore_key__,
    and __subflavors__ are properly tracked for deterministic training resumption.
    """
    latents: torch.Tensor              # [C, H, W] VAE latents
    prompt_embeds: torch.Tensor        # [seq_len, hidden_dim] T5 embeddings
    pooled_prompt_embeds: torch.Tensor # [hidden_dim] CLIP pooled
    caption: str = ""
```

### Cooker Functions

Cookers are `@stateless` functions registered with a `Cooker` wrapper that specifies which `subflavors` they handle:

```python
# (imports like torch, io omitted for brevity)
from megatron.energon import Cooker, basic_sample_keys, stateless

@stateless
def cook_preencoded_diffusion(sample: dict) -> DiffusionSample:
    """Load precalculated VAE latents and text embeddings from disk."""

    def load_tensor(data):
        if isinstance(data, bytes):
            return torch.load(io.BytesIO(data), map_location='cpu')
        return data

    latents = load_tensor(sample.get('latents.pth'))
    prompt_embeds = load_tensor(sample.get('prompt_embeds.pth'))
    pooled_prompt_embeds = load_tensor(sample.get('pooled_prompt_embeds.pth'))

    caption = sample.get('caption.txt', b'')
    if isinstance(caption, bytes):
        caption = caption.decode('utf-8')

    return DiffusionSample(
        **basic_sample_keys(sample),
        latents=latents,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        caption=caption,
    )


@stateless
def cook_raw_images(sample: dict) -> Dict[str, Any]:
    """Load raw images and text -- NO encoding, just data loading."""
    return {
        **basic_sample_keys(sample),
        'images': sample.get('images'),
        'txt': sample.get('txt'),
    }
```

The cooker code above is simplified for clarity. See `primus/backends/megatron/data/diffusion/task_encoders/image.py` for the full implementation, which includes additional input type handling and validation.

### EncodedDiffusionTaskEncoder

The TaskEncoder registers Cookers and provides the `batch()` method:

```python
from megatron.energon import DefaultTaskEncoder, SampleDecoder, Cooker, WorkerConfig

class EncodedDiffusionTaskEncoder(DefaultTaskEncoder[DiffusionSample, DiffusionSample, dict, dict]):
    """TaskEncoder for PRE-ENCODED diffusion data."""

    decoder = SampleDecoder(image_decode="pil")

    cookers = [
        Cooker(cook_preencoded_diffusion, has_subflavors={"encoding": "preencoded"}),
    ]

    def __init__(self, worker_config: Optional[WorkerConfig] = None):
        super().__init__()
        self.worker_config = worker_config

    def batch(self, samples: List[DiffusionSample]) -> Dict[str, torch.Tensor]:
        return {
            'latents': torch.stack([s.latents for s in samples]),
            'prompt_embeds': torch.stack([s.prompt_embeds for s in samples]),
            'pooled_prompt_embeds': torch.stack([s.pooled_prompt_embeds for s in samples]),
        }
```

### RawDiffusionTaskEncoder

For raw (on-the-fly encoding) datasets:

```python
class RawDiffusionTaskEncoder(DefaultTaskEncoder):
    """TaskEncoder for RAW diffusion data (images and text)."""

    decoder = SampleDecoder(image_decode="pil")

    cookers = [
        Cooker(cook_raw_images, has_subflavors={"encoding": "raw"}),
    ]

    def __init__(self, worker_config: Optional[WorkerConfig] = None):
        super().__init__()
        self.worker_config = worker_config

    def batch(self, samples: List[Dict]) -> Dict[str, Any]:
        return {
            'images': [s['images'] for s in samples],
            'txt': [s['txt'] for s in samples],
        }
```

### How Cooker Dispatch Works

The Cooker framework matches samples to cooker functions based on `subflavors`:

1. `dataset.yaml` specifies `subflavors: { encoding: preencoded }`
2. Energon reads a sample from the tar shard
3. The `has_subflavors` on each `Cooker` is checked against the sample's subflavors
4. The matching cooker function is called to transform the raw dict into a typed sample
5. `TaskEncoder.batch()` stacks multiple samples into a training batch

This decouples data format (what's in the tar) from data loading logic (how to interpret it).

---

## Dataloader Setup

### MegatronDataloaderWrapper

Location: `primus/backends/megatron/data/dataloader.py`

The `MegatronDataloaderWrapper` is a generic wrapper that makes any iterable compatible with Megatron's training loop. It provides:
- Cyclic iteration (never raises StopIteration)
- Optional checkpoint support via duck typing (`save_state_rank()` / `restore_state_rank()`)
- Works with PyTorch DataLoader, Energon loaders, and synthetic data

```python
from primus.backends.megatron.data.dataloader import MegatronDataloaderWrapper

wrapper = MegatronDataloaderWrapper(pytorch_or_energon_loader)
```

Note: Originally named `EnergonDataloader`, renamed to `MegatronDataloaderWrapper` to reflect its generic nature (it has no Energon dependencies). The old name is available as a deprecated alias.

### Usage Example

```python
from primus.backends.megatron.data.dataloader import MegatronDataloaderWrapper

# Create Energon loader (with TaskEncoder configured)
energon_loader = get_loader(...)
dataloader = MegatronDataloaderWrapper(energon_loader)

# Use in training loop
for batch in dataloader:
    latents = batch['latents']                          # [B, 64, H, W]
    prompt_embeds = batch['prompt_embeds']              # [B, seq_len, 4096]
    pooled_prompt_embeds = batch['pooled_prompt_embeds'] # [B, 768]

    output = model(latents, timesteps, prompt_embeds, pooled_prompt_embeds)
    loss = criterion(output, target)
    loss.backward()
```

---

## Dataset Configuration

### Per-Dataset dataset.yaml

Each dataset directory has a `.nv-meta/dataset.yaml` that specifies the Energon dataset type:

```yaml
__module__: megatron.energon
__class__: CrudeWebdataset
subflavors:
  encoding: preencoded
```

This file is auto-generated by Primus during finalization. See [Data Preprocessing Guide](data_preprocessing.md#finalization).

### Metadataset (Multi-Dataset Mixing)

For combining multiple datasets with different weights:

```yaml
__module__: megatron.energon
__class__: Metadataset

splits:
  train:
    datasets:
      - weight: 0.7
        path: /data/laion_precalculated/
      - weight: 0.3
        path: /data/coco_precalculated/
```

Energon samples proportionally to weights (70% from LAION, 30% from COCO).

---

## Implementation Examples

### Example 1: Pre-encoded Training

```python
from primus.backends.megatron.data.dataloader import MegatronDataloaderWrapper
from primus.backends.megatron.data.diffusion.task_encoders import EncodedDiffusionTaskEncoder

# TaskEncoder is configured by the dataset provider
# The cooker automatically handles subflavors dispatch
energon_loader = get_loader(...)
dataloader = MegatronDataloaderWrapper(energon_loader)

for batch in dataloader:
    output = model(
        batch['latents'],
        batch['prompt_embeds'],
        batch['pooled_prompt_embeds'],
    )
```

### Example 2: Raw Data Training (On-the-Fly Encoding)

```python
from primus.backends.megatron.data.diffusion.task_encoders import RawDiffusionTaskEncoder

# Raw TaskEncoder passes through images and text without encoding
# Encoding happens in the model's forward_step
energon_loader = get_loader(...)
dataloader = MegatronDataloaderWrapper(energon_loader)

for batch in dataloader:
    images = batch['images']   # List of PIL Images
    captions = batch['txt']    # List of caption strings
    # Model handles VAE/T5/CLIP encoding in forward_step
```

### Example 3: Multi-GPU Training

```python
import torch.distributed as dist

dist.init_process_group(backend='nccl')

energon_loader = get_loader(...)
dataloader = MegatronDataloaderWrapper(energon_loader)

for batch in dataloader:
    output = model(batch['latents'], ...)
```

---

## Best Practices

### 1. Pre-encoded Mode
- **Always use for production training**: 5-10x faster
- **Validate first**: Test with small dataset using `quickstart_pokemon.yaml`
- **Version datasets**: Track which preprocessing config produced each dataset

### 2. Cooker Design
- **Use `@stateless`**: Cookers must be stateless and side-effect-free
- **Use `basic_sample_keys()`**: Always forward `__key__`, `__restore_key__`, `__subflavors__`
- **Keep it simple**: Cookers should only deserialize and restructure data, not transform it
- **Use subflavors for dispatch**: Let the framework choose the right cooker

### 3. TaskEncoder Design
- **Single responsibility**: One task encoder per data format family
- **Minimal `batch()`**: Only stack tensors, avoid computation in the batch method
- **Separate concerns**: Data loading in Cooker, encoding in model forward_step

### 4. Dataloader Configuration
- **Num workers**: Match CPU cores (typically 4-8)
- **Batch size**: Max out GPU memory
- **Shuffle**: Always true for training
- **Drop last**: True to avoid irregular batches

### 5. Debugging
- **Small dataset**: Test with 100 samples first (`--max-samples 100`)
- **Single worker**: Set `num_workers=1` for debugging
- **Validate**: Run `python -m primus.backends.megatron.data.diffusion.preprocessing.validate /path/to/dataset`

---

## Customization Patterns

### Custom Cooker Function

Add a new cooker for a different data format:

```python
from megatron.energon import Cooker, basic_sample_keys, stateless

@stateless
def cook_my_custom_format(sample: dict) -> DiffusionSample:
    """Custom cooker for a different data layout."""
    latents = torch.load(io.BytesIO(sample['vae_output.pth']), map_location='cpu')
    prompt = torch.load(io.BytesIO(sample['text_embed.pth']), map_location='cpu')
    pooled = torch.load(io.BytesIO(sample['clip_embed.pth']), map_location='cpu')

    return DiffusionSample(
        **basic_sample_keys(sample),
        latents=latents,
        prompt_embeds=prompt,
        pooled_prompt_embeds=pooled,
    )

class CustomDiffusionTaskEncoder(DefaultTaskEncoder[DiffusionSample, DiffusionSample, dict, dict]):
    decoder = SampleDecoder(image_decode="pil")
    cookers = [
        Cooker(cook_my_custom_format, has_subflavors={"encoding": "custom_v2"}),
    ]

    def batch(self, samples):
        return {
            'latents': torch.stack([s.latents for s in samples]),
            'prompt_embeds': torch.stack([s.prompt_embeds for s in samples]),
            'pooled_prompt_embeds': torch.stack([s.pooled_prompt_embeds for s in samples]),
        }
```

### Multiple Cookers in One TaskEncoder

A single TaskEncoder can register multiple cookers for different subflavors:

```python
class MultiFormatTaskEncoder(DefaultTaskEncoder[DiffusionSample, DiffusionSample, dict, dict]):
    cookers = [
        Cooker(cook_preencoded_diffusion, has_subflavors={"encoding": "preencoded"}),
        Cooker(cook_my_custom_format, has_subflavors={"encoding": "custom_v2"}),
    ]
```

---

## References

- **Megatron-Energon**: [nvidia/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) multimodal examples
- **WebDataset**: [webdataset/webdataset](https://github.com/webdataset/webdataset)
- **NeMo Implementation**: `nemo/collections/diffusion/data/`
- **Primus TaskEncoders**: `primus/backends/megatron/data/diffusion/task_encoders/image.py`
- **Primus Dataloader**: `primus/backends/megatron/data/dataloader.py`

---

**Last Updated**: March 2026
