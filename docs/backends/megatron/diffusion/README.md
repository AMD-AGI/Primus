# Diffusion Models in Primus - Developer & Architecture Guide

**Purpose:** Developer-focused documentation for understanding Primus diffusion architecture, design decisions, and implementation details.

**For training/usage instructions, see:** [examples/megatron/diffusion/README.md](../../../../examples/megatron/diffusion/README.md)

**For test documentation, see:** [tests/unit_tests/backends/megatron/diffusion/](../../../../tests/unit_tests/backends/megatron/diffusion/)

---

## Architecture Philosophy

Primus diffusion models are built as **Megatron-Core native implementations**, designed for:
- Production-scale distributed training
- Seamless integration with Megatron parallelism strategies (TP, PP, DP, EP)
- Advanced checkpoint management with heterogeneous layers
- Clean separation of concerns (no framework dependencies like PyTorch Lightning)

### Key Design Decisions

**1. Megatron-Core Integration**
- Models in `core/models/diffusion/` follow Megatron-Core patterns
- Extends `TransformerConfig` for configurations (inherits all Megatron features)
- Uses `TransformerBlock` with heterogeneous layer support
- Compatible with Megatron's distributed checkpointing

**2. Unified TransformerBlock Architecture**
- Unlike HuggingFace's ModuleLists, uses Megatron's unified TransformerBlock
- Simplifies checkpoint management
- More efficient gradient synchronization
- Note: pipeline parallelism is not supported for diffusion models (`pipeline_model_parallel_size` must be 1)

**3. No Framework Dependencies**
- Direct PyTorch implementation (no PyTorch Lightning)
- Uses Megatron's distributed primitives directly
- Simpler debugging and profiling
- Better control over distributed training

**4. Extensibility First**
- Base classes designed for multiple diffusion models (Flux, DiT, MovieGen)
- Clear shared vs model-specific separation
- Hierarchical encoder registry for easy extension

---

## Supported Models

### Flux ✅ Production Ready
Flow-based diffusion model with MMDiT (Multimodal Diffusion Transformer) architecture.

- **Architecture**: Dual-stream with joint and single transformer blocks
- **Sizes**: 535M (testing) and 12B (production)
- **Reference**: [Black Forest Labs FLUX.1](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- **Status**: Fully implemented and tested (390 tests)

### Future Models ⏳ Planned
- **DiT**: Diffusion Transformer for image generation
- **MovieGen**: Video diffusion models
- **Custom Models**: Extensible framework for new architectures

---

## Project Structure

```
primus/backends/megatron/
├── core/models/
│   ├── common/diffusion_module/    # DiffusionModule (base class with sharded state dict)
│   │   └── diffusion_module.py
│   └── diffusion/                  # Model implementations (Megatron-Core style)
│       ├── common/                 # Shared components (MMDiT layers, attention)
│       │   ├── config.py           # BaseDiffusionConfig (extends TransformerConfig)
│       │   └── layers.py           # Shared layers (if any)
│       └── flux/                   # Flux-specific code
│           ├── config.py           # FluxConfig with factory methods (535M, 12B)
│           ├── model.py            # Flux model (extends DiffusionModule)
│           └── layer_spec.py       # Flux layer specifications
│
├── training/diffusion/             # Training utilities
│   ├── noise_utils.py              # Noise application (flow matching, DDPM)
│   ├── loss_computation.py         # Loss functions (flow matching, epsilon, v-prediction)
│   ├── timestep_sampling.py        # Timestep sampling strategies
│   └── schedulers/
│       ├── base.py                 # BaseScheduler
│       └── flow_matching.py        # FlowMatchEulerDiscreteScheduler
│
└── data/
    ├── energon/                    # Shared Energon infrastructure
    └── diffusion/                  # Diffusion-specific data
        ├── encoders/               # Hierarchical encoder registry
        │   ├── image/vae/          # VAE variants (SD VAE, custom VAEs)
        │   ├── text/t5/            # T5 variants (XXL, etc.)
        │   └── text/clip/          # CLIP variants (L, H, etc.)
        ├── preprocessing/          # Data preprocessing utilities
        │   ├── download.py         # Reusable download utils (retry, MD5, manifests)
        │   ├── finalize.py         # Energon dataset finalization
        │   ├── validate.py         # Dataset structure validation
        │   └── pipelines/          # Dataset preparation pipelines
        │       ├── base.py         # DatasetPipeline abstract base class
        │       ├── raw.py          # Raw image pipeline
        │       ├── encoded.py      # Pre-encoded pipeline
        │       └── ingest.py       # StreamingIngestPipeline (MLPerf Arrow->WDS)
        └── task_encoders/          # Energon TaskEncoders for diffusion

primus/configs/models/megatron/diffusion/
├── flux_535m.yaml                  # Flux 535M config
├── flux_12b.yaml                   # Flux 12B config
└── encoders.yaml                   # Encoder configuration

tests/unit_tests/backends/megatron/diffusion/  # Comprehensive test suite (390 tests)
├── models/                         # Model-level tests
├── layers/                         # Layer-level tests
├── unit/                          # Unit tests for utilities
├── distributed/                    # Distributed training tests
├── functional/                     # End-to-end functional tests
└── checkpointing/                  # Checkpoint tests

docs/backends/megatron/diffusion/   # This directory
├── README.md                       # This file (developer guide)
├── architecture_overview.md        # Detailed architecture
├── data_preprocessing.md           # Data pipeline guide (includes Flux-specific section)
├── energon_integration.md          # Energon patterns
├── flux_architecture.md            # Flux deep dive
├── fp8_training.md                 # FP8 training guide (benchmarks, tuning, troubleshooting)
├── api_reference.md                # API documentation
├── adding_new_models.md            # Extension guide
└── STRUCTURE.md                    # Directory tree and organization
```

---

## Key Technical Features

### 1. DiffusionModule Base Class

All diffusion models inherit from `DiffusionModule`, which provides:
- Megatron-Core integration (process groups, parallelism)
- Sharded state dict support for distributed checkpointing
- Gradient checkpointing
- Mixed precision support
- Device placement utilities

**Location:** `primus/backends/megatron/core/models/common/diffusion_module/diffusion_module.py`

### 2. BaseDiffusionConfig

Configuration class extending `TransformerConfig`:
- Inherits all Megatron-Core configuration (TP, PP, sequence_parallel, etc.)
- Adds diffusion-specific parameters (channels, patch_size, etc.)
- Factory methods for common presets

**Location:** `primus/backends/megatron/core/models/diffusion/common/config.py`

### 3. Hierarchical Encoder Registry

Organized by modality → type → variant:
```
encoders/
├── image/vae/
│   ├── sd_vae.py              # Standard SD VAE
│   └── (future: custom VAEs)
├── text/t5/
│   ├── t5_xxl.py              # T5-XXL encoder
│   └── (future: T5 variants)
└── text/clip/
    ├── clip_l.py              # CLIP-L encoder
    └── (future: CLIP-H, etc.)
```

Benefits:
- Easy to add new encoder variants (5+ planned per modality)
- Config-driven selection via `encoders.yaml`
- Lazy loading (encoders loaded only when needed)
- Shared base classes for common functionality

### 4. Training Utilities Structure

**Noise Application** (`noise_utils.py`):
- `apply_flow_matching_noise()`: For flow matching models (Flux)
- `apply_ddpm_noise()`: For DDPM-based models
- Support for different noise schedules

**Loss Computation** (`loss_computation.py`):
- `compute_flow_matching_loss()`: For flow matching
- `compute_epsilon_loss()`: For epsilon prediction (DDPM)
- `compute_v_prediction_loss()`: For v-prediction
- Unified interface for different loss types

**Timestep Sampling** (`timestep_sampling.py`):
- `LogitNormalSampler`: Logit-normal distribution
- `UniformSampler`: Uniform distribution
- `ModeSampler`: Mode-focused sampling
- Base class for custom samplers

### 5. Shared Energon Infrastructure

Located in `data/energon/` for reusability across models:
- Shared data loading utilities
- Common preprocessing functions
- WebDataset integration
- Model-specific TaskEncoders in `data/diffusion/task_encoders/`

### 6. Precalculated Data Support

**Performance**: 5-10x faster training than on-the-fly encoding

**Supported encodings**:
- `preencoded` -- Primus-encoded PyTorch `.pth` format (VAE latents + text embeddings)
- `preencoded_numpy` -- MLPerf NumPy uint16 format (bfloat16 tensors as `.bytes` entries)

**Workflow**:
1. Precompute VAE latents and text embeddings offline
2. Store in WebDataset/Energon format
3. Load directly during training (no encoder overhead)

**Benefits**:
- Faster training iteration
- Consistent encoder versions across runs
- Lower GPU memory (no encoders loaded during training)
- Better reproducibility

### 7. MLPerf Streaming Ingest Pipeline

**Location:** `data/diffusion/preprocessing/pipelines/ingest.py`

The `StreamingIngestPipeline` downloads Apache Arrow IPC files from MLCommons R2 storage and converts them directly into Energon WebDataset tar shards in a single streaming pass. This avoids storing the full ~6 TB raw Arrow dataset on disk.

**Architecture**: Producer-consumer with concurrent download and sequential conversion:
- **Producer thread**: Acquires a semaphore permit, submits downloads to a `ThreadPoolExecutor`, passes completed futures to a drain thread
- **Drain thread**: Processes futures in submission order and feeds the prefetch queue
- **Consumer (main thread)**: Converts Arrow data to tar shards, deletes temporary files, releases semaphore permits

**Key properties**:
- Bounded disk usage: `threading.Semaphore(prefetch_depth)` limits Arrow files on disk
- Deterministic shard ordering preserved via in-order future draining
- Retry with exponential backoff for HTTP 429/503 and MD5 mismatches (`download.py`)
- Skip-and-log: individual failures are recorded in `failed_files.json`
- Resume: re-running skips shards that already exist on disk

**Related modules**:
- `download.py`: `download_with_backoff()`, `fetch_manifest()`, `parse_md5_manifest()`
- `pipelines/base.py`: `DatasetPipeline` ABC (shared by `raw.py`, `encoded.py`, `ingest.py`)
- `finalize.py`: Energon dataset finalization (`.nv-meta/dataset.yaml` + `energon prepare`)
- `validate.py`: Post-finalization structural validation

---

## Implementation Status

### Core Infrastructure ✅
- ✅ Directory structure with 25+ directories
- ✅ Base classes (DiffusionModule, BaseDiffusionConfig, BaseScheduler)
- ✅ DiffusionModule with Megatron-Core integration
- ✅ FluxConfig with factory methods (flux_535m, flux_12b)
- ✅ FlowMatchEulerDiscreteScheduler
- ✅ Configuration system (YAML files)
- ✅ Testing framework (390 tests)
- ✅ Comprehensive documentation

### Flux Model Implementation ✅
- ✅ Flux model architecture (dual-stream MMDiT)
- ✅ MMDiT layers and attention (joint + single blocks)
- ✅ Embeddings (3D RoPE, timestep, vector)
- ✅ Hierarchical encoder registry
- ✅ Data pipeline and TaskEncoders
- ✅ Training utilities (noise, loss, sampling)
- ✅ Checkpoint conversion (HF <-> Megatron)

---

## Documentation Map

### Core Guides

📖 **[Architecture Overview](architecture_overview.md)**
High-level design, directory structure, and architectural decisions.

📖 **[Directory Structure](STRUCTURE.md)**
Complete directory tree and file organization.

📖 **[Data Preprocessing Guide](data_preprocessing.md)**
How to prepare datasets, precalculate latents, and use Energon.

📖 **[Energon Integration](energon_integration.md)**
Megatron-Energon patterns and TaskEncoder implementation.

📖 **[Adding New Models](adding_new_models.md)**
Step-by-step guide for implementing new diffusion models.

### Advanced Documentation

📖 **[Flux Architecture Deep Dive](flux_architecture.md)**
Mathematical formulation, detailed component descriptions, and performance optimizations.

📖 **[API Reference](api_reference.md)**
Complete API documentation with function signatures and usage examples.

📖 **[FP8 Training Guide](fp8_training.md)**
FP8 precision training on AMD MI300X: configuration, benchmarks, tuning recipes, and troubleshooting.

### Related Documentation

📖 **[Training Guide](../../../../examples/megatron/diffusion/README.md)**
User-facing guide for training Flux models (quick start, configurations, troubleshooting).

📖 **[Test Directory](../../../../tests/unit_tests/backends/megatron/diffusion/)**
Test suite for diffusion models.

---

## Testing Architecture

**Test Organization** (following Megatron-LM patterns):
- One comprehensive file per model (`test_flux_model.py`)
- Unit tests for utilities (`unit/test_utils.py`, etc.)
- Distributed tests in separate directory (`distributed/`)
- Functional tests for workflows (`functional/`)

**Test Status**: ✅ 390 tests passing

See [tests/unit_tests/backends/megatron/diffusion/](../../../../tests/unit_tests/backends/megatron/diffusion/) for details.

---

## Hardware Requirements

### Flux 535M (Testing)
- **Training**: 1x MI300X 192GB (compatible with H100/A100)
- **Inference**: 1x MI300X 192GB
- **Batch Size**: 1-8 per GPU

### Flux 12B (Production)
- **Training**: 8x MI300X 192GB (recommended) or 4x MI300X 192GB with TP=2
- **Inference**: 1x MI300X 192GB
- **Batch Size**: 1-2 per GPU for training, 1-4 for inference

---

## Contributing

See the main guide: [Adding New Models](adding_new_models.md)

**To contribute:**
1. Follow the established directory structure
2. Extend base classes (DiffusionModule, BaseDiffusionConfig)
3. Add comprehensive tests in `tests/unit_tests/backends/megatron/diffusion/`
4. Update documentation (architecture guide + API reference)
5. Submit PR with clear description

---

## License

- **Primus Code**: AMD Copyright 2025, Apache License 2.0
- **Flux Encoders**:
  - FLUX.1 [dev]: Non-commercial license
  - FLUX.1 [schnell]: Apache 2.0 (commercial use allowed)
  - Individual components (T5, CLIP, VAE): Check respective licenses

---

## Resources

- **Megatron-Core**: [nvidia/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) - Core framework
- **Flux Model**: [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- **Flow Matching**: Rectified flow and flow matching papers
- **NeMo**: [nvidia/NeMo](https://github.com/NVIDIA/NeMo) - Alternative diffusion implementation

---

**Last Updated**: January 2026
