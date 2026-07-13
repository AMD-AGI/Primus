# Data Preprocessing Guide

This guide explains how to prepare datasets for Flux and other diffusion models in Primus, including pre-encoding of VAE latents and text embeddings into Energon WebDataset format.

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Two Pipelines](#two-pipelines)
4. [Running Preprocessing](#running-preprocessing)
5. [Configuration](#configuration)
6. [Authentication](#authentication)
7. [Finalization](#finalization)
8. [Output Format](#output-format)
9. [Validation](#validation)
10. [Troubleshooting](#troubleshooting)

---

## Overview

Diffusion models require three types of encodings:
1. **VAE latents**: Images encoded to latent space
2. **Text embeddings**: Captions encoded with T5-XXL (sequence)
3. **Pooled embeddings**: Captions encoded with CLIP-L (pooled)

### Why Pre-encode?

**Benefits**:
- 5-10x faster training (no online encoding)
- Lower GPU memory usage (encoders not loaded during training)
- Deterministic inputs (same preprocessing for all runs)
- Eliminates encoder differences as a variable

**When to Use**:
- Training (highly recommended)
- Fine-tuning on fixed datasets
- Benchmarking and reproducibility

**When NOT to Use**:
- Interactive data augmentation needed
- Dataset too large to store pre-encoded
- Rapid prototyping with changing data

---

## Quick Start

Preprocess the Pokemon dataset with a single command:

```bash
primus-cli direct -- data diffusion-encoded \
  --config primus/configs/data/megatron/diffusion/preprocessing/quickstart_pokemon.yaml \
  --hf-token-file /path/to/.hf_token
```

This will:
1. Download the `diffusers/pokemon-gpt4-captions` dataset from HuggingFace
2. Encode all images with VAE and captions with T5/CLIP
3. Write Energon WebDataset tar shards to `/workspace/Primus/data/quickstart_pokemon`
4. Automatically finalize the dataset (create `dataset.yaml`, run `energon prepare`, validate)

The default encoder model (`black-forest-labs/FLUX.1-dev`) is gated and requires a HuggingFace token. Get one at https://huggingface.co/settings/tokens and save it to a file.

---

## Two Pipelines

Primus provides two preprocessing pipelines via the `primus data` CLI:

### `diffusion-encoded` (Recommended for Training)

Pre-encodes images with VAE and text with T5/CLIP. Produces larger datasets but enables faster training since encoders are not needed at training time.

```bash
primus-cli direct -- data diffusion-encoded \
  --config primus/configs/data/megatron/diffusion/preprocessing/example_huggingface.yaml \
  --hf-token-file /path/to/.hf_token
```

**Output**: WebDataset shards containing `latents.pth`, `prompt_embeds.pth`, `pooled_prompt_embeds.pth`, `caption.txt`

**Use when**: Training on production datasets, maximum training speed is needed, storage is available.

### `diffusion-raw`

Stores raw images and captions without encoding. Smaller datasets but encoding happens on-the-fly during training (requires GPU + encoders loaded in memory).

```bash
primus-cli direct -- data diffusion-raw \
  --source-type huggingface \
  --hf-dataset diffusers/pokemon-gpt4-captions \
  --output-dir /workspace/Primus/data/raw_pokemon \
  --hf-token-file /path/to/.hf_token
```

**Output**: WebDataset shards containing `jpg` (or `png`/`webp`) and `txt` files.

**Use when**: Storage is limited, experimenting with different encoders, rapid prototyping.

**Note**: `diffusion-raw` does not support `--config` files. All options must be passed as CLI arguments.

---

## Running Preprocessing

### Using a Config File (Recommended)

The `--config` flag is supported by `diffusion-encoded` only. The simplest approach uses a YAML config file:

```bash
primus-cli direct -- data diffusion-encoded \
  --config primus/configs/data/megatron/diffusion/preprocessing/example_huggingface.yaml \
  --hf-token-file /path/to/.hf_token
```

Available example configs in `primus/configs/data/megatron/diffusion/preprocessing/`:

| Config | Source | Description |
|--------|--------|-------------|
| `quickstart_pokemon.yaml` | HuggingFace | Minimal config, 256px, fast |
| `example_huggingface.yaml` | HuggingFace | Full example with all options |
| `example_directory.yaml` | Local directory | Images + captions from disk |
| `example_webdataset.yaml` | WebDataset | Existing tar archives |
| `example_base.yaml` | N/A | Comprehensive reference with all fields |
| `text_to_image_2m_10k.yaml` | HuggingFace | 10K subset of text-to-image-2M (1024px) |

### Using CLI Arguments Directly

All config values can be provided as CLI arguments:

```bash
primus-cli direct -- data diffusion-encoded \
  --source-type huggingface \
  --hf-dataset diffusers/pokemon-gpt4-captions \
  --output-dir /workspace/Primus/data/encoded_pokemon \
  --model-path black-forest-labs/FLUX.1-dev \
  --batch-size 8 \
  --precision bf16 \
  --hf-token-file /path/to/.hf_token
```

### CLI Overrides Config Values

When using both `--config` and CLI arguments, CLI arguments take priority:

```bash
primus-cli direct -- data diffusion-encoded \
  --config primus/configs/data/megatron/diffusion/preprocessing/example_huggingface.yaml \
  --hf-token-file /path/to/.hf_token \
  --output-dir /my/custom/path \
  --batch-size 16 \
  --max-samples 1000
```

Priority order (highest to lowest):
1. Explicitly provided CLI arguments
2. YAML config file values
3. CLI default values

### Multi-GPU Processing

Use `--nproc-per-node` for data-parallel preprocessing across multiple GPUs:

```bash
primus-cli direct -- --nproc-per-node=8 data diffusion-encoded \
  --config primus/configs/data/megatron/diffusion/preprocessing/example_huggingface.yaml \
  --hf-token-file /path/to/.hf_token
```

Each GPU processes a subset of the data. Shards are named to avoid conflicts across ranks.

---

## Configuration

### YAML Config Structure

Preprocessing configs have four sections:

```yaml
source:
  type: huggingface          # huggingface | directory | webdataset
  hf_dataset: diffusers/pokemon-gpt4-captions
  hf_split: train

output:
  output_dir: /workspace/Primus/data/encoded_pokemon
  shard_size: 1000           # samples per tar shard
  max_samples: null          # null = process all
  compress: false

model:
  model_path: black-forest-labs/FLUX.1-dev   # HF repo or local path
  precision: bf16            # bf16 | fp16 | fp32
  batch_size: 8
  # Optional per-encoder overrides:
  vae_path: null
  t5_path: null
  clip_path: null

image:
  image_size: 1024
  center_crop: false
```

### Source Types

**HuggingFace** (`type: huggingface`):
```yaml
source:
  type: huggingface
  hf_dataset: diffusers/pokemon-gpt4-captions
  hf_split: train
  hf_data_files: null        # optional: specific files within dataset
```

**Local Directory** (`type: directory`):
```yaml
source:
  type: directory
  input_dir: /data/my_images
```

Expected directory structure for `directory` source:
```
my_images/
├── images/
│   ├── 00001.jpg
│   ├── 00002.png
│   └── ...
└── captions/
    ├── 00001.txt
    ├── 00002.txt
    └── ...
```

**WebDataset** (`type: webdataset`):
```yaml
source:
  type: webdataset
  input_path: /data/existing_shards/*.tar
```

### Model Configuration

The `model_path` defaults to `black-forest-labs/FLUX.1-dev`, which downloads VAE, T5-XXL, and CLIP-L encoders from HuggingFace. This model is gated and requires authentication (see [Authentication](#authentication)).

Individual encoder paths can be overridden:

```yaml
model:
  model_path: black-forest-labs/FLUX.1-dev
  vae_path: /local/models/vae        # use local VAE instead
  t5_path: null                       # falls back to model_path
  clip_path: null                     # falls back to model_path
```

---

## Authentication

The default encoder model (`FLUX.1-dev`) is gated on HuggingFace and requires authentication. Primus supports three authentication methods, checked in priority order:

### 1. Token File (Recommended)

```bash
primus-cli direct -- data diffusion-encoded \
  --config your_config.yaml \
  --hf-token-file /path/to/.hf_token
```

The token file must have secure permissions (600 or 400). Create it with:

```bash
echo "hf_your_token_here" > /path/to/.hf_token
chmod 600 /path/to/.hf_token
```

### 2. Environment Variable

```bash
export HF_TOKEN=hf_your_token_here
primus-cli direct -- data diffusion-encoded --config your_config.yaml
```

### 3. HuggingFace CLI Login

```bash
huggingface-cli login
primus-cli direct -- data diffusion-encoded --config your_config.yaml
```

If authentication fails, Primus provides a clear error message indicating which encoder failed and how to fix it.

---

## Finalization

Finalization is **automatic by default**. After preprocessing completes, Primus automatically:

1. **Creates `.nv-meta/dataset.yaml`** with `CrudeWebdataset` sample type and encoding subflavor
2. **Runs `energon prepare`** to index the tar shards and create split assignments
3. **Validates the dataset** using Primus's custom validation (metadata checks, sample count verification, energon API spot-check)

### Skipping Finalization

To skip automatic finalization (e.g., for manual post-processing):

```bash
primus-cli direct -- data diffusion-encoded \
  --config your_config.yaml \
  --hf-token-file /path/to/.hf_token \
  --no-finalize
```

### Custom Train/Val/Test Splits

By default, 100% of data goes to the training split. To create validation and test splits:

```bash
primus-cli direct -- data diffusion-encoded \
  --config your_config.yaml \
  --hf-token-file /path/to/.hf_token \
  --train-split 0.8
```

This creates an 80% train / 10% val / 10% test split.

---

## Output Format

### Directory Structure

After preprocessing and finalization, the output directory contains:

```
encoded_pokemon/
├── 000000.tar              # WebDataset shard
├── 000001.tar
├── 000000.tar.idx          # Energon index files
├── 000001.tar.idx
└── .nv-meta/               # Energon metadata
    ├── dataset.yaml         # Dataset type configuration
    ├── split.yaml           # Train/val/test split assignments
    └── .info.json           # Shard counts and sample counts
```

### Pre-encoded Shard Contents (`diffusion-encoded`)

Each tar shard contains samples with these keys:

```
000000.tar:
├── 0000000000.latents.pth              # VAE latents tensor
├── 0000000000.prompt_embeds.pth        # T5-XXL embeddings tensor
├── 0000000000.pooled_prompt_embeds.pth # CLIP-L pooled embeddings tensor
├── 0000000000.caption.txt              # Original caption text
├── 0000000001.latents.pth
├── 0000000001.prompt_embeds.pth
├── ...
```

Tensor shapes:
- `latents.pth`: `[64, H/8, W/8]` (e.g., `[64, 128, 128]` for 1024x1024 images)
- `prompt_embeds.pth`: `[seq_len, 4096]` (T5-XXL hidden dim)
- `pooled_prompt_embeds.pth`: `[768]` (CLIP-L pooled dim)

### Raw Shard Contents (`diffusion-raw`)

```
000000.tar:
├── 0000000000.jpg       # Preprocessed image
├── 0000000000.txt       # Caption text
├── 0000000001.jpg
├── 0000000001.txt
├── ...
```

### Dataset YAML

The auto-generated `.nv-meta/dataset.yaml` uses `CrudeWebdataset` format:

```yaml
__module__: megatron.energon
__class__: CrudeWebdataset
subflavors:
  encoding: preencoded    # or 'raw' for diffusion-raw
```

---

## Validation

### Automatic Validation

Validation runs automatically as part of finalization. It performs four checks:

1. **Metadata check**: Verifies `.nv-meta/` files exist and are valid (`.info.json`, `split.yaml`, `dataset.yaml`)
2. **Sample count check**: Spot-checks that tar shard entry counts match `.info.json`
3. **Sample load check**: Loads one sample through Energon's Python API (same code path as training)
4. **Summary report**: Prints dataset statistics (encoding, total samples, splits, data shapes, size)

### Standalone Validation

To validate a dataset independently:

```bash
python -m primus.backends.megatron.data.diffusion.preprocessing.validate /path/to/dataset

# For raw datasets:
python -m primus.backends.megatron.data.diffusion.preprocessing.validate /path/to/dataset --encoding raw
```

### Programmatic Validation

```python
from primus.backends.megatron.data.diffusion.preprocessing.validate import validate_energon_dataset

ok = validate_energon_dataset('/path/to/dataset', encoding='preencoded')
```

### Known Energon CLI Limitations

The standard Energon CLI tools (`energon info`, `energon preview`, `energon lint`) do **not** work correctly with `CrudeWebdataset` format. Primus uses custom validation instead:

- `energon info` raises `KeyError: 'sample_type'`
- `energon preview` raises `TypeError` (expects dataclass, `CrudeSample` is a dict)
- `energon lint` raises `AssertionError` (expects registered cookers)

Use Primus's built-in validation or the standalone script above.

---

## Troubleshooting

### HuggingFace Authentication Failure

**Symptoms**: Error mentioning "token", "gated", "401", or "403" when downloading encoders.

**Solutions**:
1. Provide a token: `--hf-token-file /path/to/.hf_token`
2. Set environment variable: `export HF_TOKEN=hf_xxx`
3. Run `huggingface-cli login`
4. Accept the model's license on https://huggingface.co/black-forest-labs/FLUX.1-dev

### Out of Memory During Preprocessing

**Symptoms**: CUDA out of memory error during encoding.

**Solutions**:
1. Reduce batch size: `--batch-size 4` or `--batch-size 1`
2. Use smaller image size: `--image-size 512`
3. Use fp16 precision: `--precision fp16`
4. Use multi-GPU to distribute work: `--nproc-per-node=8`

### Missing Dependencies

**Symptoms**: `ModuleNotFoundError` for `webdataset`, `megatron-energon`, `tqdm`, etc.

**Solution**:
```bash
pip install -r requirements.txt
```

Or set `PRIMUS_AUTO_INSTALL=1` in the container to auto-install missing packages.

### Finalization Fails

**Symptoms**: Error during `energon prepare` or validation after preprocessing.

**Solutions**:
1. Check that tar shards exist in the output directory
2. Ensure `megatron-energon` is installed (`pip install megatron-energon`)
3. Re-run with `--no-finalize`, then manually inspect the output before finalizing

### Slow Preprocessing

**Symptoms**: Low throughput (< 10 samples/sec).

**Solutions**:
1. Increase batch size (limited by VRAM): `--batch-size 16`
2. Use multiple GPUs: `--nproc-per-node=8`
3. Use bf16 precision (faster on supported hardware): `--precision bf16`
4. For raw pipeline, reduce image quality: `--image-quality 85`

---

## Best Practices

1. **Always pre-encode for production training**: 5-10x speedup is worth the storage
2. **Test on small dataset first**: Use `--max-samples 100` to verify the pipeline works
3. **Use bf16 precision**: Good balance of speed, storage, and quality
4. **Start with quickstart_pokemon.yaml**: Verify your setup before processing large datasets
5. **Keep raw data**: Pre-encoding is a one-way transformation
6. **Version your configs**: Track which preprocessing config produced each dataset

---

## Flux-Specific Data Preparation (torchrun)

This section covers preparing datasets for Flux training using `torchrun` directly
inside a Docker container, as an alternative to the `primus-cli` workflow above.

### Prerequisites

Start the container and optionally install requirements:

```bash
bash tools/docker/start_container.sh
docker exec dev_primus bash -c 'pip install -r /workspace/Primus/requirements.txt'
```

This mounts the repository to `/workspace/Primus` inside the container.
Override the image with `DOCKER_IMAGE`:

```bash
DOCKER_IMAGE=docker.io/rocm/primus:v26.1 bash tools/docker/start_container.sh
```

### Input Directory Structure

When using `--source-type directory`, organize your data as follows:

```
dataset/
├── images/
│   ├── 0000000.png
│   ├── 0000001.png
│   └── ...
└── captions/
    ├── 0000000.txt
    ├── 0000001.txt
    └── ...
```

The file stem links each image to its caption (e.g., `images/0000000.png` pairs
with `captions/0000000.txt`). Images can be `.jpg`, `.jpeg`, `.png`, or `.webp`.
Captions must be UTF-8 `.txt` files. Samples without a matching caption are skipped.

Other supported source types:
- **huggingface** -- Load directly from HuggingFace Hub. Requires `--hf-dataset`.
- **webdataset** -- Read from existing WebDataset tar archives. Requires `--input-path`.

### Image Sizing

**Fixed size (default):** Every image is resized to `--image-size` pixels square
(default 1024). Use `--center-crop` to control center-cropping.

**Variable size (`--variable-size`):** Preserves aspect ratio by scaling the
longest side to `--max-size` (default 1024) and rounding dimensions to multiples
of 16. Only use this when all images share the same dimensions -- mixed tensor
sizes cause load imbalance across GPUs.

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--source-type` | — | `directory`, `huggingface`, or `webdataset` |
| `--input-dir` | — | Dataset root (for `directory` source) |
| `--image-size` | 1024 | Square target size |
| `--variable-size` | off | Preserve aspect ratio |
| `--max-size` | 1024 | Maximum dimension in variable-size mode |
| `--model-path` | `black-forest-labs/FLUX.1-dev` | Base model for encoder weights |
| `--t5-max-length` | 512 | T5 token limit (use 256 for FLUX.1-schnell) |
| `--batch-size` | 8 | Encoding batch size per GPU |
| `--output-dir` | — | Destination for encoded WebDataset shards |
| `--shard-size` | 1000 | Samples per tar shard |

### Examples

**From host (via `docker exec`):**

```bash
docker exec dev_primus bash -c '\
  export HF_HOME=/workspace/Primus/checkpoints/flux; \
PYTHONPATH=/workspace/Primus:/workspace/Primus/third_party/Megatron-LM:$PYTHONPATH \
  torchrun --nproc_per_node=8 /workspace/Primus/primus/cli/main.py \
    data diffusion-encoded \
    --source-type directory \
    --input-dir /workspace/Primus/data/dataset \
    --output-dir /workspace/Primus/data/dataset_encoded_256 \
    --image-size 256  \
    --t5-max-length 256'
```

**Inside the container:**

```bash
export HF_HOME=/workspace/Primus/checkpoints/flux
PYTHONPATH=/workspace/Primus:/workspace/Primus/third_party/Megatron-LM:$PYTHONPATH \
  torchrun --nproc_per_node=8 /workspace/Primus/primus/cli/main.py \
    data diffusion-encoded \
    --source-type directory \
    --input-dir /workspace/Primus/data/dataset \
    --output-dir /workspace/Primus/data/dataset_encoded_256 \
    --image-size 256 \
    --t5-max-length 256
```

Both commands encode a local directory dataset at 256px on 8 GPUs, then create an Energon dataset by default.

### Limitations

**Uniform output size only.** When using `--variable-size`, images may produce
tensors of different shapes. The current data pipeline does not support mixed
tensor sizes because samples are distributed evenly across GPUs and unequal
shapes cause load imbalance and eventual timeout. Use fixed `--image-size` if
your images have various sizes.

---

## Next Steps

After preprocessing:
1. **Training**: Use the preprocessed dataset path in your training config
2. **Validation**: The dataset is ready for training immediately after finalization

See:
- [Energon Integration](energon_integration.md) for TaskEncoder and dataloader details
- [Config Directory Guide](../../../primus/configs/data/megatron/diffusion/README.md) for config file reference
- Example configs in `primus/configs/data/megatron/diffusion/preprocessing/`

---

**Last Updated**: June 2026
