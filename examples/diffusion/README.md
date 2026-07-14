# Diffusion Examples

These examples exercise the independent PyTorch diffusion backend under
`primus/backends/diffusion`, including Wan video training and FLUX text-to-image
training. For backend details, data/checkpoint layout, and minimal configs, see
`primus/backends/diffusion/README.md`.

## Data

Wan examples use the `zirui3/tiny-video-samples` smoke-test dataset:

```bash
huggingface-cli download zirui3/tiny-video-samples \
  --repo-type dataset \
  --local-dir /data/tiny-video-samples
```

Expected layout:

```text
/data/tiny-video-samples/
  meta.jsonl
  data/*.mp4
```

FLUX examples use either:

- precomputed encodings: a Hugging Face `datasets` directory with
  `t5_encodings`, `clip_encodings`, `mean`, and `logvar` fields.
- raw image-text data: a local webdataset directory, a local JSONL file, a local
  Hugging Face dataset, or a Hugging Face dataset repo.

See `primus/backends/diffusion/README.md` for checkpoint and data layout details.

## Run

Set the shared `torchrun` environment first:

```bash
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
```

### Wan Pretrain

```bash
DATASET_PATH=/data/tiny-video-samples/meta.jsonl \
DATA_FOLDER=/data/tiny-video-samples/data \
ATTENTION_BACKEND=flash_attn_aiter \
SP_SIZE=1 \
MAX_STEPS=10 \
torchrun \
  --nnodes="$NNODES" --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  --nproc_per_node="$GPUS_PER_NODE" \
  -m primus.cli.main train pretrain \
  --config examples/diffusion/configs/MI355X/wan2.2_ti2v_5b-pretrain.yaml
```

Use `SP_SIZE=4` or `SP_SIZE=8` to enable Ulysses sequence parallelism
when the model head count supports it.

### FLUX Precomputed Pretrain

```bash
DATASET_PATH=/data/flux_precomputed \
EMPTY_ENCODINGS_PATH=/data/flux_empty_encodings \
MAX_STEPS=10 \
torchrun \
  --nnodes="$NNODES" --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  --nproc_per_node="$GPUS_PER_NODE" \
  -m primus.cli.main train pretrain \
  --config examples/diffusion/configs/MI355X/flux.1_dev_t2i-pretrain.yaml
```

Set `PRETRAINED_PATH=/path/to/flux1-dev.safetensors` only when you want to
initialize the DiT from existing FLUX weights. Empty `PRETRAINED_PATH` means
random initialization.

### FLUX Raw Image-Text Pretrain

```bash
DATASET=cc12m-test \
DATASET_PATH=/data/cc12m_test \
T5_ENCODER=google/t5-v1_1-xxl \
CLIP_ENCODER=openai/clip-vit-large-patch14 \
VAE_CHECKPOINT=black-forest-labs/FLUX.1-dev/ae.safetensors \
MAX_STEPS=10 \
torchrun \
  --nnodes="$NNODES" --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  --nproc_per_node="$GPUS_PER_NODE" \
  -m primus.cli.main train pretrain \
  --config examples/diffusion/configs/MI355X/flux.1_dev_t2i-raw-pretrain.yaml
```

For raw training, `T5_ENCODER`, `CLIP_ENCODER`, and `VAE_CHECKPOINT` can be local
paths or Hugging Face identifiers.

### Wan Posttrain

```bash
INIT_CHECKPOINT=/models/Wan2.2-TI2V-5B \
DATASET_PATH=/data/tiny-video-samples/meta.jsonl \
DATA_FOLDER=/data/tiny-video-samples/data \
MAX_STEPS=10 \
torchrun \
  --nnodes="$NNODES" --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  --nproc_per_node="$GPUS_PER_NODE" \
  -m primus.cli.main train posttrain \
  --config examples/diffusion/configs/MI355X/wan2.2_ti2v_5b-posttrain.yaml
```

The MI355X configs use Primus-style override sections such as `training`,
`data`, `parallelism`, `optimizer`, `runtime`, and `metrics`. The diffusion
adapter normalizes those sections into diffusion model/dataset/trainer
arguments at runtime.
