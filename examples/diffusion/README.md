# Diffusion Examples

This directory contains launch examples for the in-tree `diffusion` backend.

## Common Launch Env

```bash
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
```

## FLUX Schnell Precomputed

Use this path when T5, CLIP, and VAE outputs are already saved in a Hugging Face
`datasets.save_to_disk()` directory. Each sample must contain:

```text
t5_encodings, clip_encodings, mean, logvar
```

If `PROMPT_DROPOUT_PROB > 0`, `EMPTY_ENCODINGS_PATH` must contain
`t5_empty.npy` and `clip_empty.npy` with shapes matching one sample's T5 and
CLIP encodings.

```bash
DATASET_PATH=/data/flux_precomputed \
EMPTY_ENCODINGS_PATH=/data/flux_empty_encodings \
MAX_STEPS=10 \
torchrun \
  --nnodes="$NNODES" --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  --nproc_per_node="$GPUS_PER_NODE" \
  -m primus.cli.main train pretrain \
  --config examples/diffusion/configs/MI355X/flux.1_schnell_t2i-pretrain.yaml
```

`PRETRAINED_PATH` is optional. For schnell initialization, point it at
`flux1-schnell.safetensors` or a directory containing that file.

## FLUX Schnell Raw Image-Text

Raw mode loads image-text samples and runs frozen T5, CLIP, and FLUX AE online.
It is slower than precomputed training but useful for bring-up and custom data.

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
  --config examples/diffusion/configs/MI355X/flux.1_schnell_t2i-raw-pretrain.yaml
```

`DATASET=cc12m-test` expects `DATASET_PATH` to be a local WebDataset directory
with `.tar` shards. `DATASET=cc12m-wds` uses the Hugging Face
`pixparse/cc12m-wds` dataset unless `DATASET_PATH` overrides it.

To run FLUX.1-dev instead of schnell, set:

```bash
export FLUX_MODEL_VARIANT=flux-dev
export PRETRAINED_PATH=/path/to/flux1-dev.safetensors
```

FLUX.1-dev has a guidance embedding module; schnell does not. Keep checkpoints
matched to the selected `FLUX_MODEL_VARIANT`.

## Wan Smoke Runs

Wan examples use a JSONL metadata file and media directory:

```bash
huggingface-cli download zirui3/tiny-video-samples \
  --repo-type dataset \
  --local-dir /data/tiny-video-samples
```

```bash
DATASET_PATH=/data/tiny-video-samples/meta.jsonl \
DATA_FOLDER=/data/tiny-video-samples/data \
MAX_STEPS=10 \
torchrun \
  --nnodes="$NNODES" --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  --nproc_per_node="$GPUS_PER_NODE" \
  -m primus.cli.main train pretrain \
  --config examples/diffusion/configs/MI355X/wan2.2_ti2v_5b-pretrain.yaml
```
