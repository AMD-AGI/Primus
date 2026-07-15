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

## FLUX.1-schnell Precomputed

Use precomputed mode when T5, CLIP, and VAE outputs are already saved in a
Hugging Face `datasets.save_to_disk()` directory. Each sample must contain:

```text
t5_encodings, clip_encodings, mean, logvar
```

If `PROMPT_DROPOUT_PROB > 0`, `EMPTY_ENCODINGS_PATH` must contain
`t5_empty.npy` and `clip_empty.npy` with shapes matching one sample's T5 and
CLIP encodings.

`PRETRAINED_PATH` is optional. For schnell initialization, point it at
`flux1-schnell.safetensors` or a directory containing that file.

```bash
DATASET_PATH=/data/flux_precomputed \
EMPTY_ENCODINGS_PATH=/data/flux_empty_encodings \
PRETRAINED_PATH=/models/flux1-schnell.safetensors \
MAX_STEPS=10 \
torchrun \
  --nnodes="$NNODES" --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  --nproc_per_node="$GPUS_PER_NODE" \
  -m primus.cli.main train pretrain \
  --config examples/diffusion/configs/MI355X/flux.1_schnell_t2i-pretrain.yaml
```

## FLUX.1-schnell Raw Image-Text

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

To run FLUX.1-dev, use the same training example shape with
`model: flux.1_dev_t2i.yaml` and a matching `flux1-dev.safetensors` checkpoint.
FLUX.1-dev has a guidance embedding module; FLUX.1-schnell does not.

## Wan Data

Wan examples use a JSONL metadata file plus a media directory:

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

Download Wan checkpoints separately and set the model paths used by the selected
config. For Wan2.2 TI2V 5B, the default paths can be overridden with:

```bash
export PRETRAINED_PATH=/models/Wan2.2-TI2V-5B
export INIT_CHECKPOINT=/models/Wan2.2-TI2V-5B
export TEXT_TOKENIZER=/models/Wan2.2-TI2V-5B/google/umt5-xxl
export TEXT_ENCODER=/models/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth
export VAE_CHECKPOINT=/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
```

## Wan Pretrain

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

Use `SP_SIZE=4` or `SP_SIZE=8` when the model head count supports it.

## Wan Posttrain

```bash
INIT_CHECKPOINT=/models/Wan2.2-TI2V-5B \
DATASET_PATH=/data/tiny-video-samples/meta.jsonl \
DATA_FOLDER=/data/tiny-video-samples/data \
ATTENTION_BACKEND=flash_attn_aiter \
SP_SIZE=1 \
MAX_STEPS=10 \
torchrun \
  --nnodes="$NNODES" --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  --nproc_per_node="$GPUS_PER_NODE" \
  -m primus.cli.main train posttrain \
  --config examples/diffusion/configs/MI355X/wan2.2_ti2v_5b-posttrain.yaml
```

## Prepare Check

Validate configured paths before launching:

```bash
python3 runner/helpers/hooks/train/pretrain/diffusion/prepare.py \
  --config examples/diffusion/configs/MI355X/flux.1_schnell_t2i-pretrain.yaml

python3 runner/helpers/hooks/train/pretrain/diffusion/prepare.py \
  --config examples/diffusion/configs/MI355X/flux.1_schnell_t2i-raw-pretrain.yaml

python3 runner/helpers/hooks/train/pretrain/diffusion/prepare.py \
  --config examples/diffusion/configs/MI355X/wan2.2_ti2v_5b-pretrain.yaml
```

On success the hook prints `env.PREPARED=1`.
