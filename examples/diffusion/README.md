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

## FLUX.1-schnell Raw Image-Text

Raw mode loads image-text samples and runs frozen T5, CLIP, and FLUX AE online.
The default `DATASET=cc12m-test` uses the Hugging Face dataset
`zirui3/cc12m-test`, so no dataset preprocessing is required for a smoke test.

Download the encoders and autoencoder before launching training:

```bash
huggingface-cli download google/t5-v1_1-xxl \
  --local-dir /models/t5-v1_1-xxl
huggingface-cli download openai/clip-vit-large-patch14 \
  --local-dir /models/clip-vit-large-patch14
huggingface-cli download black-forest-labs/FLUX.1-dev ae.safetensors \
  --local-dir /models/FLUX.1-dev
```

Launch raw training:

```bash
T5_ENCODER=/models/t5-v1_1-xxl \
CLIP_ENCODER=/models/clip-vit-large-patch14 \
VAE_CHECKPOINT=/models/FLUX.1-dev/ae.safetensors \
MAX_STEPS=10 \
torchrun \
  --nnodes="$NNODES" --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  --nproc_per_node="$GPUS_PER_NODE" \
  -m primus.cli.main train pretrain \
  --config examples/diffusion/configs/MI355X/flux.1_schnell_t2i-raw-pretrain.yaml
```

To use a local WebDataset directory instead, set `DATASET_PATH=/path/to/tars`.
To use the full Hugging Face dataset directly, add `DATASET=cc12m-wds` to the
launch command and omit `DATASET_PATH`.

To run FLUX.1-dev, use the same training example shape and set the model preset
to `flux.1_dev_t2i.yaml`. FLUX.1-dev has a guidance embedding module;
FLUX.1-schnell does not.

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
  --config examples/diffusion/configs/MI355X/flux.1_schnell_t2i-raw-pretrain.yaml

python3 runner/helpers/hooks/train/pretrain/diffusion/prepare.py \
  --config examples/diffusion/configs/MI355X/wan2.2_ti2v_5b-pretrain.yaml
```

On success the hook prints `env.PREPARED=1`.
