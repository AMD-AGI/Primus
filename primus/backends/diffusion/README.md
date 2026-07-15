# Primus Diffusion Backend

`diffusion` is an in-tree PyTorch backend for Wan video training and FLUX
text-to-image training. Primus owns config loading and launch; backend code owns
model construction, datasets, attention selection, FSDP2 wrapping, and training.

## Supported Scope

- Models: `wan` and `flux`.
- Trainer: FSDP2.
- FLUX variants: `flux-schnell` and `flux-dev`.
- Wan sequence parallelism: supported through `sp_size`.
- FLUX sequence parallelism: not supported; keep `sp_size: 1`.

Install backend dependencies with:

```bash
pip install -r runner/helpers/hooks/train/pretrain/diffusion/requirements-diffusion.txt
```

## FLUX Variants

The default FLUX examples run `flux-schnell`.

`flux-dev` and `flux-schnell` use the same main transformer shape, but they are
not just different `FLUX_GUIDANCE` values. `flux-dev` has a guidance embedding
module and requires a guidance tensor; `flux-schnell` does not. Checkpoints must
match the selected variant.

Use the same model preset for both variants:

```bash
# Default
export FLUX_MODEL_VARIANT=flux-schnell
export PRETRAINED_PATH=/path/to/flux1-schnell.safetensors  # optional

# FLUX.1-dev
export FLUX_MODEL_VARIANT=flux-dev
export PRETRAINED_PATH=/path/to/flux1-dev.safetensors      # optional
```

## FLUX Data

FLUX has two dataset modes.

### Precomputed

`dataset_type: precomputed` expects a Hugging Face `datasets.save_to_disk()`
directory. Each sample must contain:

```text
t5_encodings, clip_encodings, mean, logvar
```

Prompt dropout needs fixed empty prompt encodings:

```text
EMPTY_ENCODINGS_PATH/
  t5_empty.npy
  clip_empty.npy
```

The empty encoding shapes must match one sample's T5 and CLIP encoding shapes.

Launch:

```bash
DATASET_PATH=/data/flux_precomputed \
EMPTY_ENCODINGS_PATH=/data/flux_empty_encodings \
torchrun \
  --nnodes="${NNODES:-1}" --node_rank="${NODE_RANK:-0}" \
  --master_addr="${MASTER_ADDR:-127.0.0.1}" --master_port="${MASTER_PORT:-29500}" \
  --nproc_per_node="${GPUS_PER_NODE:-8}" \
  -m primus.cli.main train pretrain \
  --config examples/diffusion/configs/MI355X/flux.1_schnell_t2i-pretrain.yaml
```

### Raw Image-Text

`dataset_type: raw` loads images and prompts, then runs frozen T5, CLIP, and
FLUX AE online. Encoder values can be local paths or Hugging Face identifiers.

```bash
DATASET=cc12m-test \
DATASET_PATH=/data/cc12m_test \
T5_ENCODER=google/t5-v1_1-xxl \
CLIP_ENCODER=openai/clip-vit-large-patch14 \
VAE_CHECKPOINT=black-forest-labs/FLUX.1-dev/ae.safetensors \
torchrun \
  --nnodes="${NNODES:-1}" --node_rank="${NODE_RANK:-0}" \
  --master_addr="${MASTER_ADDR:-127.0.0.1}" --master_port="${MASTER_PORT:-29500}" \
  --nproc_per_node="${GPUS_PER_NODE:-8}" \
  -m primus.cli.main train pretrain \
  --config examples/diffusion/configs/MI355X/flux.1_schnell_t2i-raw-pretrain.yaml
```

Supported raw formats:

- `DATASET=cc12m-test`: local WebDataset directory with `.tar` shards.
- `DATASET=cc12m-wds`: Hugging Face `pixparse/cc12m-wds`, unless
  `DATASET_PATH` overrides it.
- `DATASET_FORMAT=jsonl`: local JSONL with `image` plus `prompt`/`txt`/`caption`
  fields; set `DATA_FOLDER` when image paths are relative.
- `DATASET_FORMAT=hf_dataset` or `hf_repo`: local `load_from_disk()` dataset or
  Hugging Face dataset repo.

## Prepare Check

Validate configured paths before launching:

```bash
python3 runner/helpers/hooks/train/pretrain/diffusion/prepare.py \
  --config examples/diffusion/configs/MI355X/flux.1_schnell_t2i-pretrain.yaml

python3 runner/helpers/hooks/train/pretrain/diffusion/prepare.py \
  --config examples/diffusion/configs/MI355X/flux.1_schnell_t2i-raw-pretrain.yaml
```

On success the hook prints `env.PREPARED=1`.

## Wan Data

Wan examples read JSONL metadata plus a media directory:

```jsonl
{"prompt": "text prompt", "video": "example.mp4"}
```

```bash
huggingface-cli download zirui3/tiny-video-samples \
  --repo-type dataset \
  --local-dir /data/tiny-video-samples
```

Use the matching Wan config under `examples/diffusion/configs/MI355X/` and set
`DATASET_PATH`, `DATA_FOLDER`, and checkpoint/tokenizer paths as needed.
