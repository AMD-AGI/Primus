# Primus Diffusion Backend

`diffusion` integrates PyTorch diffusion-model training as an independent Primus
backend. Primus provides the config/launch entrypoint, while model, dataset,
attention, and FSDP2 training logic are owned by the in-tree Wan implementation
under `primus/backends/diffusion`.

All runtime code resolves through the Primus namespace, for example
`primus.backends.diffusion.models` and `primus.backends.diffusion.trainers`,
so Wan training is a first-class part of the Primus `diffusion` backend.

Supported scope:

- Model implementation: `wan` for Wan2.1/Wan2.2 and `flux` for FLUX.1-dev T2I.
- Trainer: FSDP2 only.
- Sequence parallelism: Ulysses SP via `trainer.args.sp_size` for Wan. FLUX
  currently requires `sp_size=1`.

Wan-specific dependencies are kept out of top-level Primus requirements. See
`runner/helpers/hooks/train/pretrain/diffusion/requirements-diffusion.txt`.

## Data

Wan training reads a jsonl metadata file and a video folder:

```jsonl
{"prompt": "text prompt", "video": "example.mp4"}
```

The default small dataset for smoke tests is the Hugging Face dataset
`zirui3/tiny-video-samples`:

```bash
huggingface-cli download zirui3/tiny-video-samples \
  --repo-type dataset \
  --local-dir /data/tiny-video-samples
```

This produces:

```text
/data/tiny-video-samples/
  meta.jsonl
  data/*.mp4
```

Use these public config fields to point training at another dataset:

```yaml
data:
  dataset_path: /path/to/meta.jsonl
  data_folder: /path/to/videos
  video_backend: imageio
```

### FLUX Checkpoints And Data

FLUX supports two training-data modes:

- `dataset_type: precomputed`: samples already contain T5 encodings, CLIP
  encodings, and VAE latent statistics. This is the fastest path and matches the
  MLPerf/TorchTitan preprocessed recipe.
- `dataset_type: raw`: samples contain original image-text pairs. Primus loads
  frozen T5, CLIP, and FLUX autoencoder modules and computes the encodings online
  during training. This is useful for bring-up and custom datasets, but it is
  slower and uses more memory.

#### FLUX Pretrain Checkpoints

For FLUX pretraining, the DiT checkpoint is optional:

```yaml
model:
  config:
    load_from_pretrained_path: ""  # empty means random DiT init
```

Raw image-text training needs frozen encoders/autoencoder. These can be local
paths or Hugging Face identifiers in the YAML:

```yaml
model:
  config:
    encoder:
      t5_encoder: google/t5-v1_1-xxl
      clip_encoder: openai/clip-vit-large-patch14
      autoencoder: black-forest-labs/FLUX.1-dev/ae.safetensors
```

`t5_encoder` and `clip_encoder` are loaded with Hugging Face `from_pretrained`.
`autoencoder` accepts either a local safetensors path or `repo_id/filename`;
for example `black-forest-labs/FLUX.1-dev/ae.safetensors`.

The shipped precomputed example expects these paths through environment
variables:

```text
DATASET_PATH=/path/to/cc12m_preprocessed
EMPTY_ENCODINGS_PATH=/path/to/empty_encodings
```

The example config is:

```text
examples/diffusion/configs/MI355X/flux.1_dev_t2i-pretrain.yaml
```

#### FLUX Raw Data

Raw FLUX data is selected from YAML. The supported built-in dataset names mirror
TorchTitan's FLUX recipe:

```yaml
data:
  dataset_type: raw
  dataset: cc12m-test   # small smoke-test webdataset
```

For full CC12M webdataset training, use:

```yaml
data:
  dataset_type: raw
  dataset: cc12m-wds    # maps to Hugging Face dataset pixparse/cc12m-wds
```

Do not use full `cc12m-wds` as the default smoke test; it is large. For
`cc12m-test`, pass a local webdataset directory explicitly:

```text
DATASET=cc12m-test
DATASET_PATH=/path/to/cc12m_test
```

The raw FLUX example is:

```text
examples/diffusion/configs/MI355X/flux.1_dev_t2i-raw-pretrain.yaml
```

Raw FLUX data can also be a local JSONL file:

```jsonl
{"image": "000001.jpg", "prompt": "a caption"}
{"image": "000002.jpg", "txt": "another caption"}
```

Use:

```yaml
data:
  dataset_type: raw
  dataset_format: jsonl
  dataset_path: /path/to/meta.jsonl
  data_folder: /path/to/images
  img_size: 256
```

You can also point `dataset_path` at a local webdataset directory or a Hugging
Face dataset repo, depending on `dataset_format`:

```yaml
data:
  dataset_type: raw
  dataset_format: hf_repo
  dataset_path: pixparse/cc12m-wds
```

In short, the intended design is: checkpoint and dataset references live in YAML;
each value can be a local path when assets are staged manually, or a Hugging Face
identifier when `transformers`, `huggingface_hub`, or `datasets` should resolve
it. This keeps runs reproducible while still allowing local cache/download
behavior.

## Checkpoints

Each Wan model is a single Hugging Face repo that already bundles everything Wan
training needs: the DiT weights, the UMT5-XXL text encoder
(`models_t5_umt5-xxl-enc-bf16.pth`), the VAE (`Wan2.1_VAE.pth` /
`Wan2.2_VAE.pth`), and the tokenizer under `google/umt5-xxl`. Download the
model(s) you plan to train:

| Model | Preset | Hugging Face repo |
| --- | --- | --- |
| Wan2.1-T2V-1.3B | `wan2.1_t2v_1.3b.yaml` | [Wan-AI/Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) |
| Wan2.2-TI2V-5B | `wan2.2_ti2v_5b.yaml` | [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) |
| Wan2.1-T2V-14B | (no shipped preset) | [Wan-AI/Wan2.1-T2V-14B](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) |

```bash
# Wan2.1-T2V-1.3B
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B \
  --local-dir /models/Wan2.1-T2V-1.3B

# Wan2.2-TI2V-5B
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B \
  --local-dir /models/Wan2.2-TI2V-5B
```

A downloaded Wan repo looks like this (the T5 encoder, VAE, and tokenizer are
shipped inside the same repo, so no separate download is required):

```text
/models/Wan2.1-T2V-1.3B/
  config.json
  diffusion_pytorch_model*.safetensors   # DiT weights
  models_t5_umt5-xxl-enc-bf16.pth         # T5 (UMT5-XXL) text encoder
  Wan2.1_VAE.pth                          # VAE (Wan2.2_VAE.pth in the 5B repo)
  google/umt5-xxl/                        # tokenizer
```

The model presets reference these paths by default. Override any asset with
environment variables:

```bash
export PRETRAINED_PATH=/models/Wan2.1-T2V-1.3B   # pretrain DiT init
export INIT_CHECKPOINT=/models/Wan2.1-T2V-1.3B   # post-train/SFT DiT init
export TEXT_TOKENIZER=/models/Wan2.1-T2V-1.3B/google/umt5-xxl
export TEXT_ENCODER=/models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth
export VAE_CHECKPOINT=/models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth
```

Pretrain presets use `PRETRAINED_PATH`; post-train SFT presets use
`INIT_CHECKPOINT` for DiT initialization. For Wan2.2-5B use the matching
`/models/Wan2.2-TI2V-5B` directory and `Wan2.2_VAE.pth`.

## Pre-flight validation

The diffusion prepare hook validates the prepared assets; it does **not**
download them. Download the dataset and checkpoints above first, then run the
hook to confirm the configured dataset, tokenizer, and DiT/T5/VAE paths exist
before launching distributed training:

```bash
# pretrain config (validates modules.pre_trainer)
python3 runner/helpers/hooks/train/pretrain/diffusion/prepare.py \
  --config examples/diffusion/configs/MI355X/wan2.1_t2v_1.3b-pretrain.yaml

# FLUX precomputed config; DiT checkpoint is optional
python3 runner/helpers/hooks/train/pretrain/diffusion/prepare.py \
  --config examples/diffusion/configs/MI355X/flux.1_dev_t2i-pretrain.yaml

# FLUX raw image-text config; requires T5/CLIP access and FLUX AE checkpoint
python3 runner/helpers/hooks/train/pretrain/diffusion/prepare.py \
  --config examples/diffusion/configs/MI355X/flux.1_dev_t2i-raw-pretrain.yaml

# post-train config (validates modules.post_trainer)
python3 runner/helpers/hooks/train/posttrain/diffusion/prepare.py \
  --config examples/diffusion/configs/MI355X/wan2.1_t2v_1.3b-posttrain.yaml
```

On success it prints `env.PREPARED=1`. Set `SKIP_PREPARE=1` to bypass the check
for debugging.

## Launch

Training runs through the Primus CLI (`primus.cli.main train <pretrain|posttrain>`)
under `torchrun`. Single-node and multi-node share **one** launch command: the
same `torchrun` invocation runs on every node, parameterized by the standard
rendezvous variables. The defaults below give a single-node 8-GPU run.

```bash
# distributed knobs (defaults = single node, 8 GPUs)
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}

torchrun \
  --nnodes="$NNODES" --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  --nproc_per_node="$GPUS_PER_NODE" \
  -m primus.cli.main train pretrain --config /path/to/wan_config.yaml
```

- **Single node**: run as-is (the defaults above).
- **Multi-node**: run the same command on each node with a shared
  `MASTER_ADDR`/`MASTER_PORT` (a routable IP of node rank 0) and a distinct
  `NODE_RANK` per node. World size is `NNODES * GPUS_PER_NODE`.
- **Post-train**: identical command with `train posttrain` and a posttrain config.

Useful runtime knobs:

- `trainer.args.attention_backend`: defaults to `flash_attn_aiter` for Wan
  training on ROCm. Use `sdpa` as the portable fallback when AITER flash
  attention is unavailable.
- `trainer.args.sp_size`: Ulysses sequence parallel size. It must divide the
  model attention head count; for example Wan2.1-1.3B supports `sp_size=4` but
  not `sp_size=8`.
- `trainer.args.dp_replicate`: data parallel replication size.
- `FIXED_TIMESTEP` and `FIXED_SEED`: optional debug variables for reproducible
  loss-alignment checks.

On ROCm clusters, point compiler/cache directories to a large filesystem:

```bash
export TMPDIR=/path/to/large/tmp
export TRITON_CACHE_DIR=/path/to/large/cache/triton
export TORCHINDUCTOR_CACHE_DIR=/path/to/large/cache/inductor
export AMD_COMGR_CACHE_DIR=/path/to/large/cache/comgr
```

## Primus-Style Configs

New examples should use Primus-style override sections and let the diffusion
adapter normalize them into Wan args. See:

```text
examples/diffusion/configs/MI355X/wan2.1_t2v_1.3b-pretrain.yaml
examples/diffusion/configs/MI355X/wan2.2_ti2v_5b-pretrain.yaml
examples/diffusion/configs/MI355X/wan2.1_t2v_1.3b-posttrain.yaml
examples/diffusion/configs/MI355X/wan2.2_ti2v_5b-posttrain.yaml
```

Minimal Wan2.1-1.3B shape:

```yaml
work_group: local
user_name: local
exp_name: wan2.1_t2v_1.3b-pretrain
workspace: ./output

platform:
  config: platform_local.yaml

modules:
  pre_trainer:
    framework: diffusion
    config: pre_trainer.yaml
    model: wan2.1_t2v_1.3b.yaml
    overrides:
      metrics:
        log_freq: 1
        enable_wandb: false
      training:
        local_batch_size: 1
        steps: 100
        gradient_accumulation_steps: 1
        output_dir: ./output/wan2.1_t2v_1.3b-pretrain
        save_steps: 0
      data:
        dataset_path: /data/tiny-video-samples/meta.jsonl
        data_folder: /data/tiny-video-samples/data
        text_tokenizer: /models/Wan2.1-T2V-1.3B/google/umt5-xxl
        height: 480
        width: 832
      parallelism:
        sp_size: 1
        dp_replicate: 1
      runtime:
        attention_backend: flash_attn_aiter
        report_to: none
```

Minimal Wan2.2-5B uses the same shape with `model: wan2.2_ti2v_5b.yaml` and
the Wan2.2 tokenizer path:

```yaml
modules:
  pre_trainer:
    framework: diffusion
    config: pre_trainer.yaml
    model: wan2.2_ti2v_5b.yaml
    overrides:
      data:
        text_tokenizer: /models/Wan2.2-TI2V-5B/google/umt5-xxl
```
