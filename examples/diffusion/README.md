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
./primus-cli direct -- train pretrain \
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
./primus-cli direct -- train pretrain \
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
./primus-cli direct -- train posttrain \
  --config examples/diffusion/configs/MI355X/wan2.2_ti2v_5b-posttrain.yaml
```

## HY-WorldPlay AR

Train the in-tree WorldPlay AR model on video + camera + action. Both runs
use the same encoded DL3DV subset, trainer, and loss. They write separate
checkpoints. These examples do not distill, run WorldCompass RL, or do
inference.

| | Pretrain | Posttrain |
|---|---|---|
| Config | `examples/diffusion/configs/MI300X/worldplay_ar_8b-pretrain.yaml` | `examples/diffusion/configs/MI300X/worldplay_ar_8b-posttrain.yaml` |
| Model preset | `worldplay_ar_8b.yaml` | `worldplay_ar_8b_sft.yaml` |
| HunyuanVideo-1.5 480p I2V | Loaded | Loaded |
| Released `HY-WorldPlay` `ar_model` | Not loaded | Loaded on top of the Hunyuan base |
| New action modules | Zero-initialized | Already trained inside `ar_model` |
| Data | `/data/dataset/worldplay_dl3dv_100` | Same files |
| Checkpoints | `./output/worldplay-ar-8b-pretrain` | `./output/worldplay-ar-8b-posttrain` |

- **Pretrain:** load HunyuanVideo-1.5, add zero-initialized WorldPlay action
  modules, and train them on encoded DL3DV.
- **Posttrain:** load HunyuanVideo-1.5 plus Tencent’s trained WorldPlay AR
  checkpoint, then fine-tune it on encoded DL3DV. This is recommended for the
  small 100-scene set.

### How it is integrated

WorldPlay is a new `worldplay` registration on Primus’s **in-tree PyTorch
diffusion backend** — the same backend as Wan and FLUX. It is **not** Wan with
different checkpoints.

**Reused from Primus**

| Piece | Reuse |
|---|---|
| Launch | `primus-cli … train pretrain` or `train posttrain`, YAML examples |
| Trainer | FSDP2 loop, AdamW, logging, checkpoint skip (`save_strategy: none`) |
| Mesh | Device mesh + DP sampler; default `sp_size=1` (FSDP2 still shards 8 GPUs) |
| Config | `DiffusionArgBuilder` public sections (`data`, `training`, `parallelism`) |
| Prepare hook | Path checks before launch |

**New in Primus (WorldPlay-specific)**

| Piece | Role |
|---|---|
| `WorldPlayForTraining` | `GenAIModel.forward_train` adapter |
| `WorldPlayARTrainPipeline` | Flow-match noise, I2V concat, camera/action kwargs, memory-window loss |
| `WorldPlayLatentDataset` | Precomputed latents + poses; actions derived from camera motion |
| `worldplay_ar_8b.yaml` | Hunyuan 480p I2V load; action modules stay zero-init |
| `worldplay_ar_8b_sft.yaml` | Pretrain preset plus the released AR overlay |
| `prepare_worldplay_dl3dv.py` | Encode a 100 / 500 / 1000-scene DL3DV subset |

**Fully in-tree**

The Hunyuan AR transformer, camera RoPE, action embeddings,
and offline encoding helpers are maintained inside Primus. Primus loads
`ARHunyuanVideo_1_5_DiffusionTransformer.from_pretrained(...)` from its
diffusion backend. Posttrain then overlays
`ar_model/diffusion_pytorch_model.safetensors`. Pretrain skips that overlay.
Hunyuan VAE / Qwen-VL / ByT5 /
SigLIP run **only in the offline encode script**, not in the FSDP train graph.
No HY-WorldPlay checkout, submodule, or `WORLDPLAY_SOURCE_PATH` is required.

### Layout on disk

| Kind | Path |
|---|---|
| Weights | `/data/models` (`HF_HOME`) |
| Encoded SFT data | `/data/dataset/worldplay_dl3dv_100` |
| Raw DL3DV zips | `/data/dataset/dl3dv_raw` |
| HY-WorldPlay code | `primus/backends/diffusion/models/worldplay` |

### 1. Packages (ROCm image)

Do **not** reinstall PyTorch or `requirements-diffusion.txt`.

```bash
python -m pip install imageio==2.37.0 remote-pdb
```

### 2. Download weights into `/data/models`

Need Hugging Face access to `tencent/HunyuanVideo-1.5`, `tencent/HY-WorldPlay`,
and gated `black-forest-labs/FLUX.1-Redux-dev` (SigLIP, encode only).

```bash
export HF_TOKEN=hf_xxxxxxxx
mkdir -p /data/models /data/dataset

# HY-WorldPlay download_models.py, with Primus paths under /data/models.
python examples/diffusion/scripts/download_worldplay_models.py --hf_token "${HF_TOKEN}"
# creates stable links:
#   /data/models/HunyuanVideo-1.5
#   /data/models/HY-WorldPlay
```

### 3. Encode a DL3DV subset into `/data/dataset`

Request access to [`DL3DV/DL3DV-ALL-480P`](https://huggingface.co/datasets/DL3DV/DL3DV-ALL-480P).
`--scene-count` is 100, 500, or 1000 scenes from `--batch 1K` (not the whole 1K zip dump).

```bash
python examples/diffusion/scripts/prepare_worldplay_dl3dv.py \
  --hunyuan-checkpoint-path /data/models/HunyuanVideo-1.5 \
  --batch 1K \
  --scene-count 100 \
  --raw-dir /data/dataset/dl3dv_raw \
  --output-dir /data/dataset/worldplay_dl3dv_100
```

Writes `train.json`, `latents/`, `poses/`, and `neg_prompts/*.pt`. For a tiny
smoke, pass `--scene-hash HASH` (repeatable) instead of `--scene-count`.

### 4. Check paths, then train

```bash
export DATASET_PATH=/data/dataset/worldplay_dl3dv_100/train.json
export HY_NEG_PROMPT=/data/dataset/worldplay_dl3dv_100/neg_prompts/hunyuan_neg_prompt.pt
export HY_NEG_BYT5_PROMPT=/data/dataset/worldplay_dl3dv_100/neg_prompts/hunyuan_neg_byt5_prompt.pt
```

Posttrain:

```bash
python3 runner/helpers/hooks/train/pretrain/diffusion/prepare.py \
  --config examples/diffusion/configs/MI300X/worldplay_ar_8b-posttrain.yaml

SP_SIZE=1 MAX_STEPS=30 \
./primus-cli direct -- train posttrain \
  --config examples/diffusion/configs/MI300X/worldplay_ar_8b-posttrain.yaml
```

Pretrain:

```bash
python3 runner/helpers/hooks/train/pretrain/diffusion/prepare.py \
  --config examples/diffusion/configs/MI300X/worldplay_ar_8b-pretrain.yaml

SP_SIZE=1 MAX_STEPS=30 \
./primus-cli direct -- train pretrain \
  --config examples/diffusion/configs/MI300X/worldplay_ar_8b-pretrain.yaml
```

On MI355X, use the same filenames under `examples/diffusion/configs/MI355X/`.

Default is AdamW + FSDP2 + `sp_size=1`. Token SP (`SP_SIZE=8`) and Muon are
follow-up.

## Prepare Check

Validate configured paths before launching:

```bash
python3 runner/helpers/hooks/train/pretrain/diffusion/prepare.py \
  --config examples/diffusion/configs/MI355X/flux.1_schnell_t2i-raw-pretrain.yaml

python3 runner/helpers/hooks/train/pretrain/diffusion/prepare.py \
  --config examples/diffusion/configs/MI355X/wan2.2_ti2v_5b-pretrain.yaml

python3 runner/helpers/hooks/train/pretrain/diffusion/prepare.py \
  --config examples/diffusion/configs/MI300X/worldplay_ar_8b-posttrain.yaml

python3 runner/helpers/hooks/train/pretrain/diffusion/prepare.py \
  --config examples/diffusion/configs/MI300X/worldplay_ar_8b-pretrain.yaml
```

On success the hook prints `env.PREPARED=1`.
