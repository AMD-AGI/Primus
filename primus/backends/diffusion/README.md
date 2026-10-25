# Primus Diffusion Backend

`diffusion` is an in-tree PyTorch backend for Wan, FLUX, SDXL, Mochi, and
HunyuanVideo training.
Primus owns config loading and launch; this backend owns model construction,
datasets, attention selection, FSDP2 wrapping, and the training loop.

## Supported Scope

- Native models: `wan`, `flux`, `flux.1-dev`, and `flux.1-schnell`.
- AMDiffusionBenchmark workloads, in `amdb_benchmarks/`: `flux-dev`, `sdxl`,
  `mochi-1`, `hunyuan-video`, and `wan2.1-i2v`. These do not modify the shared
  Flux and Wan modules or the FSDP2 trainer.
- Trainer: FSDP2 for the native models. The benchmark workloads use
  `amdb_benchmark`.
- Wan sequence parallelism: supported through `sp_size`.
- FLUX and every AMDiffusionBenchmark workload keep `sp_size: 1`.
- `primus-cli` downloads those benchmark checkpoints into
  `~/cache/huggingface/hub` and the datasets into
  `~/cache/huggingface/datasets/<org>/<name>`. `PRIMUS_HF_HOME` changes the
  cache root.

Install backend dependencies with:

```bash
pip install -r runner/helpers/hooks/train/pretrain/diffusion/requirements-diffusion.txt
```

## Model Presets

Model presets live under `primus/configs/models/diffusion/`.

```text
wan2.1_t2v_1.3b.yaml
wan2.1_t2v_1.3b_sft.yaml
wan2.2_ti2v_5b.yaml
wan2.2_ti2v_5b_sft.yaml
flux.1_schnell_t2i.yaml
flux.1_dev_t2i.yaml
benchmark_flux_dev.yaml
benchmark_sdxl.yaml
benchmark_mochi_1.yaml
benchmark_hunyuan_video.yaml
benchmark_wan2_1_i2v.yaml
```

`flux-dev` and `flux.1-dev` both build the native FLUX.1-dev DiT.
`flux-dev` is the post-train preset: raw images, learning rate `1e-5`, and
resolution 512. `sdxl`, `mochi-1`, `hunyuan-video`, and `wan2.1-i2v` are
native post-train presets. Their checkpoints are the official weight files
(FLUX `flux1-dev.safetensors`, SDXL UNet, Mochi DiT and VAE encoder,
HunyuanVideo transformer, Wan2.1 I2V 14B DiT plus CLIP), not Diffusers-layout
keys.

FLUX.1-schnell and FLUX.1-dev are separate presets. They share the same main
transformer shape, but `flux.1-dev` has a guidance embedding module while
`flux.1-schnell` does not. Select the preset by choosing
`flux.1_schnell_t2i.yaml` or `flux.1_dev_t2i.yaml`.

This follows the TorchTitan-style separation where `flux_schnell()` is its own
config/preset and architecture differences such as `guidance_embed=False` are
part of that preset, not a launch-time switch.

## AMDiffusionBenchmark

Run these from the Primus root. `primus-cli` downloads the checkpoint and the
dataset before training. MI355X defaults are in each config. `LOCAL_BATCH_SIZE`
and `MAX_STEPS` override them. On MI300X or MI325X, change the config directory.
Keep sequence parallelism at 1.

```bash
# FLUX.1-dev, batch 16, 100 steps, frank-chieng/chinese_architecture_siheyuan
./primus-cli direct -- train posttrain \
  --config examples/diffusion/configs/MI355X/benchmark_flux_dev-posttrain.yaml

# SDXL, batch 12, 200 steps, frank-chieng/chinese_architecture_siheyuan
./primus-cli direct -- train posttrain \
  --config examples/diffusion/configs/MI355X/benchmark_sdxl-posttrain.yaml

# Mochi-1, batch 4, 100 steps, Wild-Heart/Disney-VideoGeneration-Dataset
./primus-cli direct -- train posttrain \
  --config examples/diffusion/configs/MI355X/benchmark_mochi_1-posttrain.yaml

# HunyuanVideo, batch 3, 100 steps, Wild-Heart/Disney-VideoGeneration-Dataset
./primus-cli direct -- train posttrain \
  --config examples/diffusion/configs/MI355X/benchmark_hunyuan_video-posttrain.yaml

# Wan2.1 I2V 14B, batch 1, 100 steps, Wild-Heart/Disney-VideoGeneration-Dataset
./primus-cli direct -- train posttrain \
  --config examples/diffusion/configs/MI355X/benchmark_wan2_1_i2v-posttrain.yaml
```

## Public Config Sections

Diffusion examples use Primus override sections that the backend converts into
model, dataset, and trainer args:

```yaml
training:
  local_batch_size: 1
  steps: 50
  output_dir: ./output/run

data:
  dataset_path: /path/to/data

parallelism:
  sp_size: 1
  dp_replicate: 1

optimizer:
  lr: 2.0e-4
  weight_decay: 0.1

lr_scheduler:
  lr_scheduler_type: constant_with_warmup
  warmup_steps: 1600

runtime:
  attention_backend: flash_attn_aiter
  gradient_checkpointing: false
  report_to: none
```

`examples/diffusion/README.md` contains data preparation notes and launch
commands for the shipped examples.
