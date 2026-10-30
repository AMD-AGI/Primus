# Primus Diffusion Backend

`diffusion` is an in-tree PyTorch backend for Wan and FLUX training.
Primus owns config loading and launch; this backend owns model construction,
datasets, attention selection, FSDP2 wrapping, and the training loop.

## Supported Scope

- Native models: `wan`, `flux`, `flux.1-dev`, and `flux.1-schnell`.
- AMDiffusionBenchmark workload `hunyuan-video`, built by
  `models/registrations/hunyuan.py`. The trainable module is the Tencent transformer.
  It uses the shared benchmark trainer and does not change the native
  Wan or FLUX trainers.
- Trainer: FSDP2 for the native models. The benchmark workloads use
  `amdb_benchmark`.
- Wan sequence parallelism: supported through `sp_size`.
- FLUX and every AMDiffusionBenchmark workload keep `sp_size: 1`.
- `primus-cli` downloads those benchmark checkpoints into
  `~/cache/huggingface/hub` and the datasets into
  `~/cache/huggingface/datasets/<org>/<name>`. `PRIMUS_HF_HOME` changes the
  cache root. An explicit `HF_HOME` is used unless it is the runner's
  automatic `$DATA_PATH/huggingface`.

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
benchmark_hunyuan_video.yaml
```

`hunyuan-video` is the post-train preset. Gradient checkpointing stays
off, and a snapshot without `vae/` still trains from latents.

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
# HunyuanVideo, Wild-Heart/Disney-VideoGeneration-Dataset
./primus-cli direct -- train posttrain \
  --config examples/diffusion/configs/MI355X/benchmark_hunyuan_video-posttrain.yaml
```

`DATASET_PATH=synthetic:hunyuan` repeats one denoiser batch and skips the
dataset download. `BENCHMARK_DENOISER_ONLY=1` also skips the text encoder and
VAE. A path ending in `.pt` loads cached encodings. The Disney dataset is
pinned to commit `316b85f5d7c263260cc526a25eece96cc30e0c06` (Hugging Face PR 4,
which adds `metadata.csv`).

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
