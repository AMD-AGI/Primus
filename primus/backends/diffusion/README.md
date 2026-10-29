# Primus Diffusion Backend

`diffusion` is an in-tree PyTorch backend for Wan and FLUX training.
Primus owns config loading and launch; this backend owns model construction,
datasets, attention selection, FSDP2 wrapping, and the training loop.

## Supported Scope

- Native models: `wan`, `flux`, `flux.1-dev`, and `flux.1-schnell`.
- AMDiffusionBenchmark workload `wan2.1-i2v`, built by `models/registrations/wan.py`.
  It extends the native Wan DiT with image-to-video cross-attention. That path
  stays off the existing `wan` trainer unless `model_type` is `i2v`.
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
benchmark_wan2_1_i2v.yaml
```

`wan2.1-i2v` is the post-train preset. It trains the native Wan2.1 DiT
with CLIP cross-attention, the Wan2.1 VAE, UMT5, and the CLIP vision tower.

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
# Wan2.1 I2V 14B, Wild-Heart/Disney-VideoGeneration-Dataset
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
