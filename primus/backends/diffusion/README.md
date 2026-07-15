# Primus Diffusion Backend

`diffusion` is an in-tree PyTorch backend for Wan video training and FLUX
text-to-image training. Primus owns config loading and launch; this backend owns
model construction, datasets, attention selection, FSDP2 wrapping, and the
training loop.

## Supported Scope

- Models: `wan`, `flux.1-schnell`, and `flux.1-dev`.
- Trainer: FSDP2.
- Wan sequence parallelism: supported through `sp_size`.
- FLUX sequence parallelism: not supported; keep `sp_size: 1`.

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
```

FLUX.1-schnell and FLUX.1-dev are separate presets. They share the same main
transformer shape, but `flux.1-dev` has a guidance embedding module while
`flux.1-schnell` does not. Use checkpoints that match the selected preset:

```text
flux.1_schnell_t2i.yaml -> flux1-schnell.safetensors
flux.1_dev_t2i.yaml     -> flux1-dev.safetensors
```

This follows the TorchTitan-style separation where `flux_schnell()` is its own
config/preset and architecture differences such as `guidance_embed=False` are
part of that preset, not a launch-time switch.

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
