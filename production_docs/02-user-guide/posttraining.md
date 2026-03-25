# Post-Training Workflows

Post-training (supervised fine-tuning) adapts a pre-trained foundation model to new tasks or domains. In Primus, post-training runs through the **Megatron Bridge** backend using the `train posttrain` subcommand. Example YAML configurations live under `examples/megatron_bridge/configs/`.

For YAML field details, see [Megatron Bridge parameters](../03-configuration-reference/megatron-bridge-parameters.md). Related tooling: [Benchmark suite](./benchmarking.md), [Preflight diagnostics](./preflight.md), [Memory and performance projection](./projection.md).

---

## Overview: SFT vs LoRA

| Aspect | SFT (full fine-tuning) | LoRA (parameter-efficient) |
|--------|------------------------|----------------------------|
| **PEFT setting** | `peft: "none"` | `peft: lora` |
| **What is trained** | All model parameters | Low-rank adapters only |
| **Memory** | Higher | Lower |
| **Throughput** | Typically slower per step | Often faster iteration |
| **Learning rate** | Lower: roughly `5e-6` to `1e-5` | Higher: roughly `1e-4` to `5e-4` |
| **Typical use** | Maximum adaptation when memory allows | Limited GPU memory, many task-specific adapters, rapid experimentation |

---

## Quick start commands

General form:

```bash
./primus-cli <mode> -- train posttrain --config <path-to-yaml>
```

From a clone of the Primus repository, the same entrypoint is often invoked as `./runner/primus-cli`.

### Direct mode (bare metal / local ROCm)

```bash
# SFT — example: Qwen3 32B on MI355X
./runner/primus-cli direct train posttrain \
  --config ./examples/megatron_bridge/configs/MI355X/qwen3_32b_sft_posttrain.yaml

# LoRA — same model family
./runner/primus-cli direct train posttrain \
  --config ./examples/megatron_bridge/configs/MI355X/qwen3_32b_lora_posttrain.yaml
```

### Container mode

```bash
./runner/primus-cli container --image rocm/primus:latest \
  train posttrain \
  --config ./examples/megatron_bridge/configs/MI355X/qwen3_32b_sft_posttrain.yaml
```

**Prerequisites:** AMD ROCm (recommended ≥ 7.0), Docker with ROCm support (optional but typical), Instinct-class GPUs (for example MI300X, MI355X). Quick check: `rocm-smi && docker --version`.

---

## Configuration reference

These keys are commonly set under `modules.post_trainer.overrides` in your experiment YAML (see the examples under `examples/megatron_bridge/configs/`).

| Area | Parameters | Notes |
|------|------------|--------|
| **Method** | `peft` | `"none"` for SFT; `lora` for LoRA. |
| **Learning rate** | `finetune_lr`, `min_lr`, `lr_warmup_iters`, `lr_decay_iters` | LoRA usually needs a higher `finetune_lr` than SFT. |
| **Precision** | `precision_config` | Typical: `bf16_mixed`. Alternatives include `fp16_mixed`, `fp32` depending on backend support. |
| **Parallelism** | `tensor_model_parallel_size`, `pipeline_model_parallel_size`, `context_parallel_size`, `sequence_parallel` | Increase TP/PP when the model does not fit on fewer GPUs. |
| **Recompute (memory)** | `recompute_granularity`, `recompute_method`, `recompute_num_layers` | Use to trade compute for activation memory (for example `recompute_granularity: full` with uniform recompute). |
| **Batching / length** | `train_iters`, `global_batch_size`, `micro_batch_size`, `seq_length` | `micro_batch_size` is per-GPU; tune with sequence length and memory. |

**Snippet — SFT (illustrative)**

```yaml
modules:
  post_trainer:
    framework: megatron_bridge
    config: sft_trainer.yaml
    model: qwen3_32b.yaml
    overrides:
      peft: "none"
      finetune_lr: 5.0e-6
      precision_config: bf16_mixed
      tensor_model_parallel_size: 1
      global_batch_size: 8
      micro_batch_size: 1
      seq_length: 8192
```

**Snippet — LoRA (illustrative)**

```yaml
modules:
  post_trainer:
    framework: megatron_bridge
    config: sft_trainer.yaml
    model: qwen3_32b.yaml
    overrides:
      peft: lora
      finetune_lr: 1.0e-4
      precision_config: bf16_mixed
      recompute_granularity: full
      recompute_method: uniform
      recompute_num_layers: 1
```

---

## MI300X configurations

Paths are relative to `examples/megatron_bridge/configs/`.

| Model | Method | Config path | TP | GBS | MBS | Seq len |
|-------|--------|-------------|----|-----|-----|---------|
| Qwen3 32B | SFT | `MI300X/qwen3_32b_sft_posttrain.yaml` | 2 | 8 | 2 | 8192 |
| Qwen3 32B | LoRA | `MI300X/qwen3_32b_lora_posttrain.yaml` | 1 | 32 | 2 | 8192 |

**Example**

```bash
./runner/primus-cli direct train posttrain \
  --config ./examples/megatron_bridge/configs/MI300X/qwen3_32b_sft_posttrain.yaml
```

**Legend:** TP = tensor parallel size; GBS = global batch size; MBS = micro batch size per GPU; Seq len = `seq_length`.

---

## MI355X configurations

Paths are relative to `examples/megatron_bridge/configs/`.

| Model | Method | Config path | TP | GBS | MBS | Seq len |
|-------|--------|-------------|----|-----|-----|---------|
| Qwen3 32B | SFT | `MI355X/qwen3_32b_sft_posttrain.yaml` | 1 | 8 | 1 | 8192 |
| Qwen3 32B | LoRA | `MI355X/qwen3_32b_lora_posttrain.yaml` | 1 | 32 | 4 | 8192 |

**Example**

```bash
./runner/primus-cli direct train posttrain \
  --config ./examples/megatron_bridge/configs/MI355X/qwen3_32b_lora_posttrain.yaml
```

---

## Best practices

### When to use SFT vs LoRA

- **Prefer SFT** when you need the strongest possible task fit, have enough GPU memory, and can afford longer runs.
- **Prefer LoRA** when memory is tight, you want fast iteration, or you plan to maintain multiple adapters for different tasks.

### Learning rates

- **SFT:** start in the `5e-6`–`1e-5` range; adjust with validation loss.
- **LoRA:** often `1e-4`–`5e-4`; still use warmup (`lr_warmup_iters`) for stability.

### Batch sizes

- SFT: starting with `global_batch_size: 8` is a reasonable default for development; scale up when stable (for example 64, 128, or higher) if memory and throughput allow.
- LoRA: larger global batches are often feasible (for example 32 in the reference configs); align `micro_batch_size` with sequence length and available HBM.
- Very long sequences (for example 8192) may require smaller micro-batches or more parallelism.

### Parallelism

- **SFT:** large models may need higher `tensor_model_parallel_size` (for example TP 8 for very large models). The bundled 32B examples use TP 2 on MI300X and TP 1 on MI355X for SFT.
- **LoRA:** adapters reduce memory pressure; lower TP is often sufficient for a given model size.

---

## Troubleshooting

### Out of memory (OOM)

**SFT**

1. Increase `tensor_model_parallel_size` (and/or pipeline parallelism for very large models).
2. Reduce `micro_batch_size` or `seq_length`.
3. Enable activation recomputation (`recompute_granularity`, `recompute_method`, `recompute_num_layers`).

**LoRA**

1. Confirm `peft: lora` is set.
2. Reduce `micro_batch_size` if OOM persists.
3. Apply the same recompute settings as for SFT.

### Training instability (loss spikes, NaNs)

1. Lower `finetune_lr`.
2. Increase `lr_warmup_iters`.
3. Keep mixed precision stable (`precision_config: bf16_mixed` where supported).
4. Monitor gradients and clipping settings if exposed by your trainer config.

### Slow training

1. Increase effective batch size where memory allows (`global_batch_size` / `micro_batch_size` tuning).
2. Revisit TP/PP/CP for your cluster topology.
3. Run [benchmarks](./benchmarking.md) or [preflight](./preflight.md) to isolate network or GPU issues.

### Configuration errors

1. Verify YAML paths and indentation.
2. Set `PRIMUS_WORKSPACE` and other environment variables expected by your team’s templates.
3. Confirm checkpoint and data paths in the merged config (use `--export_config` if your CLI supports exporting the resolved config).

---

## See also

- [Megatron Bridge parameters](../03-configuration-reference/megatron-bridge-parameters.md)
- [Benchmark suite](./benchmarking.md)
- [Preflight diagnostics](./preflight.md)
- [Memory and performance projection](./projection.md)
