# Parallelism Configuration Guide

Primus is a YAML-driven training framework for AMD GPUs. Megatron-LM, TorchTitan, and MaxText each expose parallelism through different configuration namespaces. This guide explains how to set parallelism and batch-related parameters, how global batch size relates to micro batch size and data parallel width, and how to choose a parallel strategy for common model sizes.

Default values cited below come from Primus module presets:

- Megatron trainer: `primus/configs/modules/megatron/trainer_base.yaml`
- Megatron model (tensor/pipeline/expert/context parallel): `primus/configs/models/megatron/language_model.yaml`
- TorchTitan: `primus/configs/modules/torchtitan/pre_trainer.yaml`

Experiment YAMLs in `examples/` often override these defaults for specific models and hardware.

---

## 1. Megatron parallelism configuration

Model-parallel degrees live on the **model** config (for example under `model:` in your experiment YAML, merged from `language_model.yaml`). Training batch and overlap settings live on the **trainer** module (`trainer_base.yaml`).

### Core parallel degrees

| Parameter | Default (Primus `language_model.yaml`) | Description |
|-----------|------------------------------------------|-------------|
| `tensor_model_parallel_size` | `1` | Tensor parallelism (TP): shards attention and MLP across this many GPUs. |
| `pipeline_model_parallel_size` | `1` | Pipeline parallelism (PP): number of pipeline stages. |
| `expert_model_parallel_size` | `1` | Expert parallelism (EP) for MoE: shards experts across this many GPUs. |
| `context_parallel_size` | `1` | Context parallelism (CP) for long sequences. |
| `sequence_parallel` | `true` | Sequence parallelism (SP); typically used with TP greater than 1. |

### Virtual pipeline (VPP) and pipeline communication

| Parameter | Description |
|-----------|-------------|
| `virtual_pipeline_model_parallel_size` | Interleaved pipeline depth (null disables VPP). |
| `num_layers_per_virtual_pipeline_stage` | Layers per virtual stage when using VPP. |
| `overlap_p2p_comm` | Overlap pipeline P2P with compute (default `true` in `trainer_base.yaml`). |

### Optimizer, FSDP, and overlap (trainer module)

| Parameter | Default (`trainer_base.yaml`) | Description |
|-----------|-------------------------------|-------------|
| `use_distributed_optimizer` | `false` | ZeRO-1 style optimizer state sharding when enabled. |
| `use_torch_fsdp2` | `false` | Full FSDP2 integration. |
| `overlap_grad_reduce` | `false` | Overlap gradient all-reduce with backward. |
| `overlap_param_gather` | `false` | Overlap parameter gathering with forward. |

Set these to `true` in your experiment when you want communication/compute overlap; many production configs enable `use_distributed_optimizer` and overlap flags for large runs.

### Data parallel size (implicit)

For Megatron, data parallel size is not a single YAML key; it is implied by the world size and the product of parallel degrees:

\[
\text{DP} = \frac{\text{world\_size}}{\text{TP} \times \text{PP} \times \text{EP}}
\]

(Adjust if you also use context parallelism or other groupings; your job’s process layout must match the configured degrees.)

### Batch parameters

| Parameter | Default (`trainer_base.yaml`) | Description |
|-----------|-------------------------------|-------------|
| `micro_batch_size` | `2` | Micro batch size per data-parallel rank (MBS). |
| `global_batch_size` | `128` | Target global batch size (GBS) across the data parallel group. |

Megatron derives **gradient accumulation** from `global_batch_size`, `micro_batch_size`, and the effective data parallel size so that:

\[
\text{GBS} = \text{MBS} \times \text{DP} \times \text{gradient\_accumulation\_steps}
\]

Equivalently:

\[
\text{gradient\_accumulation\_steps} = \frac{\text{GBS}}{\text{MBS} \times \text{DP}}
\]

You normally set `global_batch_size` and `micro_batch_size` in YAML; Megatron computes the number of accumulation steps automatically.

---

## 2. TorchTitan parallelism configuration

TorchTitan parallelism is grouped under the `parallelism:` key in the TorchTitan module (see `primus/configs/modules/torchtitan/pre_trainer.yaml`).

### `parallelism.*` parameters

| Key | Default | Description |
|-----|---------|-------------|
| `parallelism.tensor_parallel_degree` | `1` | Tensor parallelism degree. |
| `parallelism.pipeline_parallel_degree` | `1` | Pipeline parallelism degree. |
| `parallelism.data_parallel_shard_degree` | `-1` | FSDP shard degree; `-1` lets the framework choose. |
| `parallelism.data_parallel_replicate_degree` | `1` | DDP-style replication degree. |
| `parallelism.expert_parallel_degree` | `1` | Expert parallelism for MoE. |
| `parallelism.context_parallel_degree` | `1` | Context parallelism. |
| `parallelism.fsdp_reshard_after_forward` | `default` | FSDP reshard policy (`default` uses TorchTitan’s default behavior). |
| `parallelism.enable_async_tensor_parallel` | `false` | Async tensor-parallel communication. |
| `parallelism.pipeline_parallel_schedule` | `1F1B` | Pipeline schedule (for example `1F1B`). |
| `parallelism.pipeline_parallel_microbatch_size` | `1` | Microbatch size for pipeline stages. |

### Batch parameters under `training.*`

| Key | Default | Description |
|-----|---------|-------------|
| `training.global_batch_size` | `-1` | Global batch size; `-1` typically means unset or derived. |
| `training.local_batch_size` | `8` | Per-rank local (micro) batch size. |

### Global batch relationship

For TorchTitan, a useful relationship when using replicate and shard degrees explicitly is:

\[
\text{global\_batch\_size} \approx \text{local\_batch\_size} \times \text{data\_parallel\_replicate\_degree} \times \text{data\_parallel\_shard\_degree}
\]

Exact semantics follow TorchTitan’s distributed layout; set `training.global_batch_size` and parallelism degrees consistently with your launcher’s world size.

---

## 3. MaxText parallelism configuration

MaxText (JAX) uses a **device mesh** with **ICI** (intra-node / “in-cluster interconnect”) and **DCN** (inter-node / “data center network”) axes for parallelism. Defaults and parameter names come from upstream MaxText, for example `third_party/maxtext/src/MaxText/configs/base.yml`, not from Primus presets alone.

### Common parallelism keys (from `base.yml`)

Examples include:

- `ici_tensor_parallelism` — tensor parallelism within a node
- `ici_fsdp_parallelism` — FSDP-style sharding on ICI (default `-1` for auto in many layouts)
- `dcn_data_parallelism` — data parallelism across nodes (default `-1` for auto)
- `dcn_fsdp_parallelism` — FSDP across DCN

### Batch sizing

- `per_device_batch_size` — primary knob for per-device batch (see `base.yml`).

Consult MaxText’s mesh documentation and your chosen model YAML for valid combinations of ICI/DCN axes.

---

## 4. Batch size relationships

### Megatron-style identity

\[
\text{GBS} = \text{MBS} \times \text{DP} \times \text{grad\_accum}
\]

\[
\text{DP} = \frac{\text{world\_size}}{\text{TP} \times \text{PP} \times \text{EP}}
\]

(Subject to your exact parallel groups; CP and custom layouts can introduce additional groups.)

### How GBS, MBS, and DP interact

| Goal | What to change |
|------|----------------|
| Increase global batch without more per-GPU memory | Increase `gradient_accumulation_steps` (Megatron) or increase accumulation / GBS while keeping MBS fixed. |
| Increase throughput per step | Increase `micro_batch_size` if memory allows; may require lowering accumulation to keep GBS fixed. |
| Scale to more GPUs | Increase world size; often increase DP; keep GBS stable by adjusting accumulation. |

### Memory and convergence

| Factor | Effect |
|--------|--------|
| **MBS** | Strongly affects per-GPU activation memory; larger MBS often improves GPU utilization but can OOM. |
| **GBS** | Affects effective noise in the gradient and optimal learning rate scaling; many recipes scale LR with GBS. |

**Practical recommendation:** start with `micro_batch_size` of `1` or `2`, verify stability and memory. Increase `global_batch_size` (via accumulation or more DP ranks) gradually while monitoring loss and adjusting learning rate per your recipe.

### Example numeric table (Megatron-style)

Assume TP=1, PP=1, EP=1, so DP equals world size.

| World size (DP) | MBS | Grad accum | GBS |
|-----------------|-----|------------|-----|
| 8 | 1 | 16 | 128 |
| 8 | 2 | 8 | 128 |
| 16 | 1 | 8 | 128 |
| 16 | 2 | 4 | 128 |

---

## 5. Decision guide: choosing parallelism

| Situation | Suggested direction |
|-----------|----------------------|
| Model fits on **one GPU** | Use DP and/or FSDP only; TP=1, PP=1. |
| Model fits on **one node** but not one GPU | **TP** within the node; **DP** across any remaining replicas. |
| Model needs **multiple nodes** | **TP** within node where possible; **PP** across nodes for very large depth; **DP** for remaining width. |
| **MoE** | Add **EP**; align expert count and routing with `expert_model_parallel_size` / `parallelism.expert_parallel_degree`. |
| **Very long sequences** | Increase **CP** (`context_parallel_size` / `context_parallel_degree`) as supported by the backend. |

### Example configurations (illustrative)

These are representative topologies; always validate with your checkpoint format, memory profile, and hardware interconnect.

| Profile | GPUs | TP | PP | EP | DP (illustrative) |
|---------|------|----|----|----|---------------------|
| ~7B | 8 | 1 | 1 | 1 | 8 |
| ~70B | 64 | 8 | 2 | 1 | 4 |
| Large MoE (~671B class) | many | 8 | 4 | 8 | remainder |

---

## 6. Common parallelism recipes (YAML snippets)

### Megatron: 8-GPU data parallel only

```yaml
# model (or merged language_model section)
tensor_model_parallel_size: 1
pipeline_model_parallel_size: 1
expert_model_parallel_size: 1
context_parallel_size: 1
sequence_parallel: false

# trainer
micro_batch_size: 2
global_batch_size: 128
```

### Megatron: tensor + pipeline + data parallel

```yaml
tensor_model_parallel_size: 8
pipeline_model_parallel_size: 2
expert_model_parallel_size: 1
context_parallel_size: 1
sequence_parallel: true

micro_batch_size: 1
global_batch_size: 512
```

### TorchTitan: TP + PP with explicit schedule

```yaml
parallelism:
  tensor_parallel_degree: 4
  pipeline_parallel_degree: 2
  data_parallel_replicate_degree: 1
  data_parallel_shard_degree: -1
  expert_parallel_degree: 1
  context_parallel_degree: 1
  pipeline_parallel_schedule: 1F1B
  pipeline_parallel_microbatch_size: 1
  enable_async_tensor_parallel: false

training:
  global_batch_size: 256
  local_batch_size: 4
```

For full worked examples, see `examples/megatron/configs/` and `examples/torchtitan/configs/` under your target hardware (for example `MI300X/`).
