# TorchTitan Backend Configuration Reference

This page lists **all** configuration keys available to Primus when `framework: torchtitan`. Defaults are taken from the TorchTitan module preset (`pre_trainer.yaml`), its `extends` chain (`module_base.yaml`, `quantize.yaml`), and the example model preset `llama3_8B.yaml`.

**Where parameters live.** Provide overrides under `modules.pre_trainer.overrides:` in your experiment YAML. TorchTitan’s `JobConfig` is hierarchical: use **dot notation** for flat overrides, or nest YAML objects under `overrides`—both are equivalent when merged.

**Example (flat dot paths):**

```yaml
framework: torchtitan

modules:
  pre_trainer:
    overrides:
      training.steps: 20000
      training.global_batch_size: 512
      optimizer.lr: 0.00015
      parallelism.tensor_parallel_degree: 2
```

**Example (nested YAML):**

```yaml
modules:
  pre_trainer:
    overrides:
      training:
        steps: 20000
        global_batch_size: 512
      optimizer:
        lr: 0.00015
      parallelism:
        tensor_parallel_degree: 2
```

**Presets.**

- Module presets: `primus/configs/modules/torchtitan/` (main entry: `pre_trainer.yaml`).
- Model presets: `primus/configs/models/torchtitan/` (example: `llama3_8B.yaml`).

**Mapping to TorchTitan.** Keys are translated into TorchTitan’s `JobConfig` via `TorchTitanJobConfigBuilder` (same nested structure as upstream TorchForge/TorchTitan).

**Upstream reference.** TorchForge / TorchTitan documentation: [https://meta-pytorch.org/torchforge/main/](https://meta-pytorch.org/torchforge/main/)

---

## 1. Base module parameters

*Source: `primus/configs/modules/module_base.yaml` (merged before TorchTitan-specific keys; `pre_trainer.yaml` does not override `trainable`).*

| Parameter | Default | Description |
|-----------|---------|-------------|
| `trainable` | `false` | Whether this module is active in training orchestration. (TorchTitan preset inherits `false` from `module_base.yaml`.) |
| `sink_level` | `null` | Structured logging sink level; `null` uses defaults. |
| `file_sink_level` | `DEBUG` | Minimum level for file logging. |
| `stderr_sink_level` | `INFO` | Minimum level for stderr logging. |

---

## 2. Training (`training.*`)

*Source: `primus/configs/modules/torchtitan/pre_trainer.yaml`.*

| Parameter | Default | Description |
|-----------|---------|-------------|
| `training.mock_data` | `true` | Use synthetic data instead of reading `dataset_path`. |
| `training.debug_moe_force_load_balance` | `false` | Debug helper to force MoE load balancing behavior. |
| `training.dataset` | `c4` | Dataset name key for TorchTitan dataset loaders. |
| `training.dataset_path` | `null` | Filesystem or remote path to dataset assets. |
| `training.deterministic` | `false` | Prefer deterministic algorithms (often slower). |
| `training.enable_cpu_offload` | `false` | Offload optimizer or activations to CPU when supported. |
| `training.gc_debug` | `false` | Extra garbage-collection diagnostics. |
| `training.gc_freq` | `50` | Run Python GC every N steps when enabled. |
| `training.global_batch_size` | `-1` | Global batch size across all ranks (`-1` often means “auto” / unset in TorchTitan). |
| `training.local_batch_size` | `8` | Per-rank microbatch size before gradient accumulation. |
| `training.max_norm` | `1.0` | Gradient clipping max norm (global). |
| `training.mixed_precision_param` | `bfloat16` | Parameter dtype for mixed precision (`bfloat16`, `float16`, etc.). |
| `training.mixed_precision_reduce` | `float32` | Dtype for reduction / gradient accumulation. |
| `training.seed` | `null` | RNG seed; `null` lets the framework choose. |
| `training.seq_len` | `2048` | Sequence length per sample. |
| `training.steps` | `10000` | Total optimizer steps. |

---

## 3. Optimizer (`optimizer.*`)

*Source: `pre_trainer.yaml`.*

| Parameter | Default | Description |
|-----------|---------|-------------|
| `optimizer.name` | `AdamW` | Optimizer class (`AdamW`, `Adam`, …). |
| `optimizer.lr` | `0.0008` | Base learning rate. |
| `optimizer.beta1` | `0.9` | First moment decay. |
| `optimizer.beta2` | `0.95` | Second moment decay. |
| `optimizer.eps` | `1.0e-08` | Numerical stability term. |
| `optimizer.weight_decay` | `0.1` | Weight decay coefficient. |
| `optimizer.implementation` | `fused` | Kernel implementation (`fused`, `foreach`, …). |
| `optimizer.early_step_in_backward` | `false` | Experimental: step optimizer during backward when supported. |

---

## 4. Learning rate scheduler (`lr_scheduler.*`)

*Source: `pre_trainer.yaml`.*

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr_scheduler.decay_ratio` | `null` | Fraction of training at the end used for decay; `null` uses framework default. |
| `lr_scheduler.decay_type` | `linear` | LR decay curve (`linear`, `cosine`, etc.). |
| `lr_scheduler.min_lr_factor` | `0.0` | LR floor as a fraction of base LR after decay. |
| `lr_scheduler.warmup_steps` | `200` | Linear warmup steps before decay. |

---

## 5. Parallelism (`parallelism.*`)

*Source: `pre_trainer.yaml`.*

| Parameter | Default | Description |
|-----------|---------|-------------|
| `parallelism.tensor_parallel_degree` | `1` | Tensor parallelism (intra-layer) degree. |
| `parallelism.pipeline_parallel_degree` | `1` | Pipeline parallelism stages. |
| `parallelism.pipeline_parallel_microbatch_size` | `1` | Microbatches per pipeline round. |
| `parallelism.pipeline_parallel_schedule` | `1F1B` | Pipeline schedule name (`1F1B`, `GPipe`, …). |
| `parallelism.pipeline_parallel_schedule_csv` | `''` | Optional CSV schedule definition. |
| `parallelism.pipeline_parallel_split_points` | `[]` | Layer indices for manual PP splits. |
| `parallelism.pipeline_parallel_layers_per_stage` | `null` | Layers per stage when auto-balanced. |
| `parallelism.pipeline_parallel_first_stage_less_layers` | `1` | Fewer layers on first PP stage (for imbalance). |
| `parallelism.pipeline_parallel_last_stage_less_layers` | `1` | Fewer layers on last PP stage. |
| `parallelism.data_parallel_shard_degree` | `-1` | FSDP / shard degree (`-1` = auto). |
| `parallelism.data_parallel_replicate_degree` | `1` | Replicated data-parallel groups. |
| `parallelism.expert_parallel_degree` | `1` | Expert parallelism for MoE models. |
| `parallelism.expert_tensor_parallel_degree` | `1` | Tensor parallelism inside experts. |
| `parallelism.context_parallel_degree` | `1` | Context (sequence) parallelism degree. |
| `parallelism.context_parallel_rotate_method` | `allgather` | Communication pattern for context parallel. |
| `parallelism.disable_loss_parallel` | `false` | Disable loss parallel layout when TP is used. |
| `parallelism.enable_async_tensor_parallel` | `false` | Overlap TP collectives with compute. |
| `parallelism.enable_compiled_autograd` | `false` | Use `torch.compile` on autograd regions. |
| `parallelism.fsdp_reshard_after_forward` | `default` | FSDP reshard policy (`default`, `always`, `never`). |
| `parallelism.module_fqns_per_model_part` | `null` | Map of pipeline stage → module FQNs for multi-part models. |

---

## 6. Checkpoint (`checkpoint.*`)

*Source: `pre_trainer.yaml`.*

| Parameter | Default | Description |
|-----------|---------|-------------|
| `checkpoint.enable` | `false` | Master switch for checkpointing. |
| `checkpoint.folder` | `checkpoint` | Output directory for checkpoints. |
| `checkpoint.interval` | `500` | Save every N steps. |
| `checkpoint.initial_load_path` | `null` | Path to load from at startup. |
| `checkpoint.initial_load_model_only` | `true` | Load weights only (skip optimizer/scheduler). |
| `checkpoint.initial_load_in_hf` | `false` | Load initial weights from Hugging Face format. |
| `checkpoint.last_save_model_only` | `true` | Final save stores weights only. |
| `checkpoint.last_save_in_hf` | `false` | Export final weights in Hugging Face format. |
| `checkpoint.export_dtype` | `float32` | Dtype for exported checkpoints. |
| `checkpoint.async_mode` | `disabled` | Async checkpoint (`disabled`, `async`, …). |
| `checkpoint.keep_latest_k` | `10` | Retain only the newest k checkpoints. |
| `checkpoint.load_step` | `-1` | Step index to load (`-1` = latest). |
| `checkpoint.exclude_from_loading` | `[]` | FQNs or keys to skip when loading. |
| `checkpoint.enable_first_step_checkpoint` | `false` | Save checkpoint at step 0 for debugging. |
| `checkpoint.create_seed_checkpoint` | `false` | Save a seed checkpoint before training starts. |

---

## 7. Activation checkpoint (`activation_checkpoint.*`)

*Source: `pre_trainer.yaml`.*

| Parameter | Default | Description |
|-----------|---------|-------------|
| `activation_checkpoint.mode` | `none` | Activation checkpointing mode (`none`, `selective`, `full`). |
| `activation_checkpoint.selective_ac_option` | `"2"` | Selective AC policy string (TorchTitan-specific). |
| `activation_checkpoint.per_op_sac_force_recompute_mm_shapes_by_fqns` | `["moe.router.gate"]` | FQNs that always recompute matmuls in selective AC. |
| `activation_checkpoint.early_stop` | `false` | Stop AC early in certain subgraphs. |

---

## 8. Metrics and profiling

### 8.1 Metrics (`metrics.*`)

*Source: `pre_trainer.yaml`.*

| Parameter | Default | Description |
|-----------|---------|-------------|
| `metrics.disable_color_printing` | `false` | Disable ANSI colors in logs. |
| `metrics.enable_tensorboard` | `false` | Write TensorBoard scalars. |
| `metrics.enable_wandb` | `false` | Log to Weights & Biases. |
| `metrics.log_freq` | `10` | Steps between metric logs. |
| `metrics.save_for_all_ranks` | `false` | Save metric files per rank (not just rank 0). |
| `metrics.save_tb_folder` | `tb` | TensorBoard subdirectory / name. |

### 8.2 Profiling (`profiling.*`)

*Source: `pre_trainer.yaml`.*

| Parameter | Default | Description |
|-----------|---------|-------------|
| `profiling.enable_profiling` | `false` | Enable PyTorch profiler traces. |
| `profiling.enable_memory_snapshot` | `false` | Capture CUDA memory snapshots. |
| `profiling.profile_freq` | `10` | Steps between profiler activations. |
| `profiling.save_traces_folder` | `profile_traces` | Directory for profiler traces. |
| `profiling.save_memory_snapshot_folder` | `memory_snapshot` | Directory for memory snapshots. |

---

## 9. Quantization (`quantize.*`)

*Source: `primus/configs/modules/torchtitan/quantize.yaml` (merged into the module preset).*

### 9.1 Linear FP8 (`quantize.linear.float8.*`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `quantize.linear.float8.enable_fsdp_float8_all_gather` | `false` | FP8 all-gather for FSDP sharded params (recommended for tensorwise scaling). |
| `quantize.linear.float8.precompute_float8_dynamic_scale_for_fsdp` | `false` | Precompute dynamic scales for FSDP FP8. |
| `quantize.linear.float8.recipe_name` | `null` | Recipe (`tensorwise`, `rowwise`, `rowwise_with_gw_hp`); `null` disables. |
| `quantize.linear.float8.filter_fqns` | `[]` | Module FQNs to skip for FP8 training. |
| `quantize.linear.float8.emulate` | `false` | Emulate FP8 in FP32 (no FP8 HW); not compatible with `torch.compile`. |

### 9.2 Linear MX (`quantize.linear.mx.*`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `quantize.linear.mx.mxfp8_dim1_cast_kernel_choice` | `"triton"` | Kernel backend for MXFP8 dim-1 cast (`triton`, `cuda`, `torch`). |
| `quantize.linear.mx.recipe_name` | `"mxfp8_cublas"` | MX recipe name (see torchao `mx_formats`). |
| `quantize.linear.mx.filter_fqns` | `["output"]` | FQNs to skip; output layer skipped by default. |

### 9.3 Grouped GEMM FP8 (`quantize.grouped_mm.float8.*`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `quantize.grouped_mm.float8.fqns` | `[]` | MoE layer FQNs for FP8 grouped GEMM (prototype; may require torchao nightly). |

### 9.4 Grouped GEMM MX (`quantize.grouped_mm.mx.*`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `quantize.grouped_mm.mx.recipe_name` | `"mxfp8"` | MX recipe for grouped GEMMs. |
| `quantize.grouped_mm.mx.fqns` | `[]` | MoE module FQNs for MXFP8 grouped GEMM (prototype). |

---

## 10. Compile (`compile.*`)

*Source: `pre_trainer.yaml`.*

| Parameter | Default | Description |
|-----------|---------|-------------|
| `compile.enable` | `true` | Enable `torch.compile` on selected subsystems. |
| `compile.components` | `["model", "loss"]` | Which components to compile. |

---

## 11. Communication and fault tolerance

### 11.1 Communicator (`comm.*`)

*Source: `pre_trainer.yaml`.*

| Parameter | Default | Description |
|-----------|---------|-------------|
| `comm.init_timeout_seconds` | `300` | Timeout for initial process-group setup. |
| `comm.train_timeout_seconds` | `100` | Timeout for training collectives. |
| `comm.trace_buf_size` | `20000` | Flight recorder buffer size for NCCL traces. |
| `comm.save_traces_folder` | `comm_traces` | Where to dump communication traces. |

### 11.2 Fault tolerance (`fault_tolerance.*`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fault_tolerance.enable` | `false` | Enable fault-tolerant training hooks. |
| `fault_tolerance.process_group` | `gloo` | Backend for control-plane process group. |
| `fault_tolerance.process_group_timeout_ms` | `10000` | Control-plane timeout. |
| `fault_tolerance.replica_id` | `0` | Replica index in elastic setups. |
| `fault_tolerance.group_size` | `0` | Group size (0 = unset / default). |
| `fault_tolerance.min_replica_size` | `1` | Minimum replicas to continue. |
| `fault_tolerance.semi_sync_method` | `null` | Optional semi-synchronous strategy name. |

### 11.3 Memory estimation (`memory_estimation.*`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `memory_estimation.enable` | `false` | Run memory-estimation fake-mode passes. |
| `memory_estimation.disable_fake_mode` | `false` | Disable fake tensor mode inside estimation. |

### 11.4 Experimental (`experimental.*`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `experimental.custom_import` | `""` | Optional Python module import path for custom extensions. |
| `experimental.custom_args_module` | `"primus.backends.torchtitan.primus_turbo_extensions.config_extension"` | Module providing extra `JobConfig` fields for Primus-Turbo. |

---

## 12. Primus-Turbo (`primus_turbo.*`)

*Source: `pre_trainer.yaml`.*

| Parameter | Default | Description |
|-----------|---------|-------------|
| `primus_turbo.enable_primus_turbo` | `true` | Master switch for Primus-Turbo integrations in TorchTitan. |
| `primus_turbo.enable_attention_float8` | `false` | FP8 attention path inside Turbo attention. |
| `primus_turbo.use_turbo_attention` | `true` | Use Turbo attention kernels. |
| `primus_turbo.use_classic_attention` | `false` | Fall back to classic attention implementation. |
| `primus_turbo.use_turbo_async_tp` | `true` | Async tensor-parallel communication in Turbo. |
| `primus_turbo.use_turbo_mx_linear` | `true` | MX linear layers via Turbo. |
| `primus_turbo.use_turbo_float8_linear` | `true` | FP8 linear layers via Turbo. |
| `primus_turbo.use_turbo_grouped_mm` | `false` | Turbo grouped GEMM for MoE (off by default). |
| `primus_turbo.use_moe_fp8` | `true` | FP8 paths for MoE experts when applicable. |
| `primus_turbo.enable_embedding_autocast` | `true` | Autocast policy around embeddings for Turbo. |

---

## 13. Model (`model.*` and `job.*`)

### 13.1 Model preset (`models.*` / `model.*`)

*Example defaults from `primus/configs/models/torchtitan/llama3_8B.yaml`.*

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.name` | `"llama3"` | Model family key for TorchTitan recipes. |
| `model.flavor` | `"8B"` | Size / variant within the family. |
| `model.hf_assets_path` | `"meta-llama/Meta-Llama-3-8B"` | Hugging Face Hub repo or local path for weights/tokenizer. |
| `model.converters` | `["primus_turbo"]` | Weight converter pipeline stages applied at load. |

### 13.2 Job metadata (`job.*`)

*Source: `llama3_8B.yaml`.*

| Parameter | Default | Description |
|-----------|---------|-------------|
| `job.dump_folder` | `"./outputs"` | Root directory for logs, checkpoints, and exports. |
| `job.description` | `"Llama 3 8B training"` | Human-readable label for run metadata. |

---

## 14. Validation (`validation.*`)

*Source: `pre_trainer.yaml`.*

| Parameter | Default | Description |
|-----------|---------|-------------|
| `validation.enable` | `false` | Run periodic validation loops. |
| `validation.dataset` | `c4_validation` | Validation dataset key. |
| `validation.dataset_path` | `null` | Filesystem path to validation data. |
| `validation.local_batch_size` | `8` | Per-rank validation batch size. |
| `validation.seq_len` | `2048` | Validation sequence length. |
| `validation.freq` | `10` | Run validation every N training steps. |
| `validation.steps` | `-1` | Max validation steps (`-1` = full pass / framework default). |

---

### See also

- TorchForge documentation: [https://meta-pytorch.org/torchforge/main/](https://meta-pytorch.org/torchforge/main/)
- Primus TorchTitan presets: `primus/configs/modules/torchtitan/`
- Primus TorchTitan model presets: `primus/configs/models/torchtitan/`
