# Llama2 custom recipe: default enabled knobs

Reference for **defaults** built by `llama2_70b_lora_config()` → `_llama2_lora()` in [`llama2_custom.py`](llama2_custom.py). YAML or Python overrides replace these per field.

For TE op fuser vs LoRA behavior and tuning, see [README_llama2_te_lora_optimization.md](README_llama2_te_lora_optimization.md).

---

## Mixed precision (`precision_config` default: `"bf16_with_fp8_hybrid"`)

Registered recipe `bf16_with_fp8_hybrid()` enables:

- BF16 compute (via `bf16_mixed()` base)
- FP8 mode: **`hybrid`**
- FP8 recipe: **`delayed`**
- **`fp8_amax_history_len`**: 4
- **`fp8_amax_compute_algo`**: `most_recent`
- **`fp8_param_gather`**: True

Model config mirrors FP8: **`fp8`**, **`fp8_recipe`**, **`fp8_amax_history_len`** set on `model_cfg`.

---

## Model (`model_cfg`)

| Knob | Default |
|------|---------|
| **BF16** (`bf16`, dtypes) | On |
| **FP16** | Off |
| **`perform_initialization`** | On |
| **`cross_entropy_loss_fusion`** | On (`cross_entropy_fusion_impl="te"`) |
| **`gradient_accumulation_fusion`** | On |
| **`fused_single_qkv_rope`** | On |
| **`apply_rope_fusion`** | On |
| **`use_transformer_engine_op_fuser`** | **On** — sets ``model_cfg.use_transformer_engine_op_fuser`` (backbone TE op fuser). |
| **`stable_lora_with_te_op_fuser`** | **On** — **Primus “safe op fuser” knob** (see below). |
| **`cpu_offloading_activations`** | On (activation offload hint for TP paths) |
| **`cuda_graph_retain_backward_graph`** | On |
| **`cuda_graph_use_single_mempool`** | On |
| **`bias_dropout_fusion`** | Off |
| **`disable_parameter_transpose_cache`** | False (cache **not** disabled) |
| **`fine_grained_activation_offloading`** | Off |
| **`use_transformer_engine_full_layer_spec`** | Off |
| **`cpu_offloading`** / **`cpu_offloading_num_layers`** | Off / 0 |
| **`empty_unused_memory_level`** | 0 |

### `stable_lora_with_te_op_fuser` (default: **on**)

One flag that selects **Primus stable TE op fuser + LoRA** vs **legacy Megatron-Bridge coupling**.

| `stable_lora_with_te_op_fuser` | `use_transformer_engine_op_fuser` | Effect |
|--------------------------------|-------------------------------------|--------|
| **true** (default) | true | Backbone op fuser **on**; LoRA **`use_te_fused_lora=False`** (always `LoRALinear`) — stable LM/val curves with fused MLP path. |
| **true** | false | Backbone op fuser **off**; LoRA unfused. |
| **false** | true | **Legacy:** backbone op fuser **on**; LoRA **`use_te_fused_lora=True`** when TP world size is 1 (`TEFusedLoRALinear` tracks backbone — can change loss). |
| **false** | false | Both off. |

YAML example (recommended):

```yaml
use_transformer_engine_op_fuser: true
stable_lora_with_te_op_fuser: true
```

Legacy (old Bridge behavior with op fuser):

```yaml
use_transformer_engine_op_fuser: true
stable_lora_with_te_op_fuser: false
```

Optional **`te_fused_lora_include_modules`** / **`te_fused_lora_exclude_modules`** only matter when fused LoRA is actually enabled (`stable_lora_with_te_op_fuser: false` and backbone op fuser on). See [README_llama2_te_lora_optimization.md](README_llama2_te_lora_optimization.md).

---

## Optimizer

| Knob | Default |
|------|---------|
| **`use_distributed_optimizer`** | On |
| **`overlap_param_gather_with_optimizer_step`** | On |
| **`params_dtype`** | `bfloat16` |

Scheduler: distributed fused Adam + cosine annealing (LR, warmup, clip from recipe args).

---

## LoRA (`peft`)

| Knob | Default |
|------|---------|
| **Adapter targets** | `linear_qkv`, `linear_proj` |
| **`a2a_experimental`** | **On** |
| **`use_te_fused_lora`** | **Not a direct recipe arg** — derived: off when **`stable_lora_with_te_op_fuser`** is true; else follows backbone op fuser (legacy). |

Fixed LoRA hyperparameters in recipe: **dim=16**, **alpha=32**, **dropout=0.1**, **dropout_position=pre**, **lora_A=xavier**, **lora_B=zero**.

---

## Dataset / dataloader (default `dataset_type="mlperf_dataset"`)

| Knob | Default |
|------|---------|
| **`data_sharding`** | On |
| **`do_validation`** | On |
| **`do_test`** | Off |
| **`pin_memory`** | On |
| **`memmap_workers`** | 1 |
| **`num_workers`** | 0 (recipe forces after initial config) |
| **`persistent_workers`** | Off |
| **`dataloader_type`** | `batch` |
| **`packed_sequence`** | Off (no `PackedSequenceSpecs` unless enabled) |

---

## Training (`TrainingConfig`)

| Knob | Default |
|------|---------|
| **`manual_gc`** | On (`manual_gc_interval` / `manual_gc_eval`: 500) |
| **`empty_unused_memory_level`** | 0 |

---

## DDP (`DistributedDataParallelConfig`)

| Knob | Default |
|------|---------|
| **`overlap_grad_reduce`** | On |
| **`overlap_param_gather`** | On |
| **`average_in_collective`** | On |
| **`use_distributed_optimizer`** | On |
| **`gradient_reduce_div_fusion`** | On |
| **`pad_buckets_for_high_nccl_busbw`** | On |
| **`keep_fp8_transpose_cache`** | On |
| **`check_for_nan_in_grad`** | Off |
| **`grad_reduce_in_fp32`** | Off |
| **`use_megatron_fsdp`** | Off |
| **`fp8_param_gather`** | Off |

---

## Logger, profiling, checkpoint

| Area | Default |
|------|---------|
| **TensorBoard metrics** (loss scale, timers, throughput, memory, wandb, etc.) | Off / unset |
| **`profiling.use_pytorch_profiler`** | Off |
| **`profiling.record_shapes`** | On (when profiling config applies) |
| **`checkpoint.finetune`** | On |
| **`checkpoint.save_optim` / `save_rng` / default `save_interval`** | Off / None as in recipe |

---

## Rerun / loss

| Knob | Default |
|------|---------|
| **`RerunStateMachineConfig.check_for_nan_in_loss`** | **On** (`check_for_nan_in_loss=True` unless overridden) |

---

## Communication overlap

If **`comm_overlap_config`** is not passed:

- **`CommOverlapConfig(tp_comm_overlap=False)`** — TP comm overlap **off** by default.

---

## Parallelism & tokenizer (common overrides)

| Knob | Default |
|------|---------|
| **`tensor_model_parallel_size`** | 1 |
| **`pipeline_model_parallel_size`** | 1 |
| **`context_parallel_size`** | 1 |
| **`sequence_parallel`** | Off |
| **`use_null_tokenizer`** | Off (HuggingFace tokenizer from `hf_path`) |

---

## `llama2_70b_lora_config` vs `_llama2_lora`

`llama2_70b_lora_config` merges recommended kwargs; notable difference from `_llama2_lora` signature defaults:

- **`eval_iters`**: **32** in the 70B helper vs **48** in `_llama2_lora` alone.

---

## Runtime-only (training loop / env)

Not stored in `ConfigContainer` defaults from this file, but **active when configured externally**:

- **Straggler timer**: if `config.straggler` exists and **`log_straggler`** is true.
- **NVRx straggler**: if `global_state.nvrx_straggler_manager` is set.
- **RPD profiler**: if **`PROFILER=rpd`** in the environment.
- **Tensor inspect**: via `config.tensor_inspect` when enabled upstream.
