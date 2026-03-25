# Megatron Bridge Backend Configuration Reference

Megatron Bridge integrates [Megatron-Core](https://github.com/NVIDIA/Megatron-LM) training with Hugging Face–centric workflows. In Primus, post-training (supervised fine-tuning, LoRA, and similar) uses the **`megatron_bridge`** framework with module preset `sft_trainer.yaml` and model YAMLs under `primus/configs/models/megatron_bridge/`.

## Recipe system

Megatron Bridge resolves training defaults through a **recipe** and **flavor**:

- `recipe` is a Python module path under `megatron.bridge.recipes` (e.g. `qwen.qwen3`).
- `flavor` is the function name inside that module (e.g. `qwen3_8b_finetune_config`) that returns a `ConfigContainer`.

At runtime, `load_recipe_config` in `primus/backends/megatron_bridge/config_utils.py`:

1. Imports `megatron.bridge.recipes.<recipe>` and calls `<flavor>(**filtered_backend_args)` to build the baseline `ConfigContainer`.
2. **Deep-merges** Primus `backend_args` (from YAML + CLI) into that dataclass via `_merge_dict_to_dataclass`, so user overrides sit on top of recipe defaults.

You normally specify `recipe`, `flavor`, `hf_path`, and `dataset` in the model YAML; training hyperparameters and parallelism go in module overrides or experiment `modules.post_trainer.overrides`.

---

## 1. Base module parameters

From `primus/configs/modules/megatron_bridge/sft_trainer.yaml` (extends `module_base.yaml`).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `trainable` | `true` | Module participates in the training graph. |
| `sink_level` | `null` | Inherited from `module_base.yaml`; structured logging sink level. |
| `file_sink_level` | `DEBUG` | File sink verbosity. |
| `stderr_sink_level` | `INFO` | Stderr sink verbosity. |

---

## 2. Training

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stage` | `"sft"` | Backend stage selector. Primus dispatches post-training via `primus train posttrain` and loads the Megatron Bridge posttrain trainer when this module is used under `post_trainer`. |
| `trainable` | `true` | See Base module parameters. |

**CLI note:** The user-facing suite is **`posttrain`** (`primus train posttrain --config ...`). The YAML `stage` field selects the Megatron Bridge trainer implementation (`sft`), not the CLI suite name.

---

## 3. Fine-tuning method (PEFT)

Primus examples set these under `modules.post_trainer.overrides` (see `examples/megatron_bridge/configs/`).

| Parameter | Example | Description |
|-----------|---------|-------------|
| `peft` | `"none"`, `"lora"` | Parameter-efficient fine-tuning mode. |
| `peft_dim` | `16` | LoRA rank (example: `llama31_70b_lora_posttrain.yaml`). |
| `peft_alpha` | `32` | LoRA scaling alpha (same example). |
| `packed_sequence` | `false` | Pack multiple short sequences per microbatch when supported. |

Additional keys such as `pretrained_checkpoint`, `use_distributed_optimizer`, or `cross_entropy_loss_fusion` appear in larger examples and are merged into the recipe `ConfigContainer` when the dataclass exposes matching fields.

---

## 4. Parallelism

Typical overrides from Megatron Bridge examples:

| Parameter | Example | Description |
|-----------|---------|-------------|
| `tensor_model_parallel_size` | `1`, `2`, `8` | Tensor parallelism degree. |
| `pipeline_model_parallel_size` | `1` | Pipeline parallelism degree. |
| `virtual_pipeline_model_parallel_size` | `null` | Virtual pipeline stages per rank when PP > 1. |
| `context_parallel_size` | `1` | Context parallelism degree. |
| `sequence_parallel` | `false` | Sequence parallelism within TP groups. |
| `use_megatron_fsdp` | `false` | Optional Megatron FSDP path. |

---

## 5. Training hyperparameters

| Parameter | Example | Description |
|-----------|---------|-------------|
| `train_iters` | `200`, `1000` | Total training iterations. |
| `global_batch_size` | `8`, `128` | Global batch across data-parallel groups. |
| `micro_batch_size` | `1`, `2` | Per-GPU microbatch before gradient accumulation. |
| `seq_length` | `2048`, `8192` | Training sequence length. |
| `eval_interval` | `30` | Steps between evaluations. |
| `save_interval` | `50` | Steps between checkpoint saves. |

---

## 6. Learning rate

| Parameter | Example | Description |
|-----------|---------|-------------|
| `finetune_lr` | `1.0e-4`, `5.0e-6` | Peak learning rate for fine-tuning. |
| `min_lr` | `0.0` | Floor learning rate after decay. |
| `lr_warmup_iters` | `50` | Linear warmup length in iterations. |
| `lr_decay_iters` | `null` | Optional decay span; `null` defers to recipe defaults. |

---

## 7. Precision

| Parameter | Example | Description |
|-----------|---------|-------------|
| `precision_config` | `bf16_mixed`, `fp16_mixed`, `fp32` | Mixed-precision recipe for Megatron Bridge. |
| `comm_overlap_config` | `null` | Optional communication/compute overlap policy object. |
| `pipeline_dtype` | `null` | Dtype for pipeline stages when PP is enabled. |

---

## 8. Memory optimization

| Parameter | Example | Description |
|-----------|---------|-------------|
| `recompute_granularity` | `full` | Activation recomputation granularity. |
| `recompute_method` | `uniform` | How recomputation is scheduled across layers. |
| `recompute_num_layers` | `1` | Number of layers per recompute group (workload-dependent). |

---

## 9. Primus-Turbo

From `sft_trainer.yaml` (defaults shown).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_primus_turbo` | `true` | Master flag for Primus-Turbo optimized kernels and paths. |
| `use_turbo_attention` | `false` | Turbo attention implementation. |
| `use_turbo_parallel_linear` | `false` | Turbo parallel linear layers. |
| `use_turbo_grouped_mlp` | `false` | Turbo grouped MLP. |
| `moe_use_fused_router_with_aux_score` | `false` | Fused MoE router with auxiliary loss handling. |
| `enable_turbo_attention_float8` | `false` | FP8 path inside Turbo attention. |
| `use_turbo_deepep` | `false` | DeepEP-style expert-parallel integration. |
| `turbo_deepep_num_cu` | `32` | Compute-unit count hint for DeepEP. |
| `turbo_deepep_use_comm_stream` | `false` | Use dedicated communication streams. |
| `turbo_sync_free_moe_stage` | `0` | Sync-free MoE scheduling stage. |
| `use_turbo_fused_act_with_probs` | `false` | Fuse activation with probability tensors where applicable. |
| `use_turbo_rms_norm` | `false` | Turbo RMSNorm path. |

**Environment:** `PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND` (default `TURBO`) is read in `primus/modules/trainer/megatron/utils.py` to select MoE dispatch/combine behavior when Turbo MoE is active.

---

## 10. Model and dataset

Model YAML files (`qwen3_8b.yaml`, `qwen3_32b.yaml`, `llama31_70b.yaml`) supply:

| Parameter | Example | Description |
|-----------|---------|-------------|
| `recipe` | `qwen.qwen3`, `llama.llama3` | Recipe module under `megatron.bridge.recipes`. |
| `flavor` | `qwen3_8b_finetune_config`, `llama31_70b_finetune_config` | Flavor function producing the baseline `ConfigContainer`. |
| `hf_path` | `Qwen/Qwen3-8B`, `meta-llama/Meta-Llama-3.1-70B` | Hugging Face model id for weights/tokenizer flows. |
| `dataset` | nested | Example: `dataset_name: "rajpurkar/squad"` for SQuAD-style fine-tuning. |

**Logging (optional overrides in examples):** `wandb_project`, `wandb_entity`, `wandb_exp_name` may be set under `overrides` for experiment tracking when Weights & Biases is configured.

---

## Argument merge mechanics

`MegatronBridgeArgBuilder` (`primus/backends/megatron_bridge/argument_builder.py`) performs a **deep merge** of CLI and YAML into a single dict/namespace before `load_recipe_config` runs. Nested dicts (for example dataset or optimizer sections) combine recursively; explicit `None` in the merged structure can clear fields depending on merge rules in `_merge_dict_to_dataclass`.

---

## Example layouts

Under `examples/megatron_bridge/configs/`, per-GPU directories (for example `MI300X/`, `MI355X/`) contain full experiment YAMLs that set `work_group`, `user_name`, `exp_name`, `workspace`, and `modules.post_trainer` with `framework: megatron_bridge`, `config: sft_trainer.yaml`, `model: <preset>.yaml`, and a rich `overrides` block for parallelism, PEFT, LR, and precision.
