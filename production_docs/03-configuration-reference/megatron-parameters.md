# Megatron Backend Configuration Reference

This page lists **every** flat configuration key exposed by Primus when `framework: megatron`. Values are the defaults from the Megatron module and model presets in this repository (merged in YAML `extends` order).

**Where parameters live.** Set overrides under `modules.pre_trainer.overrides:` in your experiment YAML. Model architecture keys usually come from `models.<role>.overrides:` (or your chosen model preset), but the same names map to Megatron’s argparse namespace either way.

**Presets.**

- Module presets: `primus/configs/modules/megatron/` (the main training bundle is `trainer_base.yaml`, which extends Primus Megatron add-ons).
- Model presets: `primus/configs/models/megatron/` (for example `language_model.yaml`).

**Mapping to Megatron-LM.** Keys are passed through **1:1** to Megatron’s training arguments (same names as `argparse` / `Namespace`). Primus builds that namespace with `MegatronArgBuilder`.

**Upstream reference.** Full flag semantics and newer options are defined in Megatron-LM: [`megatron/training/arguments.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/arguments.py).

### Example (experiment YAML)

```yaml
framework: megatron

modules:
  pre_trainer:
    overrides:
      global_batch_size: 256
      train_iters: 50000
      tensor_model_parallel_size: 2

models:
  pre_train:
    overrides:
      hidden_size: 2048
      num_layers: 32
```

---

## 1. Base module parameters

*Source: `primus/configs/modules/module_base.yaml` (merged into Megatron presets; `trainer_base.yaml` sets `trainable: true`).*

| Parameter | Default | Description |
|-----------|---------|-------------|
| `trainable` | `true` | When `true`, this module participates in training workflows. (`module_base.yaml` alone defaults to `false`; Megatron `trainer_base.yaml` overrides to `true`.) |
| `sink_level` | `null` | Log level for the structured sink (Primus module plumbing); `null` uses framework default. |
| `file_sink_level` | `DEBUG` | Minimum level for file-backed logging. |
| `stderr_sink_level` | `INFO` | Minimum level for stderr logging. |

---

## 2. Training and batching

*Source: `primus/configs/modules/megatron/trainer_base.yaml`.*

| Parameter | Default | Description |
|-----------|---------|-------------|
| `yaml_cfg` | `null` | Reserved; not supported as a Megatron override in this preset. |
| `spec` | `null` | Optional trainer spec hook (unused in defaults). |
| `micro_batch_size` | `2` | Samples per microbatch per data-parallel rank (per forward/backward step before gradient accumulation). |
| `batch_size` | `null` | Deprecated; use `micro_batch_size` / `global_batch_size`. |
| `global_batch_size` | `128` | Total batch size across the data-parallel world (before or after splitting, per Megatron semantics). |
| `rampup_batch_size` | `null` | Optional batch-size ramp schedule string / config. |
| `decrease_batch_size_if_needed` | `false` | Allow shrinking batch if memory is insufficient. |
| `check_for_nan_in_loss_and_grad` | `true` | Abort on NaNs in loss or gradients. |
| `check_for_spiky_loss` | `false` | Detect abnormal loss spikes. |
| `check_for_large_grads` | `false` | Detect abnormally large gradients. |
| `make_vocab_size_divisible_by` | `128` | Pads vocabulary size for efficient kernels / partitioning. |
| `exit_signal_handler` | `false` | Install handlers for graceful shutdown signals. |
| `exit_duration_in_mins` | `null` | Stop training after this many minutes. |
| `exit_interval` | `null` | Exit after this many iterations (if set). |
| `onnx_safe` | `null` | ONNX export compatibility tweaks. |
| `bert_binary_head` | `true` | Use BERT binary classification head when applicable. |
| `use_flash_attn` | `false` | Prefer FlashAttention kernels when available. |
| `seed` | `1234` | RNG seed for reproducibility. |
| `data_parallel_random_init` | `false` | Random init that varies across data-parallel ranks. |
| `init_method_xavier_uniform` | `false` | Use Xavier uniform for some weights. |
| `test_mode` | `false` | Lightweight test path (fewer steps / checks). |
| `train_iters` | `null` | Total training iterations (mutually exclusive with sample-based stopping in typical setups). |
| `train_samples` | `null` | Total training samples (when using sample-based training). |
| `eval_iters` | `32` | Validation iterations per eval. |
| `eval_interval` | `2000` | Run validation every this many iterations. |
| `full_validation` | `false` | Run a full pass over validation data. |
| `multiple_validation_sets` | `false` | Multiple validation datasets / passes. |
| `skip_train` | `false` | Only run eval / test, no training updates. |
| `train_sync_interval` | `null` | Periodic distributed sync barrier for debugging. |
| `adlr_autoresume` | `false` | ADLR autoresume integration. |
| `adlr_autoresume_interval` | `1000` | Autoresume checkpoint interval. |
| `manual_gc` | `false` | Force Python GC on a schedule. |
| `manual_gc_interval` | `1` | GC every N steps when `manual_gc` is enabled. |
| `manual_gc_eval` | `false` | Run manual GC during evaluation. |
| `mask_type` | `random` | Masking strategy for MLM / similar objectives. |
| `mask_factor` | `1.0` | Masking strength multiplier. |
| `iter_per_epoch` | `1250` | Iterations interpreted as one “epoch” for logging. |

---

## 3. Mixed precision

*Source: `trainer_base.yaml`.*

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fp16` | `false` | Enable FP16 mixed precision training. |
| `bf16` | `true` | Enable BF16 mixed precision training. |
| `grad_reduce_in_bf16` | `false` | All-reduce gradients in BF16 (saves bandwidth). |
| `calculate_per_token_loss` | `false` | Normalize loss per token instead of per sample. |
| `loss_scale` | `null` | Static loss scale for FP16; `null` uses dynamic scaling. |
| `initial_loss_scale` | `4294967296` | Initial dynamic loss scale. |
| `min_loss_scale` | `1.0` | Floor for dynamic loss scale. |
| `loss_scale_window` | `1000` | Window for dynamic loss scaling updates. |
| `hysteresis` | `2` | Hysteresis steps for loss-scale decreases. |
| `accumulate_allreduce_grads_in_fp32` | `false` | Accumulate and reduce gradients in FP32. |
| `fp16_lm_cross_entropy` | `false` | Compute LM cross-entropy in FP16. |
| `fp8` | `null` | FP8 recipe selection (`e4m3`, `hybrid`, etc.); `null` disables. |
| `fp8_margin` | `0` | FP8 scaling margin. |
| `fp8_recipe` | `delayed` | FP8 recipe variant (e.g. delayed scaling). |
| `fp8_interval` | `1` | Deprecated FP8 interval (kept for compatibility). |
| `fp8_amax_history_len` | `1024` | History length for FP8 amax statistics. |
| `fp8_amax_compute_algo` | `"max"` | How to combine amax history (`max`, etc.). |
| `fp8_wgrad` | `true` | Run weight gradients in FP8 where supported. |
| `fp8_param_gather` | `false` | FP8 parameter gather for distributed optimizer paths. |
| `te_rng_tracker` | `false` | Transformer Engine RNG tracker for FP8. |
| `inference_rng_tracker` | `false` | Separate RNG tracker for inference FP8. |
| `fp4` | `null` | FP4 mode; `null` disables. |
| `fp4_recipe` | `nvfp4` | FP4 recipe name. |
| `fp4_param` | `false` | Store parameters in FP4. |
| `first_last_layers_bf16` | `false` | Keep first/last layers in BF16 for stability. |
| `num_layers_at_start_in_bf16` | `1` | Count of early layers forced to BF16 when enabled. |
| `num_layers_at_end_in_bf16` | `1` | Count of final layers forced to BF16 when enabled. |
| `no_fp8_weight_transpose_cache` | `false` | *Primus:* disable FP8 weight transpose cache (see `primus_megatron_module.yaml`). |

---

## 4. Optimizer and learning rate

*Source: `trainer_base.yaml`.*

| Parameter | Default | Description |
|-----------|---------|-------------|
| `optimizer` | `adam` | Optimizer family (`adam`, `sgd`, etc.). |
| `lr` | `2.5e-4` | Peak learning rate. |
| `lr_decay_style` | `cosine` | LR decay schedule (`cosine`, `linear`, `constant`, WSD, etc.). |
| `lr_decay_iters` | `null` | Decay duration in iterations. |
| `lr_decay_samples` | `null` | Decay duration in samples. |
| `lr_warmup_fraction` | `null` | Warmup as a fraction of total train steps. |
| `lr_warmup_iters` | `0` | Linear warmup steps. |
| `lr_warmup_samples` | `0` | Warmup in samples. |
| `lr_warmup_init` | `0.0` | LR at the start of warmup. |
| `min_lr` | `2.5e-5` | Minimum LR after decay. |
| `lr_wsd_decay_style` | `exponential` | Weight-decay schedule style for WSD when used. |
| `lr_wsd_decay_samples` | `null` | WSD decay window in samples. |
| `lr_wsd_decay_iters` | `null` | WSD decay window in iterations. |
| `head_lr_mult` | `1.0` | LR multiplier for attention/head modules when supported. |
| `weight_decay` | `0.01` | AdamW / L2-style weight decay. |
| `start_weight_decay` | `null` | Starting weight decay for schedules. |
| `end_weight_decay` | `null` | Ending weight decay for schedules. |
| `weight_decay_incr_style` | `constant` | How weight decay changes between start/end. |
| `clip_grad` | `1.0` | Global gradient norm clip. |
| `adam_beta1` | `0.9` | Adam first moment decay. |
| `adam_beta2` | `0.95` | Adam second moment decay. |
| `adam_eps` | `1.0e-08` | Adam epsilon. |
| `sgd_momentum` | `0.9` | SGD momentum when `optimizer` is SGD. |
| `override_opt_param_scheduler` | `false` | Override optimizer parameter groups’ schedulers. |
| `use_checkpoint_opt_param_scheduler` | `false` | Load optimizer scheduler state strictly from checkpoint. |
| `warmup` | `null` | Alternate warmup specification (legacy / schedule hooks). |
| `decoupled_lr` | `null` | Decoupled LR for certain param groups. |
| `decoupled_min_lr` | `null` | Minimum for decoupled LR. |
| `muon_extra_scale_factor` | `1.0` | Muon optimizer scaling. |
| `muon_scale_mode` | `"spectral"` | Muon scaling mode. |
| `muon_fp32_matmul_prec` | `"medium"` | Muon matmul precision hint. |
| `muon_num_ns_steps` | `5` | Muon Newton–Schulz iterations. |
| `muon_tp_mode` | `"blockwise"` | Muon tensor-parallel mode. |
| `muon_use_nesterov` | `false` | Muon Nesterov momentum. |
| `muon_split_qkv` | `true` | Split QKV for Muon. |
| `muon_momentum` | `0.95` | Muon momentum. |
| `muon_weight_decay` | `0.01` | Muon-specific decay. |
| `muon_weight_decay_method` | `"decoupled"` | How Muon applies decay. |
| `optimizer_cpu_offload` | `false` | Offload optimizer state to CPU. |
| `optimizer_offload_fraction` | `1.0` | Fraction of optimizer state offloaded. |
| `use_torch_optimizer_for_cpu_offload` | `false` | Use PyTorch optimizer for offload path. |
| `overlap_cpu_optimizer_d2h_h2d` | `false` | Overlap CPU optimizer device transfers. |
| `pin_cpu_grads` | `true` | Pin memory for CPU gradients. |
| `pin_cpu_params` | `true` | Pin memory for CPU params in offload. |
| `use_precision_aware_optimizer` | `false` | Use precision-aware optimizer (main grads/params in lower precision). |
| `main_grads_dtype` | `fp32` | Dtype for main gradients (`fp32`, `bf16`). |
| `main_params_dtype` | `fp32` | Dtype for master params. |
| `exp_avg_dtype` | `fp32` | Optimizer first moment dtype (`fp32`, `fp16`, `fp8`). |
| `exp_avg_sq_dtype` | `fp32` | Optimizer second moment dtype. |

---

## 5. Parallelism and distribution

*Sources: `trainer_base.yaml` (distributed runtime) and `primus/configs/models/megatron/language_model.yaml` (model-parallel sizes and TP communication).*

### 5.1 Data / distributed runtime (trainer)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `overlap_p2p_comm` | `true` | Overlap pipeline P2P with compute. |
| `distributed_backend` | `nccl` | Process-group backend (`nccl`, `gloo`, …). |
| `distributed_timeout_minutes` | `10` | Collective timeout. |
| `defer_embedding_wgrad_compute` | `false` | Defer embedding weight gradients. |
| `wgrad_deferral_limit` | `0` | Max deferred embedding wgrad steps. |
| `align_grad_reduce` | `true` | Align gradient reductions for efficiency. |
| `ddp_num_buckets` | `null` | Number of DDP buckets. |
| `ddp_bucket_size` | `null` | DDP bucket size in elements. |
| `ddp_pad_buckets_for_high_nccl_busbw` | `false` | Pad buckets for NCCL bus bandwidth. |
| `ddp_average_in_collective` | `false` | Average inside collective vs outside. |
| `overlap_grad_reduce` | `false` | Overlap gradient all-reduce with backward. |
| `overlap_param_gather` | `false` | Overlap param all-gather (distributed optimizer). |
| `overlap_param_gather_with_optimizer_step` | `false` | Overlap param gather with optimizer step. |
| `align_param_gather` | `true` | Align param gather for distributed optimizer. |
| `scatter_gather_tensors_in_pipeline` | `true` | Scatter/gather tensors across PP ranks. |
| `use_ring_exchange_p2p` | `false` | Ring-exchange P2P for PP. |
| `local_rank` | `null` | Local rank override (normally from launcher). |
| `lazy_mpu_init` | `null` | Defer Megatron parallel state init. |
| `account_for_embedding_in_pipeline_split` | `false` | Account for embedding in PP partition. |
| `account_for_loss_in_pipeline_split` | `false` | Account for loss partition in PP. |
| `empty_unused_memory_level` | `0` | Aggressiveness of `torch.cuda.empty_cache`. |
| `standalone_embedding_stage` | `false` | Dedicated PP stage for embeddings. |
| `use_distributed_optimizer` | `false` | Shard optimizer state across data parallel. |
| `use_sharp` | `false` | Use SHARP for collectives when available. |
| `sharp_enabled_group` | `null` | Which group SHARP applies to (`dp`, `dp_replica`). |
| `use_custom_fsdp` | `false` | Custom FSDP integration path. |
| `use_megatron_fsdp` | `false` | Megatron FSDP path. |
| `init_model_with_meta_device` | `false` | Build model on `meta` device first. |
| `data_parallel_sharding_strategy` | `no_shard` | FSDP / ZeRO style sharding (`no_shard`, `optim`, …). |
| `gradient_reduce_div_fusion` | `true` | Fuse division into reduce-scatter. |
| `suggested_communication_unit_size` | `400000000` | Suggested communication chunk size. |
| `keep_fp8_transpose_cache_when_using_custom_fsdp` | `false` | Keep FP8 transpose cache with custom FSDP. |
| `num_distributed_optimizer_instances` | `1` | Sharded optimizer instances per rank group. |
| `use_torch_fsdp2` | `false` | Use PyTorch FSDP2 integration. |
| `nccl_communicator_config_path` | `null` | JSON config for NCCL communicators. |
| `use_tp_pp_dp_mapping` | `false` | Custom TP/PP/DP process mapping. |
| `replication` | `false` | Data replication mode for certain schedules. |
| `replication_jump` | `null` | Stride between replicated ranks. |
| `replication_factor` | `null` | Replication factor. |
| `deterministic_mode` | `false` | Prefer deterministic algorithms (slower). |
| `check_weight_hash_across_dp_replicas_interval` | `null` | Periodically hash weights across DP replicas for debugging. |
| `overlap_moe_expert_parallel_comm` | `false` | Overlap MoE expert-parallel communication. |
| `decoder_pipeline_manual_split_list` | `null` | *Primus:* manual PP split points for decoder (list of ints). |
| `patch_moe_overlap` | `false` | *Primus:* patch MoE compute/comm overlap. |

### 5.2 Model parallelism (model preset)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_parallel_size` | `null` | Legacy combined MP size override. |
| `tensor_model_parallel_size` | `1` | Tensor parallelism degree (intra-layer split). |
| `encoder_tensor_model_parallel_size` | `0` | Encoder TP size when encoder/decoder differ. |
| `pipeline_model_parallel_size` | `1` | Pipeline parallelism stages. |
| `pipeline_model_parallel_layout` | `null` | Optional explicit PP layout string. |
| `pipeline_model_parallel_comm_backend` | `null` | `nccl` or `ucc` for PP collectives. |
| `encoder_pipeline_model_parallel_size` | `0` | Encoder PP stages (encoder–decoder models). |
| `pipeline_model_parallel_split_rank` | `null` | Rank where encoder/decoder split. |
| `decoder_first_pipeline_num_layers` | `null` | Layers on first decoder PP stage. |
| `decoder_last_pipeline_num_layers` | `null` | Layers on last decoder PP stage. |
| `virtual_pipeline_model_parallel_size` | `null` | Virtual PP (interleaved) depth. |
| `num_layers_per_virtual_pipeline_stage` | `null` | Layers per virtual stage. |
| `num_virtual_stages_per_pipeline_rank` | `null` | Virtual stages per physical PP rank. |
| `microbatch_group_size_per_vp_stage` | `null` | Microbatch grouping for interleaved PP. |
| `sequence_parallel` | `true` | Sequence parallelism when TP > 1. |
| `context_parallel_size` | `1` | Context (sequence) parallelism degree. |
| `cp_comm_type` | `p2p` | Context-parallel comm pattern (`p2p`, `a2a`, `allgather`, `a2a+p2p`). |
| `hierarchical_context_parallel_sizes` | `null` | Hierarchical CP group sizes. |
| `expert_model_parallel_size` | `1` | Expert parallelism for MoE. |
| `expert_tensor_parallel_size` | `null` | Expert tensor-parallel degree. |
| `high_priority_stream_groups` | `[]` | Named groups that get high-priority CUDA streams. |

### 5.3 Tensor-parallel communication overlap (model)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `async_tensor_model_parallel_allreduce` | `true` | Async TP all-reduces for column-parallel layers. |
| `tp_comm_overlap` | `false` | Enable TP communication overlap planner. |
| `tp_comm_overlap_cfg` | `null` | Extra JSON / path for overlap configuration. |
| `tp_comm_overlap_ag` | `true` | Overlap all-gather in TP backward. |
| `tp_comm_overlap_rs` | `true` | Overlap reduce-scatter in TP backward. |
| `tp_comm_overlap_rs_dgrad` | `false` | Overlap RS for data-grad path. |
| `tp_comm_split_ag` | `true` | Split all-gather for overlap. |
| `tp_comm_split_rs` | `true` | Split reduce-scatter for overlap. |
| `tp_comm_bulk_wgrad` | `true` | Bulk weight-gradient path for TP comm. |
| `tp_comm_bulk_dgrad` | `true` | Bulk data-gradient path for TP comm. |
| `barrier_with_L1_time` | `true` | Barrier using L1 timing hooks for TP comm profiling. |
| `tp_comm_bootstrap_backend` | `nccl` | Backend used to bootstrap TP communicators. |

---

## 6. Checkpointing

*Source: `trainer_base.yaml`.*

| Parameter | Default | Description |
|-----------|---------|-------------|
| `save` | `null` | Path prefix / pattern for checkpoints to write. |
| `save_interval` | `20000` | Save every N iterations. |
| `save_retain_interval` | `null` | Retain checkpoints at this interval. |
| `no_save_optim` | `null` | Skip optimizer state in checkpoints when truthy. |
| `no_save_rng` | `null` | Skip RNG state in checkpoints when truthy. |
| `load` | `null` | Checkpoint path to load. |
| `load_main_params_from_ckpt` | `false` | Load only main parameters. |
| `no_load_optim` | `null` | Skip loading optimizer state. |
| `no_load_rng` | `null` | Skip loading RNG state. |
| `finetune` | `false` | Finetune mode (do not require full optimizer match). |
| `use_checkpoint_args` | `false` | When `true`, restore training args from checkpoint metadata. |
| `use_mp_args_from_checkpoint_args` | `false` | Restore model-parallel args from checkpoint. |
| `use_tokenizer_model_from_checkpoint_args` | `true` | Restore tokenizer path from checkpoint args. |
| `exit_on_missing_checkpoint` | `true` | Fail if `load` is set but checkpoint is missing. |
| `non_persistent_save_interval` | `null` | Ephemeral checkpoint interval. |
| `non_persistent_ckpt_type` | `null` | `global`, `local`, `in_memory`, or `null`. |
| `non_persistent_global_ckpt_dir` | `null` | Directory for non-persistent global checkpoints. |
| `non_persistent_local_ckpt_dir` | `null` | Directory for non-persistent local checkpoints. |
| `non_persistent_local_ckpt_algo` | `"fully_parallel"` | `fully_parallel` or `atomic`. |
| `pretrained_checkpoint` | `null` | Load weights from a pretrained checkpoint path. |
| `ckpt_step` | `null` | Specific step to load within a distributed checkpoint. |
| `use_dist_ckpt_deprecated` | `false` | Use deprecated distributed checkpoint format. |
| `use_persistent_ckpt_worker` | `false` | Background worker for checkpoint IO. |
| `auto_detect_ckpt_format` | `false` | Infer checkpoint format automatically. |
| `dist_ckpt_format_deprecated` | `null` | Legacy format hint. |
| `ckpt_format` | `torch_dist` | `torch`, `torch_dist`, or `zarr`. |
| `ckpt_convert_format` | `null` | Target format for one-shot conversion. |
| `ckpt_convert_save` | `null` | Output path for conversion. |
| `ckpt_convert_update_legacy_dist_opt_format` | `false` | Update legacy distributed-optimizer layout when converting. |
| `ckpt_fully_parallel_save_deprecated` | `false` | Deprecated fully-parallel save toggle. |
| `ckpt_fully_parallel_save` | `true` | Save shards in parallel across ranks. |
| `async_save` | `null` | Async checkpoint save (`null` = framework default). |
| `ckpt_fully_parallel_load` | `false` | Load shards in parallel. |
| `ckpt_assume_constant_structure` | `false` | Assume identical layer structure across ranks. |
| `dist_ckpt_strictness` | `assume_ok_unexpected` | How to handle unexpected keys in distributed ckpt. |
| `dist_ckpt_save_pre_mcore_014` | `null` | Compatibility flag for older Megatron-Core checkpoints. |
| `dist_ckpt_optim_fully_reshardable` | `null` | Optimizer state fully reshardable layout. |
| `auto_continue_train` | `false` | *Primus:* resume from latest checkpoint in the save directory when enabled. |
| `disable_last_saving` | `false` | *Primus:* skip writing the final checkpoint at shutdown. |

---

## 7. Data

*Source: `trainer_base.yaml`.*

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data_path` | `null` | Single blended dataset path / list. |
| `data_sharding` | `true` | Shard data across ranks. |
| `split` | `"99,1,0"` | Train/valid/test split ratios as comma string. |
| `train_data_path` | `null` | Training data blend. |
| `valid_data_path` | `null` | Validation data blend. |
| `test_data_path` | `null` | Test data blend. |
| `data_args_path` | `null` | External JSON/YAML of dataset arguments. |
| `per_split_data_args_path` | `null` | Per-split dataset args file. |
| `data_cache_path` | `null` | On-disk cache for indexed datasets. |
| `mock_data` | `false` | Use synthetic data (no real files). |
| `merge_file` | `null` | Merge file for blended datasets. |
| `seq_length` | `4096` | Training sequence length. |
| `encoder_seq_length` | `null` | Encoder sequence length (encoder–decoder). |
| `decoder_seq_length` | `null` | Decoder sequence length. |
| `retriever_seq_length` | `256` | Sequence length for retriever models. |
| `sample_rate` | `1.0` | Sampling rate for dataset blending. |
| `mask_prob` | `0.15` | MLM mask probability. |
| `short_seq_prob` | `0.1` | Probability of shorter sequences in BERT-style data. |
| `num_workers` | `8` | DataLoader worker processes per rank. |
| `reset_position_ids` | `false` | Reset position IDs at document boundaries. |
| `reset_attention_mask` | `false` | Reset attention mask at boundaries. |
| `eod_mask_loss` | `false` | Mask loss at end-of-document tokens. |
| `dataloader_type` | `null` | Dataloader implementation (`single`, `cyclic`, `external`, …). |
| `mmap_bin_files` | `true` | Memory-map `.bin` index files when supported. |
| `create_attention_mask_in_dataloader` | `true` | Build attention masks in the dataloader. |
| `num_dataset_builder_threads` | `1` | Threads to build dataset indices. |

---

## 8. Recomputation (activation checkpointing)

*Sources: `trainer_base.yaml` and `primus_megatron_module.yaml`.*

| Parameter | Default | Description |
|-----------|---------|-------------|
| `recompute_activations` | `false` | Enable activation recomputation globally. |
| `recompute_granularity` | `null` | `full` or `selective` checkpointing. |
| `recompute_method` | `null` | `uniform` or `block` selective recomputation. |
| `recompute_num_layers` | `null` | Layers to recompute per block / schedule. |
| `recompute_layer_ids` | `null` | *Primus:* explicit layer indices to recompute within each PP stage (`0 … num_layers_per_pp_stage-1`). |
| `distribute_saved_activations` | `false` | Distribute saved activations across TP/PP for memory balance. |
| `checkpoint_activations` | `false` | Deprecated alias for activation checkpointing. |
| `moe_layer_recompute` | `false` | Recompute MoE layer activations (model preset). |

---

## 9. Logging and profiling

*Sources: `trainer_base.yaml` and `primus_megatron_module.yaml`.*

### 9.1 Logging

| Parameter | Default | Description |
|-----------|---------|-------------|
| `log_avg_skip_iterations` | `2` | Skip first N iterations for throughput averaging. |
| `log_avg_reset_interval` | `10` | Reset moving averages periodically. |
| `log_params_norm` | `false` | Log L2 norms of parameters. |
| `log_num_zeros_in_grad` | `false` | Log fraction of zero gradients. |
| `log_throughput` | `false` | Log tokens/sec and timing. |
| `log_progress` | `false` | Verbose progress logging. |
| `timing_log_level` | `0` | Verbosity for timing logs. |
| `timing_log_option` | `minmax` | Aggregate style for timing (`minmax`, `all`, …). |
| `tensorboard_log_interval` | `1` | Steps between TensorBoard scalars. |
| `tensorboard_queue_size` | `1000` | TensorBoard event queue size. |
| `log_timers_to_tensorboard` | `false` | Write timer stats to TensorBoard. |
| `log_batch_size_to_tensorboard` | `false` | Log batch size. |
| `log_learning_rate_to_tensorboard` | `true` | Log LR. |
| `log_validation_ppl_to_tensorboard` | `false` | Log validation perplexity. |
| `log_memory_to_tensorboard` | `false` | Log memory usage. |
| `log_world_size_to_tensorboard` | `false` | Log distributed world size. |
| `log_loss_scale_to_tensorboard` | `true` | Log FP16/FP8 loss scale. |
| `wandb_project` | `null` | Weights & Biases project name. |
| `wandb_exp_name` | `null` | W&B run name. |
| `wandb_save_dir` | `null` | W&B local directory. |
| `wandb_entity` | `null` | W&B entity / team. |
| `enable_one_logger` | `true` | Enable NVIDIA OneLogger integration. |
| `one_logger_project` | `megatron-lm` | OneLogger project string. |
| `one_logger_run_name` | `null` | OneLogger run name. |
| `log_interval` | `100` | Console log interval in iterations. |
| `tensorboard_dir` | `null` | TensorBoard output directory. |
| `logging_level` | `null` | Python logging level override. |
| `config_logger_dir` | `""` | Directory for dumped config logs. |
| `one_logger_async` | `false` | Async OneLogger flushing. |
| `app_tag_run_name` | `null` | Application tag for telemetry. |
| `app_tag_run_version` | `0.0.0` | Application tag version. |
| `disable_tensorboard` | `true` | *Primus:* disable TensorBoard integration in Primus-wrapped runs. |
| `disable_wandb` | `true` | *Primus:* disable W&B. |
| `disable_mlflow` | `true` | *Primus:* disable MLflow. |
| `mlflow_run_name` | `null` | *Primus:* MLflow run name. |
| `mlflow_experiment_name` | `null` | *Primus:* MLflow experiment name. |
| `use_rocm_mem_info` | `false` | *Primus:* collect ROCm memory info via `rocm-smi` every step when `true`. |
| `use_rocm_mem_info_iters` | `[1, 2]` | *Primus:* iterations at which to log memory if `use_rocm_mem_info` is `false`. |

### 9.2 Profiling

| Parameter | Default | Description |
|-----------|---------|-------------|
| `profile` | `false` | Enable lightweight Nsight / CUDA profiling hooks. |
| `use_pytorch_profiler` | `false` | Enable `torch.profiler` regions. |
| `profile_ranks` | `[0]` | Ranks to profile. |
| `profile_step_start` | `10` | First step to profile. |
| `profile_step_end` | `12` | Last step to profile. |
| `iterations_to_skip` | `null` | Skip listed iterations in profiling. |
| `result_rejected_tracker_filename` | `null` | Log rejected samples to this file. |
| `enable_gloo_process_groups` | `true` | Create auxiliary Gloo groups for CPU-side ops. |
| `record_memory_history` | `false` | Record CUDA memory history (debug). |
| `memory_snapshot_path` | `snapshot.pickle` | Path for memory snapshot dumps. |
| `disable_profiler_activity_cpu` | `false` | *Primus:* omit CPU activities from profiler traces. |
| `torch_profiler_record_shapes` | `true` | *Primus:* record tensor shapes in PyTorch profiler. |
| `torch_profiler_with_stack` | `true` | *Primus:* capture Python stacks in profiler. |
| `torch_profiler_use_gzip` | `false` | *Primus:* gzip profiler outputs. |

---

## 10. Model architecture

*Sources: `primus/configs/models/megatron/language_model.yaml` and `primus/configs/models/megatron/primus_megatron_model.yaml`.*

### 10.1 Core architecture

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_legacy_models` | `false` | Use legacy Megatron model code paths. |
| `deprecated_use_mcore_models` | `false` | Deprecated flag for Megatron-Core models; prefer current `transformer_impl` + stack. |
| `model_type` | `gpt` | `gpt` or `mamba` family. |
| `num_layers` | `24` | Transformer layers (decoder or unified stack). |
| `encoder_num_layers` | `null` | Encoder depth (encoder–decoder). |
| `decoder_num_layers` | `null` | Decoder depth. |
| `hidden_size` | `1024` | Hidden / model width. |
| `num_attention_heads` | `16` | Attention heads. |
| `attention_backend` | `auto` | Attention kernel backend selection. |
| `group_query_attention` | `false` | Enable grouped-query attention (GQA). |
| `qk_layernorm` | `false` | LayerNorm on Q/K projections. |
| `qk_l2_norm` | `false` | L2-normalize Q/K vectors. |
| `num_query_groups` | `null` | Number of query groups for GQA; `null` means MHA. |
| `add_position_embedding` | `false` | Add absolute position embeddings (non-RoPE stacks). |
| `position_embedding_type` | `learned_absolute` | Position embedding style. |
| `max_position_embeddings` | `null` | Maximum sequence positions (context length cap). |
| `original_max_position_embeddings` | `null` | Original pretrained length for interpolation / scaling. |
| `untie_embeddings_and_output_weights` | `true` | Separate input embedding and LM head weights. |
| `ffn_hidden_size` | `null` | FFN hidden size; `null` often defaults via `hidden_size` heuristics. |
| `kv_channels` | `null` | Per-head KV channels override. |
| `hidden_dropout` | `0.1` | Dropout on residual / hidden states. |
| `attention_dropout` | `0.1` | Attention dropout. |
| `fp32_residual_connection` | `false` | Accumulate residuals in FP32. |
| `apply_residual_connection_post_layernorm` | `false` | Apply residual after (vs before) norm where supported. |
| `add_bias_linear` | `false` | Biases in linear / column-parallel layers. |
| `add_qkv_bias` | `false` | Biases in QKV projections. |
| `swiglu` | `true` | SwiGLU activation in FFN. |
| `quick_geglu` | `false` | Faster GeGLU path. |
| `openai_gelu` | `false` | OpenAI GELU variant. |
| `squared_relu` | `false` | Squared ReLU activation. |
| `rotary_base` | `10000` | RoPE base frequency. |
| `rotary_percent` | `1.0` | Fraction of head dim spanned by RoPE. |
| `rotary_interleaved` | `false` | Interleaved RoPE layout. |
| `rotary_seq_len_interpolation_factor` | `null` | Positional interpolation factor for long contexts. |
| `use_rotary_position_embeddings` | `null` | Force RoPE on/off; `null` follows model type. |
| `use_rope_scaling` | `false` | Enable LLaMA-style rope scaling. |
| `rope_scaling_factor` | `8.0` | Scaling factor for extended contexts (LLaMA-3 style). |
| `transformer_impl` | `transformer_engine` | Backend library (`transformer_engine`, `local`, …). |
| `rope_type` | `null` | `rope` or `yarn` style extensions. |
| `norm_epsilon` | `1.0e-05` | LayerNorm / RMSNorm epsilon. |
| `normalization` | `"LayerNorm"` | Norm type (`LayerNorm`, `RMSNorm` with TE, …). |
| `apply_layernorm_1p` | `false` | LayerNorm with +1 offset trick. |
| `clone_scatter_output_in_embedding` | `true` | Clone embedding scatter for autograd safety. |
| `perform_initialization` | `true` | Run weight initialization. |
| `use_cpu_initialization` | `null` | Initialize on CPU then move to GPU. |
| `use_te_activation_func` | `false` | Use Transformer Engine activation kernels. |
| `gradient_accumulation_fusion` | `true` | Fuse gradient accumulation kernels. |
| `delay_wgrad_compute` | `false` | Delay weight-gradient computation for scheduling. |

### 10.2 Tokenizer and vocabulary

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tokenizer_type` | `null` | Tokenizer class name (`GPT2BPETokenizer`, `HuggingFaceTokenizer`, …). |
| `tokenizer_model` | `null` | Path to tokenizer model / vocabulary file. |
| `vocab_size` | `null` | Vocabulary size (often inferred from tokenizer). |
| `vocab_file` | `null` | Vocabulary file path for BPE/WP tokenizers. |
| `vocab_extra_ids` | `0` | Extra reserved token slots. |
| `tiktoken_pattern` | `null` | Regex pattern for tiktoken. |
| `tiktoken_num_special_tokens` | `1000` | Special token count for tiktoken setup. |
| `tiktoken_special_tokens` | `null` | Serialized special tokens for tiktoken. |
| `legacy_tokenizer` | `false` | Legacy tokenizer behavior. |
| `trust_remote_code` | `false` | `trust_remote_code` for Hugging Face tokenizers. |

### 10.3 Initialization and attention numerics

| Parameter | Default | Description |
|-----------|---------|-------------|
| `init_method_std` | `0.02` | Standard deviation for weight init. |
| `apply_query_key_layer_scaling` | `false` | Scale Q/K by layer index (deprecated GPT-3 trick). |
| `attention_softmax_in_fp32` | `false` | Force softmax in FP32. |

### 10.4 Kernel fusion flags

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bias_gelu_fusion` | `true` | Fuse bias + GELU. |
| `cross_entropy_loss_fusion` | `false` | Fused cross-entropy + softmax. |
| `cross_entropy_fusion_impl` | `"native"` | `native` or `te` fused CE. |
| `bias_swiglu_fusion` | `true` | Fuse bias + SwiGLU. |
| `masked_softmax_fusion` | `true` | Fused masked softmax. |
| `no_persist_layer_norm` | `false` | Non-persistent LayerNorm mode in TE. |
| `bias_dropout_fusion` | `true` | Fuse bias + dropout. |
| `apply_rope_fusion` | `true` | Fused RoPE kernels. |

### 10.5 Multi-Latent Attention (MLA)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `multi_latent_attention` | `false` | Enable MLA blocks instead of standard MHA. |
| `q_lora_rank` | `null` | Low-rank query projection rank. |
| `kv_lora_rank` | `32` | Low-rank KV compression rank. |
| `qk_head_dim` | `128` | Q/K head dimension for MLA. |
| `qk_pos_emb_head_dim` | `64` | Positional head dimension for MLA. |
| `v_head_dim` | `128` | Value head dimension for MLA. |
| `rotary_scaling_factor` | `1.0` | RoPE scaling inside MLA (distinct from `rope_scaling_factor` above). |
| `mscale` | `1.0` | Yarn / scaling m-factor. |
| `mscale_all_dim` | `1.0` | Yarn scaling on all dims. |

### 10.6 Mixture-of-Experts (MoE)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_experts` | `null` | Experts per MoE layer; `null` means dense model. |
| `moe_layer_freq` | `1` | Every Nth layer is MoE (1 = every layer). |
| `moe_ffn_hidden_size` | `null` | Expert FFN hidden size. |
| `moe_shared_expert_overlap` | `false` | Shared expert overlaps routing. |
| `moe_shared_expert_intermediate_size` | `null` | Shared expert FFN size. |
| `moe_grouped_gemm` | `false` | Grouped GEMM for experts. |
| `moe_router_load_balancing_type` | `"aux_loss"` | Router balancing (`aux_loss`, `seq_aux_loss`, `sinkhorn`, `none`). |
| `moe_router_dtype` | `null` | Router activation dtype (`fp32`, `fp64`). |
| `moe_router_score_function` | `softmax` | `softmax` or `sigmoid` routing scores. |
| `moe_router_topk` | `2` | Experts to select per token. |
| `moe_router_pre_softmax` | `false` | Apply softmax before top-k. |
| `moe_router_num_groups` | `null` | Group-limited routing: number of expert groups. |
| `moe_router_group_topk` | `null` | Groups to pick before top-k inside groups. |
| `moe_router_topk_scaling_factor` | `null` | Scaling for routing logits. |
| `moe_router_enable_expert_bias` | `false` | Learnable per-expert bias. |
| `moe_router_bias_update_rate` | `1.0e-03` | Update rate for expert bias. |
| `moe_use_legacy_grouped_gemm` | `false` | Legacy grouped GEMM path. |
| `moe_aux_loss_coeff` | `0.0` | Auxiliary load-balancing loss weight. |
| `moe_z_loss_coeff` | `null` | Router z-loss coefficient. |
| `moe_input_jitter_eps` | `null` | Input jitter for router stability. |
| `moe_token_dispatcher_type` | `allgather` | Token dispatch algorithm (`allgather`, `alltoall`, `flex`, `alltoall_seq`). |
| `moe_enable_deepep` | `false` | DeepEP-style expert parallelism. |
| `moe_per_layer_logging` | `false` | Per-layer MoE statistics logging. |
| `moe_expert_capacity_factor` | `null` | Capacity factor for token dropping / padding. |
| `moe_pad_expert_input_to_capacity` | `false` | Pad expert batches to capacity. |
| `moe_token_drop_policy` | `probs` | Token dropping policy when over capacity. |
| `moe_extended_tp` | `false` | Extended tensor-parallel for experts. |
| `moe_use_upcycling` | `false` | Expert upcycling initialization. |
| `moe_permute_fusion` | `false` | Fuse token permutation for MoE. |
| `disable_primus_topk_router` | `false` | *Primus:* disable Primus top-k router patch. |
| `moe_router_force_load_balancing` | `false` | *Primus:* force load-balanced routing. |
| `use_deprecated_20241209_moe_layer` | `false` | *Primus:* legacy MoE layer implementation. |

### 10.7 Logit softcapping (Primus / Grok-style)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `final_logit_softcapping` | `null` | Softcap value for final logits; `null` disables. |
| `attn_logit_softcapping` | `null` | Softcap for attention logits. |
| `router_logit_softcapping` | `null` | Softcap for MoE router logits. |

---

## 11. Primus extensions

### 11.1 Build and compile

| Parameter | Default | Description |
|-----------|---------|-------------|
| `disable_compile_dependencies` | `true` | *Primus:* avoid compiling dependency stacks in the trainer wrapper. |

### 11.2 Primus-Turbo (`primus_turbo.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_primus_turbo` | `false` | Master switch for Primus-Turbo integrations. Many sub-features require this plus specific kernels. |
| `use_turbo_attention` | `false` | Turbo attention implementation. |
| `use_turbo_parallel_linear` | `false` | Turbo parallel linear layers. |
| `use_turbo_grouped_mlp` | `false` | Turbo grouped MLP / SwiGLU path. |
| `moe_use_fused_router_with_aux_score` | `false` | Fused MoE router with auxiliary scores. |
| `enable_turbo_attention_float8` | `false` | FP8 path inside Turbo attention (spacing in YAML is normalized to this key). |
| `use_turbo_deepep` | `false` | Turbo DeepEP expert communication. |
| `turbo_deepep_num_cu` | `32` | DeepEP compute units / channels. |
| `turbo_deepep_use_comm_stream` | `false` | Use a dedicated communication stream for DeepEP. |
| `turbo_sync_free_moe_stage` | `0` | Stage selector for sync-free MoE. |
| `use_turbo_fused_act_with_probs` | `false` | Fuse activation + probability tensors to remove redundant work. |
| `use_turbo_rms_norm` | `false` | Turbo RMSNorm kernels. |

### 11.3 Zero-bubble pipeline (`zero_bubble.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `patch_zero_bubble` | `false` | Install Primus zero-bubble PP patches when `true`. |
| `debug_scheduler_table` | `false` | Print PP scheduler tables (also in `primus_pipeline.yaml`; last merge wins—defaults match). |
| `enable_zb_runtime` | `true` | Unified runtime for zero-bubble and related schedules. |
| `pre_communication_optimization` | `false` | Issue a tiny comm before real comm to tune overlap. |
| `zero_bubble_pipeline_timers_start_iter` | `100` | Start iter for auto-scheduler timers. |
| `zero_bubble_pipeline_timers_end_iter` | `110` | End iter for auto-scheduler timers. |
| `zero_bubble_max_pending_backward` | `auto` | Max pending backward ops (ZB1p vs ZB2p style); `auto` adapts. |
| `zero_bubble_adaptive_memory_limit_percentile` | `85` | GPU memory percentile cap for adaptive ZB. |
| `enable_optimizer_post_validation` | `false` | Post-optimizer validation step (needs FSDP path). |
| `enable_exactly_numeric_match` | `true` | Require bitwise match in post validation when enabled. |
| `enable_zero_bubble` | `true` | Enable zero-bubble schedule features in the ZB runtime. |
| `zero_bubble_v_schedule` | `false` | Zero-bubble “V” schedule without extra memory vs some baselines. |
| `zero_bubble_v_schedule_mem_setup` | `half` | Memory setup variant: `half`, `min`, or `zb`. |
| `enable_1f1b_v` | `false` | 1F1B-V schedule variant. |
| `allow_padding_num_layers` | `true` | Allow PP layer padding for divisibility. |
| `profile_memory_iter` | `-1` | Iteration to profile memory (`-1` disables). |
| `interleave_group_size` | `0` | Interleaved PP group size. |
| `offload_chunk_num` | `0` | Activation offload chunk count. |
| `offload_time` | `1.0` | Time budget for offload (scheduler hint). |
| `auto_offload_time` | `true` | Auto-tune offload timing. |
| `offload_overlap_sr` | `true` | Overlap save/resume in offload path. |
| `num_seq_splits` | `1` | Splits along sequence dimension for ZB. |
| `cpu_offload` | `false` | CPU offload of activations in ZB path. |

### 11.4 Primus pipeline (`primus_pipeline.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `patch_primus_pipeline` | `false` | Enable Primus pipeline scheduling patches. |
| `pp_algorithm` | `"1f1b-interleaved"` | Schedule name (`1f1b`, `1f1b-interleaved`, `zero-bubble`, `zbv-formatted`, `v-half`, `v-min`). |
| `communication_method` | `"async_p2p"` | `async_p2p` or `batch_p2p` PP transfers. |
| `offload` | `false` | Generic PP activation offload toggle in Primus pipeline. |
| `offload_ops` | `""` | Comma-separated offload targets (`attn` today; other ops listed in-file are not supported yet). |
| `pp_warmup` | `false` | *Primus:* warm-up PP stages to reduce first-iteration latency. |
| `dump_pp_data` | `false` | *Primus:* dump PP tensors for debugging. |

---

## 12. Reinforcement learning and GRPO-related settings

*Source: `trainer_base.yaml`. Names follow Megatron’s `grpo_*` / `rl_*` prefixes (there is no `rl_grpo` single flag in these presets).*

| Parameter | Default | Description |
|-----------|---------|-------------|
| `perform_rl_step` | `false` | Run RL / preference optimization steps (GRPO / LangRL integration). |
| `rl_prompts_per_eval` | `32` | Prompts per RL evaluation pass. |
| `grpo_prompts_per_step` | `32` | GRPO prompts sampled per training step. |
| `grpo_group_size` | `2` | Samples per prompt group for GRPO. |
| `grpo_iterations` | `2` | Inner GRPO iterations. |
| `grpo_clamp_eps_lower` | `0.01` | PPO-style lower clip epsilon. |
| `grpo_clamp_eps_upper` | `0.01` | Upper clip epsilon. |
| `grpo_kl_beta` | `0.001` | KL penalty weight toward reference policy. |
| `grpo_entropy_term_weight` | `0.0` | Entropy bonus weight. |
| `grpo_filter_groups_with_same_reward` | `false` | Drop groups with identical rewards. |
| `grpo_default_temperature` | `1.0` | Default softmax temperature for rollouts. |
| `grpo_default_top_p` | `0` | Top-p sampling (`0` often means disabled / greedy—see Megatron RL docs). |
| `langrl_inference_server_type` | `inplace_megatron` | LangRL inference backend. |
| `langrl_inference_server_conversation_template` | `null` | Conversation template path / name. |
| `langrl_env_config` | `null` | Environment / task YAML for LangRL. |
| `rl_offload_optimizer_during_inference` | `false` | Offload optimizer to CPU during rollout inference. |
| `rl_offload_kv_cache_during_training` | `false` | Offload KV cache while training forward runs. |
| `rl_remove_kv_cache_during_training` | `false` | Drop KV cache between RL phases to save memory. |
| `rl_reset_cuda_graphs` | `false` | Reset CUDA graphs when switching RL phases. |
| `rl_partial_rollouts` | `false` | Partial sequence rollouts. |
| `rl_inference_logprobs_is_correction` | `false` | Interpret inference logprobs as IS correction term. |
| `rl_importance_sampling_truncation_coef` | `null` | Truncate importance ratios at this value. |
| `rl_calculate_intra_group_similarity` | `false` | Log similarity within GRPO groups. |

---

## 13. Additional specialized parameters

*Source: `trainer_base.yaml` (remaining domains).*

### 13.1 Vision pretraining

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vision_pretraining` | `false` | Enable vision backbone pretraining. |
| `vision_pretraining_type` | `classify` | Objective (`classify`, etc.). |
| `vision_backbone_type` | `vit` | Vision backbone family. |
| `swin_backbone_type` | `tiny` | Swin variant size. |
| `num_classes` | `1000` | Classification classes. |
| `img_h` | `224` | Image height. |
| `img_w` | `224` | Image width. |
| `num_channels` | `3` | Input channels. |
| `patch_dim` | `16` | ViT patch size. |
| `classes_fraction` | `1.0` | Fraction of classes used. |
| `data_per_class_fraction` | `1.0` | Fraction of data per class. |

### 13.2 RETRO

| Parameter | Default | Description |
|-----------|---------|-------------|
| `retro_project_dir` | `null` | RETRO project directory with indices. |
| `retro_add_retriever` | `false` | Add frozen retriever tower. |
| `retro_cyclic_train_iters` | `null` | Cyclic iterator length. |
| `retro_encoder_layers` | `2` | Retriever encoder layers. |
| `retro_encoder_hidden_dropout` | `0.1` | Retriever dropout. |
| `retro_encoder_attention_dropout` | `0.1` | Retriever attention dropout. |
| `retro_num_neighbors` | `2` | Neighbors per query chunk. |
| `retro_num_retrieved_chunks` | `2` | Chunks concatenated per neighbor set. |
| `retro_attention_gate` | `1` | Gating between retrieval and LM. |
| `retro_verify_neighbor_count` | `true` | Assert neighbor counts for debugging. |

### 13.3 DINO self-supervised

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dino_local_img_size` | `96` | Local crop size. |
| `dino_local_crops_number` | `10` | Number of local crops. |
| `dino_head_hidden_size` | `2048` | Projection head width. |
| `dino_bottleneck_size` | `256` | Bottleneck dimension. |
| `dino_freeze_last_layer` | `1` | Freeze last layer epochs. |
| `dino_norm_last_layer` | `false` | Normalize last layer weights. |
| `dino_warmup_teacher_temp` | `0.04` | Teacher temperature warmup start. |
| `dino_teacher_temp` | `0.07` | Teacher temperature. |
| `dino_warmup_teacher_temp_epochs` | `30` | Epochs to warm teacher temperature. |

### 13.4 Biencoder / ICT / retriever utilities

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ict_head_size` | `null` | ICT projection head width. |
| `biencoder_projection_dim` | `0` | Biencoder shared projection dimension. |
| `biencoder_shared_query_context_model` | `false` | Share query/context encoders. |
| `ict_load` | `null` | ICT checkpoint path. |
| `bert_load` | `null` | BERT encoder checkpoint for biencoder. |
| `titles_data_path` | `null` | Titles file for ICT datasets. |
| `query_in_block_prob` | `0.1` | Probability of in-block queries. |
| `use_one_sent_docs` | `false` | Single-sentence pseudo documents. |
| `evidence_data_path` | `null` | Evidence passages for open-domain QA. |
| `retriever_report_topk_accuracies` | `[]` | k values for top-k accuracy logging. |
| `retriever_score_scaling` | `false` | Scale retriever scores. |
| `block_data_path` | `null` | Block JSON data for retrieval. |
| `embedding_path` | `null` | Precomputed embeddings path. |
| `indexer_batch_size` | `128` | Batch size when building ANN index. |
| `indexer_log_interval` | `1000` | Indexer progress log interval. |

### 13.5 Straggler detection

| Parameter | Default | Description |
|-----------|---------|-------------|
| `log_straggler` | `false` | Log straggler diagnostics. |
| `disable_straggler_on_startup` | `false` | Skip straggler detection at startup. |
| `straggler_ctrlr_port` | `65535` | Controller port for straggler service. |
| `straggler_minmax_count` | `1` | Min/max samples for straggler stats. |

### 13.6 Inference-oriented options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `inference_batch_times_seqlen_threshold` | `-1` | Heuristic threshold tying batch and sequence length. |
| `inference_dynamic_batching` | `false` | Dynamic batching for inference server. |
| `inference_dynamic_batching_buffer_size_gb` | `40.0` | GPU buffer budget (GB). |
| `inference_dynamic_batching_buffer_guaranteed_fraction` | `0.2` | Minimum reserved fraction of buffer. |
| `inference_dynamic_batching_buffer_overflow_factor` | `null` | Overflow growth factor. |
| `inference_dynamic_batching_max_requests_override` | `null` | Hard cap on concurrent requests. |
| `inference_dynamic_batching_max_tokens_override` | `null` | Hard cap on tokens in flight. |
| `max_tokens_to_oom` | `12000` | Token limit guard before OOM abort. |
| `output_bert_embeddings` | `false` | Return BERT pooled embeddings. |
| `bert_embedder_type` | `megatron` | `megatron` or `huggingface` embedder. |
| `flash_decode` | `false` | Flash decode kernels for incremental generation. |
| `enable_cuda_graph` | `false` | Capture CUDA graphs for inference. |
| `cuda_graph_warmup_steps` | `3` | Warm-up steps before capturing graphs. |
| `external_cuda_graph` | `false` | External graph provider hooks. |
| `cuda_graph_scope` | `full` | Graph scope (`full` or `attn`). |
| `inference_max_requests` | `8` | Max concurrent requests. |
| `inference_max_seq_length` | `2560` | Max prefill + decode tokens per request. |

### 13.7 Fault tolerance package and tooling

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_ft_package` | `false` | NVIDIA fault-tolerance package hooks. |
| `calc_ft_timeouts` | `false` | Auto-calculate FT timeouts. |
| `run_workload_inspector_server` | `false` | Run workload inspector sidecar. |

### 13.8 Heterogeneous layers and process resilience

| Parameter | Default | Description |
|-----------|---------|-------------|
| `heterogeneous_layers_config_path` | `null` | JSON describing variable layer widths/types per layer. |
| `heterogeneous_layers_config_encoded_json` | `null` | Inline base64/JSON blob for heterogeneous layers. |
| `inprocess_restart` | `false` | In-process restart for fault recovery experiments. |

### 13.9 Experimental and rerun controls

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_experimental` | `false` | Gate experimental Megatron features. |
| `error_injection_rate` | `0` | Fraction of iterations with injected errors (testing). |
| `error_injection_type` | `transient_error` | `correct_result`, `transient_error`, or `persistent_error`. |
| `rerun_mode` | `disabled` | `disabled`, `validate_results`, or `report_stats` for rerun harness. |

---

### See also

- Megatron-LM argument definitions: [`megatron/training/arguments.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/arguments.py)
- Primus Megatron presets: `primus/configs/modules/megatron/`
- Primus Megatron model presets: `primus/configs/models/megatron/`
