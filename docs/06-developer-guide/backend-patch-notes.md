# Backend patch notes

Primus integrates several large-model backends (Megatron-LM, TorchTitan, JAX MaxText, …) and applies a lightweight patch layer to keep configuration flags consistent with the Primus CLI. This page captures those backend-specific switches so they live alongside the rest of the documentation (instead of the historical `primus/README_patch.md` file).

## How to read these notes

- Start with the **Base Module Parameters** table below—every backend module inherits these knobs.
- Jump to the backend-specific section for details on extra CLI/config options and links to the patched source files.
- When editing configs or CLI presets, cross-reference the [Primus CLI Reference](../02-user-guide/cli-reference.md) so command examples and backend parameters stay in sync.

## Supported models

This section lists, at a high level, the model families Primus currently targets on each backend. For more details and configuration examples, refer to the backend-specific patch notes below.

### Megatron-LM

- **LLaMA family**: LLaMA2, LLaMA3, LLaMA3.1, LLaMA3.3, LLaMA4 (various sizes from 7B up to 405B+)
- **DeepSeek family**: DeepSeek-V2 (lite/base/full) and DeepSeek-V3
- **MoE / Mixtral**: Mixtral-8x7B / 8x22B, large MoE configs (515B, 1T, 2T, 4T) and DeepSeek-style MoE
- **Qwen family**: Qwen2.5 (7B/72B) and Qwen3 (8B/30B/235B variants)
- **Other GPT-style models**: Grok1/2, GPT-OSS 20B and generic `language_model.yaml`

### TorchTitan

- **LLaMA family**: LLaMA3, LLaMA3.1, LLaMA3.3 (various sizes, including FP8 variants)
- **DeepSeek family**: DeepSeek-V3 (16B and 671B, FP8 and BF16 configs)
- **Qwen family**: Qwen3 small/medium models (0.6B, 1.7B, 32B)

### JAX MaxText

- **LLaMA family**: LLaMA2 (7B/70B), LLaMA3 (8B/70B), LLaMA3.3 (70B)
- **DeepSeek family**: DeepSeek-V2 16B
- **MoE / Mixtral**: Mixtral-8x7B
- **Other models**: Grok1 and additional MaxText-supported transformers (see MaxText docs for the full list)

## Base module parameters

All modules inherit the options defined in [`primus/configs/modules/module_base.yaml`](https://github.com/AMD-AGI/Primus/blob/main/primus/configs/modules/module_base.yaml):

| Argument Name       | Default Value | Description                                                                                |
| ------------------- | ------------- | ------------------------------------------------------------------------------------------ |
| `trainable`         | `false`       | Whether the module participates in training.                                              |
| `sink_level`        | `null`        | Global sink level for logging. Overrides `file_sink_level` and `stderr_sink_level` if set. |
| `file_sink_level`   | `DEBUG`       | Logging level for file sink (log files).                                                  |
| `stderr_sink_level` | `INFO`        | Logging level for stderr/console output.                                                  |

### Backend index

- [Megatron-LM patch notes](#megatron-lm-patch-notes)
- [TorchTitan patch notes](#torchtitan-patch-notes)
- [JAX MaxText patch notes](#jax-maxtext-patch-notes)

---

## Megatron-LM patch notes

Primus keeps a curated patch layer on top of upstream Megatron-LM so CLI presets and configs can expose additional controls. Use this section with the [Base Module Parameters](#base-module-parameters) above for shared module parameters, and the [Primus CLI Reference](../02-user-guide/cli-reference.md) for CLI/config usage patterns.

### 1. Module-level parameters

These arguments are introduced in the Megatron module logic (e.g., training loop, logging, resume logic). They are defined via patching and can be configured to control training behavior and logging utilities.

| New Argument                         | Default Value | Description                                                                                    | Patched Files                                                                                                                                                                                                                                                                                                | Notes                                            |
| ------------------------------------ | ------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------ |
| `disable_tensorboard`                | `true`        | Whether to disable TensorBoard. Set to `false` if you want to enable profiling or torch trace. | NA                                                                                                                                                                                                                                                                                                           | Required for timeline and performance debugging. |
| `disable_wandb`                      | `true`        | Whether to disable Weights & Biases logging.                                                   | NA                                                                                                                                                                                                                                                                                                           | Useful for internal benchmarking.                |
| `disable_compile_dependencies`       | `true`        | Disables Megatron’s custom kernel compilation. Most ops are already covered by TE.             | NA                                                                                                                                                                                                                                                                                                           | Avoids redundant compilation steps.              |
| `auto_continue_train`                | `false`       | Automatically resume training from the latest checkpoint if found in the `--save` path.        | NA                                                                                                                                                                                                                                                                                                           | Simplifies job restarts.                         |
| `disable_last_saving`                | `false`       | Skip saving the final checkpoint at the last iteration.                                        | NA                                                                                                                                                                                                                                                                                                           | Useful for profiling or benchmarking runs.       |
| `no_fp8_weight_transpose_cache`      | `false`       | Disable the FP8 weight transpose cache to save memory.                                         | `megatron.core.extensions.transformer_engine.TELinear`, `megatron.core.extensions.transformer_engine.TELayerNormColumnParallelLinear`, `megatron.core.extensions.transformer_engine.TEDelayedScaling`                                                                                                        | May affect performance but reduce memory use.    |
| `decoder_pipeline_manual_split_list` | `null`        | Enable manual pipeline split in (interleaved) 1F1B pipeline parallelism.                       | `megatron.core.transformer.transformer_block.get_num_layers_to_build`, `megatron.core.transformer.transformer_layer.get_transformer_layer_offset`                                                                                                                                                            | Deprecated. Use `pipeline_model_parallel_layout` instead.    |
| `pp_warmup`                          | `false`       | Add fwd/bwd warmup to save iter1's time when pp degree is large.                             | NA                                                                                                                                                                                                                                                                                                           | Can save much time for pipeline debug.           |
| `dump_pp_data`                       | `false`       | Enable dumping pp schedule data for visualization.                                             | `megatron.core.pipeline_parallel.schedules.forward_step`, `megatron.core.pipeline_parallel.schedules.backward_step`, `megatron.core.pipeline_parallel.schedules.forward_backward_pipelining_with_interleaving`, `megatron.core.pipeline_parallel.schedules.forward_backward_pipelining_without_interleaving` | Useful for pipeline schedule visualization.      |
| `disable_profiler_activity_cpu`                | `false`       | Disable CPU activity in torch profiling.                                        |  NA                                                                                                                                                                                                                                                                                                         | If you only want to trace CUDA kernels and get a smaller trace JSON file, you can enable this option. However, if you plan to run with TraceLen, please do not enable it.  more torch profiler args: <br>`torch_profiler_record_shapes: true`, <br>`torch_profiler_with_stack: true`, <br>`torch_profiler_use_gzip: true`      |
| `use_rocm_mem_info`                        | `false`       | Logging ROCm memory information in Megatron-LM Trainer                             | NA                                                                                                                                                                                                                                                                                                           | If `use_rocm_mem_info = True`, ROCm memory information will be collected with `rocm-smi` at every iteration.           |
| `use_rocm_mem_info_iters`                        | `[1,2]`       | Logging ROCm memory information in Megatron-LM Trainer for some iterations                            | NA                                                                                                                                                                                                                                                                                                           | If `use_rocm_mem_info = False`, ROCm memory information will be collected at the iterations specified in `use_rocm_mem_info_iters`.           |
| `patch_zero_bubble`                        | `false`       | Using Zero-Bubble pipeline parallism  | `megatron.core.optimizer.ChainedOptimizer`, `megatron.core.pipeline_parallel.get_forward_backward_func`, `megatron.core.tensor_parallel.layers.LinearWithGradAccumulationAndAsyncCommunication`, `megatron.core.parallel_stat.default_embedding_ranks`, `megatron.core.parallel_stat.is_pipeline_last_stage`, `megatron.core.parallel_stat.is_rank_in_embedding_group`, `megatron.core.distributed.finalize_model_grads`, `megatron.core.transformer.transformer_layer.get_transformer_layer_offset` | If `patch_zero_bubble = True`, Zero bubble pipeline parallism will be enable to use. See more detail at [ZeroBubble User Guide](https://github.com/AMD-AGI/Primus/blob/main/primus/backends/megatron/core/pipeline_parallel/zerobubble/README.md)           |
| `disable_mlflow`                        | `true`       | Track model development using MLflow                             | NA                                                                                                                                                                                                                                                                                                         |  Envs: <br> `export DATABRICKS_TOKEN=your_token`<br>`export DATABRICKS_HOST=your_host`<br>`export MLFLOW_TRACKING_URI=databricks`<br>`export MLFLOW_REGISTRY_URI=databricks-uc`<br>Arguments: <br>`mlflow_run_name: null`,<br>`mlflow_experiment_name: null`     |
| `recompute_layer_ids`                    | `null`       | Specify the exact IDs of layers to recompute, enabling more flexible memory reduction                            | `megatron.core.transformer.transformer_block.TransformerBlock._checkpointed_forward`, `megatron.core.transformer.multi_token_prediction.MultiTokenPredictionLayer._checkpointed_forward`                                                                                                                                                                                                                                                                                                           | Using `recompute_layer_ids=[layer_id_0, layer_id_1,...]` together with `recompute_granularity=full` and `recompute_method=null`. Ids 0 to num_layers - 1 are decoder layers; the MTP depths continue the numbering, so MTP depth d is num_layers + d.           |
| `dataloader_mp_context`                  | `null`       | Set `DataLoader.multiprocessing_context` to avoid SIGSEGV caused by fork()-hostile native state (RDMA MRs, HIP runtime, IPC handles). | `torch.utils.data.DataLoader.__init__`                                                                                                                                                                                                                                                                       | Accepted values: `"forkserver"`, `"spawn"`, `"fork"`, or `null` (keep PyTorch default). Only takes effect when `num_workers > 0` and no explicit `multiprocessing_context` is passed. |

---

### 2. Model-definition parameters

These arguments affect the internal architecture or layer implementations. They are patched into the model construction logic and used for tuning or debugging specific variants.

| New Argument                        | Default Value | Description                                                               | Patched Files                                                                                                                                                                                                                                                                                                                   | Notes                                       |
| ----------------------------------- | ------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| `disable_primus_topk_router`   | `false`       | Disable PrimusTopkRouter and use TopkRouter implemented by megatron. | `megatron.core.transformer.moe.router.TopKRouter`                                                                                                                                                                                                                                                                               | Used to debug internal.         |
| `moe_router_force_load_balancing`   | `false`       | Force token redistribution in MoE to achieve load balance across experts. | `megatron.core.transformer.moe.router.TopKRouter`                                                                                                                                                                                                                                                                               | Use to debug MoE imbalance issues.          |
| `use_deprecated_20241209_moe_layer` | `false`       | Enable legacy MoE implementation for debugging/perf comparison.           | `megatron.core.transformer.moe.moe_layer.MoELayer`, `megatron.core.transformer.moe.moe_layer.MoESubmodules`, `megatron.core.transformer.moe.experts.GroupedMLP`, `megatron.core.transformer.moe.experts.SequentialMLP`, `megatron.core.transformer.moe.experts.TEGroupedMLP`, `megatron.core.transformer.moe.router.TopKRouter` | Deprecated, used for internal testing only. |
| `moe_permute_fusion`   | `false`       | Permutation and unpermutation fusion. | `megatron.core.extensions.transformer_engine`, `megatron.core.transformer.moe.moe_utils`                                                                                                                                                                                                                                                                               | Fuse permutation and unpermutation in moe layer.         |
| `moe_use_fused_router_with_aux_score`   | `false`       | Fused router topk and calculation of moe aux loss score. Need Primus turbo backend | `megatron.core.transformer.moe.router.TopKRouter`                                                                                                                                                                                                                                                                               | Used to reduce launch overhead of the small kernels in router.         |

### 3. Primus-Turbo related options

| New Argument                        | Default Value | Description                                                               | Patched Files                                                                                                                                                                                                                                                                                                                   | Notes                                       |
| ----------------------------------- | ------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| `use_turbo_gemm`   | `false`       | Use Primus-Turbo linear modules (`PrimusTurboLinear`, `PrimusTurboColumnParallelLinear`, `PrimusTurboRowParallelLinear`, `PrimusTurboLayerNormColumnParallelLinear`) in place of the TE linear modules. | `megatron.core.extensions.transformer_engine.TELinear`, `megatron.core.extensions.transformer_engine.TEColumnParallelLinear`, `megatron.core.extensions.transformer_engine.TERowParallelLinear`, `megatron.core.extensions.transformer_engine.TELayerNormColumnParallelLinear` | Accelerates dense GEMMs. Supports FP8 recipes (`tensorwise`, `blockwise`, `mxfp8`) and FP4 recipe (`mxfp4`). Replaces the deprecated `use_turbo_parallel_linear`. **Please set `enable_primus_turbo=True` first.** |
| `use_turbo_grouped_gemm`   | `false`       | Use Primus-Turbo grouped GEMM (`PrimusGroupedMLP` with `PrimusTurboColumnParallelGroupedLinear` / `PrimusTurboRowParallelGroupedLinear`) for MoE experts in place of `TEGroupedMLP`. | `megatron.core.transformer.moe.experts.TEGroupedMLP`, `megatron.core.extensions.transformer_engine.TEColumnParallelGroupedLinear`, `megatron.core.extensions.transformer_engine.TERowParallelGroupedLinear` | Accelerates MoE grouped GEMMs. Incompatible with `moe_use_legacy_grouped_gemm=True`. Required by Sync-Free MoE stage 2/3. Replaces the deprecated `use_turbo_grouped_mlp`. **Please set `enable_primus_turbo=True` first.** |
| `use_turbo_permute_padding`   | `false`       | Pad tokens of every experts to 16 or 32 multiple to reduce d2h and h2d. | `megatron.core.transformer.moe.token_dispatcher.MoEFlexTokenDispatcher`, `megatron.core.transformer.moe.experts.TEGroupedMLP` | Only effective under FP8/FP4 with `use_turbo_deepep=True`. Pad multiple is 16/32 (FP8, depending on recipe) or 32 (FP4). **Please set `enable_primus_turbo=True` first.** |
| `use_turbo_deepep`   | `false`       | Use Primus-turbo `DeepEPTokenDispatcher`. | `megatron.core.transformer.moe.token_dispatcher.MoEFlexTokenDispatcher`                                                                                                                                                                                                                                                              | Used Primus-Turbo DeepEP to accelerate MoE token dispatcher.  **You must both set`enable_primus_turbo=True` and `use_turbo_deepep=True` to enable this function.**         |
| `turbo_deepep_num_cu`   | `32`       | Set the number of CUs to use for Primus-Turbo DeepEP. |   | 64 or 80 for ep8, 32 for ep16-64 is best practice.  |
| `turbo_deepep_use_comm_stream`   | `false`       | Primus-Turbo DeepEP will use an internal stream to dispatch/combine when enabled, default used `current_stream` |   |  **Please both set`enable_primus_turbo=True` and `use_turbo_deepep=True` first**
| `turbo_sync_free_moe_stage`   | `0`       | Primus Sync-Free MoE has 4 stages. See [RFC: Primus-Megatron SyncFree MoE](https://github.com/AMD-AGI/Primus/issues/203) for more details. |   |   stage 2 is recommended for better performance. **Please set`enable_primus_turbo=True` first**   |
| `use_turbo_mega_moe`   | `false`       | Replace the whole MoE layer with the FlyDSL-based fused Primus-Turbo MegaMoE layer (fused router + `dispatch_grouped_gemm` → SwiGLU → `grouped_gemm_combine`). | `megatron.core.transformer.moe.moe_layer.MoELayer` | EP-only: requires `tensor_model_parallel_size=1`, `params_dtype=bf16`, and an EP process group; asserts on unsupported router options (sequence/global aux loss, z-loss, sinkhorn, input jitter, expert bias). Needs a Primus-Turbo build with MegaMoE. See [MegaMoE guide](../04-technical-guides/mega-moe.md). **Please set `enable_primus_turbo=True` first.** |
| `turbo_mega_moe_precision`   | `bf16`       | Precision of the MegaMoE experts: `bf16` or `mxfp8`. `mxfp8` covers dispatch+fc1, SwiGLU, fc2+combine, and the dW1/dW2 wgrads. | | Read only when `use_turbo_mega_moe` is on; independent of Megatron's `--fp8`, which only selects the TE recipe for the dense layers. Parameters stay bf16 either way — the op maintains the mxfp8 weight quant internally, keyed on `w._version` — so checkpointing, init and the optimizer are unaffected. `mxfp8` is not compatible with CUDA-graph capture. **Please set `enable_primus_turbo=True` and `use_turbo_mega_moe=True` first.** |
| `turbo_mega_moe_wgrad_accum_fusion`   | `null`       | Whether the bf16 MegaMoE expert wgrads (dW1, dW2) accumulate straight into the weights' `main_grad` instead of through DDP's separate add (Primus-Turbo's `fuse_wgrad_accum_pattern`). `null` follows `gradient_accumulation_fusion`; `true` / `false` set it for the MegaMoE experts alone. | | Read only when `use_turbo_mega_moe` is on. Where the ops cannot accumulate into `main_grad` (it needs gfx1250 and a bf16 `main_grad`), they write the ordinary gradient and DDP adds it as before. Off under Megatron-FSDP, which clears `grad_added_to_main_grad` before every backward, and for `turbo_mega_moe_precision: mxfp8`, whose stages keep DDP's add. **Please set `enable_primus_turbo=True` and `use_turbo_mega_moe=True` first.** |

---

## TorchTitan patch notes

TorchTitan integration uses the same Primus configuration surface (CLI flags + YAML) but exposes a few extra knobs via patches. Pair this with the [Base Module Parameters](#base-module-parameters) above for shared module parameters.

| New Argument | Default Value | Description | Patched Files | Notes |
| ------------ | ------------- | ----------- | ------------- | ----- |
| `primus_turbo.enable_embedding_autocast` | `true` | Automatically casts `nn.Embedding` outputs to the AMP dtype (e.g., bf16) during training so downstream layers stay in sync. | (Primus TorchTitan patch set) | Disable only if you manage casting manually. |

---

## JAX MaxText patch notes

Primus integrates JAX MaxText as a backend for running LLaMA and related transformer models on AMD GPUs. The MaxText configuration surface (YAML + CLI) is used as-is, plus the ROCm `release/v26.10` additions below, which Primus layers onto any ROCm/maxtext `rocm-main` checkout as patches instead of requiring a forked MaxText.

Use this section together with:

- The [Base Module Parameters](#base-module-parameters) and [Supported Models](#supported-models) above for a high-level model overview
- The [Primus CLI Reference](../02-user-guide/cli-reference.md) for Primus CLI usage patterns
- The official MaxText documentation for the full set of MaxText-specific arguments

### MaxText-specific notes

- Primus currently wires MaxText via `primus/configs/models/maxtext` and `primus/configs/modules/maxtext`.
- Model families currently exercised in examples include:
  - LLaMA2 7B/70B
  - LLaMA3 8B/70B
  - LLaMA3.3 70B
  - DeepSeek-V2 16B
  - Mixtral-8x7B
  - Grok1

### ROCm/maxtext release/v26.10 patches

The commits `38d321cff`..`2fb48f50e` of [ROCm/maxtext `release/v26.10`](https://github.com/ROCm/maxtext/commits/release/v26.10) are carried as Primus patches on top of `third_party/maxtext`, which pins an unmodified ROCm/maxtext `rocm-main` commit (`57de8f3d7`), so it trains the same way as `release/v26.10`. The patches apply to whichever MaxText Primus loads: the checkout in `MAXTEXT_PATH` when that is set (the `rocm/jax-training` images export it, pointing at their own MaxText), otherwise `third_party/maxtext`. All patches run in the `setup` phase. Each one first checks whether the installed MaxText already contains its change and skips if so (a no-op on `release/v26.10` itself). Source-rewrite patches match exact anchors in MaxText. If an anchor is missing or ambiguous, `MaxTextPretrainTrainer.init()` raises before training starts instead of training without the change.

New config keys (set them like any other MaxText key):

| New Argument | Default Value | Description | Patched Files | Notes |
| ------------ | ------------- | ----------- | ------------- | ----- |
| `use_turbo_grouped_gemm` | `false` | Primus-Turbo grouped GEMM for MoE sparse matmul. | `maxtext.layers.moe.RoutedMoE.sparse_matmul` | Requires `sparse_matmul=True`, `megablox=False`, `use_tokamax_gmm=False`, no `quantization`, bf16/fp16 `dtype`, and `primus_turbo`. Supports ring-of-experts with `use_ragged_sort`. Implementation: `primus/backends/maxtext/kernels/turbo_grouped_gemm.py`. |
| `turbo_grouped_gemm_backend` | `ck` | `ck`, `hipblaslt` or `flydsl` (gfx950). | | Read only when `use_turbo_grouped_gemm` is on. |
| `turbo_grouped_gemm_quant` | `bf16` | `bf16` or `fp8` (per-tensor e4m3 forward, dA and dW). | | `fp8` needs backend `flydsl` or `hipblaslt` on gfx950. |
| `expert_balance` | `false` | Replace the router logits with a deterministic rotation so tokens spread uniformly over the experts. | `maxtext.layers.moe.RoutedMoE.dense_matmul` | Benchmarking aid; changes model numerics. |
| `jax_distributed_heartbeat_timeout_seconds` | `100` | Heartbeat timeout passed to every `jax.distributed.initialize`. | `maxtext.utils.max_utils.maybe_initialize_jax_distributed_system` | Increase for slow NFS checkpoint restores ("stop sending heartbeats"). |

Always-on behavior (no flag; leave the patch id out of `PRIMUS_PATCHES` to disable):

| Patch id | Change | Patched Files |
| -------- | ------ | ------------- |
| `maxtext.config.primus_fields` | Adds the keys above to `MaxTextConfig` (subclass bound as `pyconfig`'s default `config_class`). | `maxtext.configs.types.MaxTextConfig`, `maxtext.configs.pyconfig` |
| `maxtext.train.sync_before_loop` | `sync_global_devices` barrier right before the training loop. | `maxtext.trainers.pre_train.train.train_loop` |
| `maxtext.train.sync_each_step` | `jax.block_until_ready` on the train state after every step. | `maxtext.trainers.pre_train.train.training_loop_iteration` |
| `maxtext.train.device_info` | Logs every device and writes `<base_output_directory>/device_info.json`. | `maxtext.utils.max_utils.print_system_information`, `maxtext.trainers.pre_train.train.initialize` |
| `maxtext.moe.routing_maps_remat` | Saves the router top-k ids (`moe_routing_maps`) across remat in the `minimal*` policies. | `maxtext.layers.moe.RoutedMoE.get_topk`, `maxtext.layers.nnx_decoders.NNXDecoder.minimal_policy` |
| `maxtext.attention.synthetic_causal_mask` | `dataset_type: synthetic` uses a causal mask instead of building the S x S mask in TE attention. | `maxtext.layers.attention_op.AttentionOp.cudnn_flash_attention` |
| `maxtext.models.query_pre_attn_scalar` | `query_pre_attn_scalar=head_dim**-0.5` in the Gemma, Gemma2, Llama2, Mistral and Mixtral decoder layers. | `maxtext.models.{gemma,gemma2,llama2,mistral,mixtral}.*DecoderLayer.__init__` |
| `maxtext.lora.set_mesh_unbox` | Unboxes restored LoRA params under `jax.set_mesh(mesh)` (flax 0.12.2). | `maxtext.utils.lora_utils.setup_initial_lora_state` |

Patch registry. Every MaxText setup patch, the ROCm/maxtext commits it replaces, and when it can go. "Permanent" patches are AMD features or Primus benchmark/ops policy and stay in Primus. "General fix" patches are candidates for upstreaming: once the pinned MaxText contains the change, the patch skips itself (logged at startup), and it is deleted at the next pin bump. Upstream status is as of the pinned `third_party/maxtext` commit; a pin bump that changes it updates this table.

| Patch id | ROCm/maxtext origin | Category | Upstream | Remove when |
| -------- | ------------------- | -------- | -------- | ----------- |
| `maxtext.config.primus_fields` | `38d321cff`, `6d3eab29e`, `a1efcbc98`..`2fb48f50e` | Config keys for the patches below | No | Each key goes with the patch that reads it |
| `maxtext.moe.turbo_grouped_gemm` | `a1efcbc98`, `5f0be4640`, `6ef3e66f5`, `84734f14b`, `11628c4a8`, `2780b6c92`, `e1bd9de00`, `8eec68a4a`, `262f6e873`, `2fb48f50e` | AMD feature | No | Permanent |
| `maxtext.moe.expert_balance` | `38d321cff` | Benchmark policy | No | Permanent |
| `maxtext.train.sync_each_step` | `8d1d43c03` | Benchmark policy | No | Permanent |
| `maxtext.train.device_info` | `1c275c1e2` | Ops policy | No | Permanent |
| `maxtext.attention.synthetic_causal_mask` | `0aaedc18c` | Benchmark policy | No | Permanent |
| `maxtext.train.sync_before_loop` | `dd8204b2d` | General fix | No | MaxText has the change |
| `maxtext.distributed.heartbeat_timeout` | `6d3eab29e` | General fix | No | MaxText has the change |
| `maxtext.moe.routing_maps_remat` | `b2632e16d` | General fix | No | MaxText has the change |
| `maxtext.lora.set_mesh_unbox` | `4270c96a1` | General fix | No | MaxText has the change |
| `maxtext.models.query_pre_attn_scalar` | `ba34cb9e2` | General fix | No | MaxText has the change |

`maxtext.moe.turbo_grouped_gemm` also gives the `jax.lax.ragged_dot` fallback the same ring-of-experts `group_offset` window. The `src/maxtext/examples/shardings.py` part of `4270c96a1` is a standalone example script and is not ported.
