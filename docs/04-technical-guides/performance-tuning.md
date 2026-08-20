# Performance tuning guide

This guide covers AMD-focused performance work in Primus: HipBLASLt autotuning for GEMMs, **Primus-Turbo** optional kernels, mixed precision, activation recomputation, communication overlap, memory settings, and MoE-specific flags. It references Primus examples and Megatron module YAMLs.

---

## 1. HipBLASLt autotuning

Transformer Engine and GEMM-heavy training benefit from HipBLASLt kernel selection. Primus integrates a **three-stage** workflow controlled by `PRIMUS_HIPBLASLT_TUNING_STAGE` (see `examples/README.md` and `examples/run_pretrain.sh`).

> **Activate tuning first.** The stage variable is only honored when the master switch `PRIMUS_HIPBLASLT_TUNING=1` is set (and `PRIMUS_DETERMINISTIC` is not `1`). Without `PRIMUS_HIPBLASLT_TUNING=1`, both `run_pretrain.sh` and the CLI hook `runner/helpers/hooks/train/pretrain/prepare_experiment.sh` skip tuning entirely and force `TE_HIPBLASLT_TUNING_RUN_COUNT=0` / `TE_HIPBLASLT_TUNING_ALGO_COUNT=0`. Export `PRIMUS_HIPBLASLT_TUNING=1` alongside the stage in every command below.

### Stage 0 (default)

No tuning:

```bash
export PRIMUS_HIPBLASLT_TUNING_STAGE=0  # default
```

### Stage 1: Dump GEMM shapes

Run a **short** training job so shapes are collected during real forward/backward passes. Reduce `train_iters` (or equivalent) for faster shape collection.

```bash
export PRIMUS_HIPBLASLT_TUNING=1
export PRIMUS_HIPBLASLT_TUNING_STAGE=1
./runner/primus-cli direct -- train pretrain \
  --config examples/megatron/configs/MI300X/llama2_7B-BF16-pretrain.yaml
```

Output layout (from `examples/README.md`):

- `./output/tune_hipblaslt/${PRIMUS_MODEL}/gemm_shape`

### Stage 2: Offline tuning

Runs offline tuning from dumped shapes (often 10–30 minutes depending on model and shapes):

```bash
export PRIMUS_HIPBLASLT_TUNING=1
export PRIMUS_HIPBLASLT_TUNING_STAGE=2
./runner/primus-cli direct -- train pretrain \
  --config examples/megatron/configs/MI300X/llama2_7B-BF16-pretrain.yaml
```

Expected output:

- `./output/tune_hipblaslt/${PRIMUS_MODEL}/gemm_tune/tune_hipblas_gemm_results.txt`

### Stage 3: Train with tuned kernels

Point the runtime at the tuned override file:

```bash
export PRIMUS_HIPBLASLT_TUNING=1
export PRIMUS_HIPBLASLT_TUNING_STAGE=3
export HIPBLASLT_TUNING_OVERRIDE_FILE=/path/to/tune_hipblas_gemm_results.txt
./runner/primus-cli direct -- train pretrain \
  --config examples/megatron/configs/MI300X/llama2_7B-BF16-pretrain.yaml
```

### Related environment variables

| Variable | Role |
|----------|------|
| `TE_HIPBLASLT_TUNING_ALGO_COUNT` | Breadth of algorithm search for TE HipBLASLt tuning (see `examples/run_pretrain.sh` defaults). |
| `TE_HIPBLASLT_TUNING_RUN_COUNT` | Number of benchmark runs per shape during TE tuning. |
| `TE_HIPBLASLT_TUNING_ALGO_FILE` | Optional algorithm file for TE tuning flows. |
| `TE_HIPBLASLT_TUNING` | When set, interacts with deterministic mode; avoid conflicting settings with shape dump (see script comments in `examples/run_pretrain.sh`). |
| `HIPBLASLT_TUNING_OVERRIDE_FILE` | Override file for stage 3 training. |

### Standalone offline tool

For manual HipBLASLt bench workflows, see `examples/offline_tune/offline_tune_gemm.py` and `examples/offline_tune/README.md` (hipblaslt-bench integration and `HIPBLASLT_TUNING_OVERRIDE_FILE` usage).

---

## 2. Primus-Turbo optimization

**Primus-Turbo** is a separate package of optimized AMD GPU kernels used by Primus Megatron and TorchTitan integrations. It is controlled by the master flag `enable_primus_turbo` in Megatron configs (`primus/configs/modules/megatron/primus_turbo.yaml` extends into trainer/model as needed). **You must install the external `primus_turbo` package** for these paths to be available.

### Master flag (Megatron)

```yaml
enable_primus_turbo: true
```

### Feature flags (Megatron)

Defaults in `primus/configs/modules/megatron/primus_turbo.yaml` are mostly `false` until enabled.

| Flag | Purpose |
|------|---------|
| `use_turbo_attention` | Optimized attention kernels. |
| `use_turbo_parallel_linear` | Optimized tensor-parallel linear layers. |
| `use_turbo_grouped_gemm` | Optimized grouped GEMM for MoE. |
| `use_turbo_grouped_mlp` | Removed—use `use_turbo_grouped_gemm` (passing this key now raises an error). |
| `use_turbo_rms_norm` | Optimized RMSNorm. |
| `moe_use_fused_router_with_aux_score` | Fused MoE router (requires Primus-Turbo backend; see [Backend Patch Notes](../06-developer-guide/backend-patch-notes.md)). |
| `use_turbo_deepep` | DeepEP token dispatcher; set with `enable_primus_turbo: true`. |
| `turbo_deepep_num_cu` | Compute units for DeepEP (patch notes suggest practices such as 64 or 80 for EP8, 32 for EP16–64). |
| `turbo_sync_free_moe_stage` | Sync-free MoE stages (`0`–`3`; `0` disables, stage `2` recommended for performance per patch notes). See [MoE training deep-dive](./moe-training.md). |
| `use_turbo_fused_act_with_probs` | Fused activation with probabilities to reduce redundant work. |

### Feature flags (TorchTitan)

TorchTitan presets include `primus_turbo` in `primus/configs/modules/torchtitan/pre_trainer.yaml`

Example keys:

```yaml
primus_turbo:
  enable_primus_turbo: true
  use_turbo_attention: true
  use_turbo_async_tp: true
  use_turbo_float8_linear: true
  use_turbo_grouped_mm: false
```

### Documentation

Extended Megatron arguments and Turbo-related behavior are summarized in [Backend Patch Notes](../06-developer-guide/backend-patch-notes.md).

---

## 3. Mixed precision training

### Megatron (`trainer_base.yaml` patterns)

| Setting | Description |
|---------|-------------|
| `bf16: true` | BFloat16 training (default `true` in `trainer_base.yaml`). |
| `fp16: false` | FP16 training (optional). |
| `fp8` | FP8 recipe control (`null` / recipes such as delayed scaling in upstream Megatron). |
| `fp8_recipe`, `fp8_margin`, `fp8_interval` | FP8 scaling behavior. |
| `fp4`, `fp4_recipe` | Experimental FP4 paths. |
| `first_last_layers_bf16: true` | Keep first/last layers in BF16 for stability (`num_layers_at_start_in_bf16`, `num_layers_at_end_in_bf16` fine-tune). |

### TorchTitan

| Setting | Location |
|---------|----------|
| `training.mixed_precision_param: bfloat16` | `pre_trainer.yaml` default |
| `training.mixed_precision_reduce: float32` | Reduce precision |
| FP8 / quantization | `quantize.linear.float8.*` in `primus/configs/modules/torchtitan/quantize.yaml` |

### Loss fusion (Megatron model)

From `primus/configs/models/megatron/language_model.yaml`:

- Default: `cross_entropy_loss_fusion: false` with `cross_entropy_fusion_impl: "native"`.
- To use Transformer Engine fused cross entropy where supported, enable the fusion explicitly and set `cross_entropy_fusion_impl: "te"`.

---

## 4. Activation recomputation

### Megatron

| Parameter | Typical values | Notes |
|-----------|----------------|-------|
| `recompute_granularity` | `full`, `selective` | `full` recomputes more; max memory savings. |
| `recompute_method` | `uniform`, `block` | How recomputation is distributed. |
| `recompute_num_layers` | integer | Layers to recompute when using selective or uniform strategies. |
| `recompute_layer_ids` | list or null | Primus extension: **global** layer indices (the patch resolves block-local indices to global ids via `layer_offset`). Decoder layers are `0` to `num_layers - 1`; the MTP depths continue the numbering, so depth *d* is `num_layers + d`. Use with `recompute_granularity: full` and `recompute_method: null`. |

### TorchTitan

| Parameter | Location |
|-----------|----------|
| `activation_checkpoint.mode` | `none` in default `pre_trainer.yaml` |
| `activation_checkpoint.selective_ac_option` | Selective AC options |

---

## 5. Communication overlap

### Megatron

From `trainer_base.yaml` and model settings:

| Setting | Purpose |
|---------|---------|
| `overlap_grad_reduce` | Overlap gradient reduction with backward. |
| `overlap_param_gather` | Overlap parameter gather with forward. |
| `overlap_p2p_comm` | Pipeline P2P overlap. |
| `async_tensor_model_parallel_allreduce` | Async TP all-reduce (model config). |

### TorchTitan

| Setting | Purpose |
|---------|---------|
| `parallelism.enable_async_tensor_parallel: true` | Async tensor parallelism. |

### Environment

`CUDA_DEVICE_MAX_CONNECTIONS=1` is commonly required for **correct** overlap behavior in TP/PP stacks (see `docs/03-configuration-reference/environment-variables.md` and Megatron tests). Primus launch scripts or your cluster setup might set this.

---

## 6. Memory optimization

### Megatron

| Parameter | Purpose |
|-----------|---------|
| `optimizer_cpu_offload: true` | Offload optimizer state to CPU. |
| `optimizer_offload_fraction` | Fraction to offload (`1.0` in `trainer_base.yaml`). |
| `use_distributed_optimizer: true` | Shards optimizer state across DP ranks (when enabled). |
| `empty_unused_memory_level` | Aggressive emptying of unused memory (`0` default). |
| `global_batch_size` + `micro_batch_size` | Increase global batch via **gradient accumulation** without increasing per-step activation memory. |

### TorchTitan

| Parameter | Purpose |
|-----------|---------|
| `training.enable_cpu_offload` | CPU offload path in `pre_trainer.yaml` |

---

## 7. MoE-specific optimization

### Megatron (model + turbo)

| Setting | Purpose |
|---------|---------|
| `moe_permute_fusion: true` | Fuse permutation / unpermutation (`patch-notes.md`). |
| `moe_use_fused_router_with_aux_score: true` | Fused router + aux loss (Primus-Turbo). |
| `use_turbo_deepep: true` | DeepEP dispatcher (`enable_primus_turbo` must be true). |
| `turbo_sync_free_moe_stage: 2` | Recommended stage for sync-free MoE (per patch notes). |
| `overlap_moe_expert_parallel_comm: true` | Overlap expert parallel communication (`trainer_base.yaml`). |

---

## 8. Known ROCm correctness workarounds

The AMD Triton backend can emit a `buffer_store_dwordx4` whose data registers are redefined by a
later instruction with no `s_waitcnt vmcnt` in between, so the store writes whatever the
clobbering instruction left behind. Nothing faults and nothing warns; a few percent of the
kernel's output elements simply hold garbage. Inductor's SwiGLU backward fusion trips it and
poisons the MLP weight gradients, which diverges TorchTitan training. Megatron is exposed to the
same kernels, since its fused activations are inductor-generated as well (`megatron/core/jit.py`
sets `jit_fuser = torch.compile` on torch 2.2+).

`primus/core/patches/triton_bufops_war_patches.py` handles this automatically on ROCm: every
kernel compiles normally, and only those whose emitted AMDGCN contains the hazard are recompiled
with buffer ops disabled. It has no switch: the recompile is a correctness fix, and a kernel that
does not need one is left alone anyway.

Two things about the design are deliberate. It selects per kernel rather than setting
`AMDGCN_USE_BUFFER_OPS=0` globally, because the hazardous kernels lose nothing without buffer
addressing while the Primus-Turbo grouped-GEMM kernels spill without it and carry no hazard — the
global switch costs ~18% on MoE recipes for kernels that never needed fixing. And it decides from
the emitted machine code rather than from a list of affected architectures, so a newly shipped
architecture cannot silently fall out of coverage.

It also appends to `TORCH_COMPILE_CACHE_KEY_TAG`, because on an FX graph cache hit inductor never
calls `triton.compile`; without that, a cache filled before the patch existed would keep serving
hazardous binaries. Note that every Triton build tried so far still emits the pattern, so this is
not something an image upgrade removes.

---

## Quick checklist

1. **GEMMs:** run HipBLASLt stages 1–3 or use `offline_tune_gemm.py` for custom workflows.
2. **Kernels:** enable `primus_turbo` after installing `primus_turbo`; turn on attention/MoE flags as needed.
3. **Precision:** BF16 by default; add FP8/FP4 only with recipe testing.
4. **Memory:** recomputation + distributed optimizer + CPU offload + accumulation before buying more GPUs.
5. **MoE:** fusion + DeepEP + sync-free stages + EP comm overlap when supported.
