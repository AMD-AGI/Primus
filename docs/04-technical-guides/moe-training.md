# MoE training deep-dive

This guide covers Mixture-of-Experts (MoE) training in Primus on AMD Instinct GPUs: the bottlenecks unique to sparse models, the Primus/Primus-Turbo optimizations that address them, and a model-by-model tuning walkthrough. It is adapted from the AMD blog [MoE Training Best Practices on AMD GPU](https://rocm.blogs.amd.com/software-tools-optimization/primus-moe-package/README.html) (`examples/moe_package/README.md`) and grounded in the actual Primus configs and run scripts.

All flags shown here are the **real CLI/YAML keys** used by `examples/moe_package/run_*_pretrain_mi355x.sh` and the Megatron module configs (`primus/configs/modules/megatron/`). The Primus-Turbo MoE optimizations in this guide (DeepEP, sync-free MoE, Turbo grouped GEMM) are **Megatron-backend** features. TorchTitan also supports MoE via expert parallelism (`expert_parallel_degree`, `expert_tensor_parallel_degree`), but its tuning is out of scope here.

---

## 1. Why MoE training is different

MoE scales model capacity by routing each token through a small subset of "expert" sub-networks instead of activating the whole network. A gating/router picks the top-k experts per token, so a model can hold many billions of parameters while only a fraction are active per token.

This sparsity creates performance challenges that dense models do not have:

- **Grouped GEMM overhead**—each expert is a separate GEMM; naive multi-stream execution leaves scheduling gaps.
- **All-to-all (A2A) communication**—token dispatch/combine across expert-parallel ranks can dominate runtime, especially with `EP >= 8` and multi-node.
- **CPU sync & launch delays**—dynamic shapes (token counts per expert) force device-to-host syncs that stall the kernel launch queue.
- **Too many small kernels**—fine-grained MoE ops stress the CPU launch path.
- **Pipeline load imbalance**—uneven layer distribution across pipeline stages quietly degrades throughput.
- **Memory pressure**—activations dominate memory at large scale, forcing recomputation.

---

## 2. Representative model configs

Primus ships Megatron model presets for DeepSeek-style MoE models plus two ultra-large research configs:

| Model | Total / Active params | Model config (`primus/configs/models/megatron/`) |
|-------|-----------------------|--------------------------------------------------|
| DeepSeek-V2-Lite | 16B / 2.4B | `deepseek_v2_lite.yaml` |
| DeepSeek-V2 | 236B / 21B | `deepseek_v2.yaml` |
| DeepSeek-V3 | 671B / 37B | `deepseek_v3.yaml` |
| MoE-1T | 1T / 44B | `moe_1T.yaml` |
| MoE-2T | 2T / 80B | `moe_2T.yaml` |

Ready-to-run pretrain scripts live in `examples/moe_package/`, e.g.:

- `examples/moe_package/run_deepseek_v2_lite_pretrain_mi355x.sh`
- `examples/moe_package/run_deepseek_v2_pretrain_mi355x.sh`
- `examples/moe_package/run_deepseek_v3_pretrain_mi355x.sh`

Each example script is a convenience wrapper: it sets environment + parallelism and selects an experiment YAML under `examples/moe_package/configs/`, then launches training. You can run the same training directly with the unified CLI, passing the experiment YAML with `--config` and the MoE feature toggles (Section 4) as overrides:

```bash
# DeepSeek-V2-Lite baseline + DeepEP + sync-free + loss fusion + manual GC, via primus-cli
export ENABLE_NUMA_BINDING=1 HSA_KERNARG_POOL_SIZE=12582912   # feature 6 (env, not CLI flags)
./runner/primus-cli direct -- train pretrain \
  --config examples/moe_package/configs/MI355X/deepseek_v2_lite-pretrain-baseline.yaml \
  --enable_primus_turbo True \
  --use_turbo_deepep True --turbo_deepep_num_cu 64 --moe_router_dtype fp32 \
  --turbo_sync_free_moe_stage 1 \
  --cross_entropy_fusion_impl te --cross_entropy_loss_fusion True \
  --manual_gc True --manual_gc_interval 1
```

Use `./runner/primus-cli slurm srun -N <nodes> -- train pretrain --config ...` for multi-node. The feature tables below list the exact flags so you can compose your own command. (Optimizations that are environment variables—NUMA binding, `HSA_KERNARG_POOL_SIZE`, UCCL-EP—are exported before the command rather than passed as `--flags`.)

---

## 3. Profiling and analysis workflow

Diagnose before optimizing. The recommended order:

1. **Torch Profiler**—capture operator times, memory, and GPU utilization. Enable through the Megatron profiling flags (`--profile`, `--use_pytorch_profiler`, `--profile_step_start`, `--profile_step_end`, `--disable_profiler_activity_cpu`). Load the trace in [Perfetto](https://ui.perfetto.dev/) to inspect CPU/GPU overlap, launch delays, and idle gaps. See [Profiling & observability](./profiling-and-observability.md).
2. **TraceLens**—AMD's automated trace analyzer for hierarchical breakdowns, roofline/efficiency, communication-vs-sync separation, and trace diffing. Wired into Primus via `generate_tracelens_report` / `mlflow_upload_tracelens_report` (see the profiling guide).
3. **Memory projection**—model VRAM across params, gradients, activations, and optimizer state *before* launching, via `./primus-cli direct -- projection memory --config <exp>.yaml`. See [Projection](../02-user-guide/projection.md).
4. **Pipeline visualization**—dump pipeline schedule data (`--dump_pp_data true`) and render stage utilization with `tools/visualization/pp_vis/vis.py` to find bubbles and stage imbalance.

---

## 4. Primus MoE optimizations

The `examples/moe_package/run_*` scripts expose these as composable "MoE features." The table maps each feature to the **actual `--flags` (or environment variables)** you pass to `train pretrain`—the same toggles the example scripts set.

| Feature | Flags (real keys) | What it does |
|---------|-------------------|--------------|
| Turbo attention | `--enable_primus_turbo True --use_turbo_attention True` | Optimized attention kernels (Primus-Turbo). |
| Turbo grouped GEMM | `--enable_primus_turbo True --use_turbo_grouped_gemm True` | Fused CK grouped GEMM processes all experts in one launch instead of multi-stream. |
| Loss fusion | `--cross_entropy_fusion_impl te --cross_entropy_loss_fusion True` | Fuses large-vocab loss into one kernel to cut memory + launch overhead. |
| DeepEP acceleration | `--enable_primus_turbo True --use_turbo_deepep True --turbo_deepep_num_cu 64 --turbo_deepep_use_comm_stream False --moe_shared_expert_overlap False --moe_router_dtype fp32` | GPU-side index calc + sync-free dispatch to cut redundant cross-node A2A traffic. |
| Sync-free MoE | `--enable_primus_turbo True --turbo_sync_free_moe_stage <N>` | Removes CPU D2H syncs across Router → Dispatcher → Permutation → GroupMLP. |
| NUMA binding | `export ENABLE_NUMA_BINDING=1` | Pins each GPU process to its NUMA socket for better memory bandwidth/stability. |
| HIP kernarg pool | `export HSA_KERNARG_POOL_SIZE=12582912` | Enlarges the kernel-argument pool (12 MB) to avoid launch stalls under many small kernels. |
| Manual GC | `--manual_gc True --manual_gc_interval 1` | Periodic host GC to remove iteration-time jitter on long runs. |
| UCCL-EP | `export USING_UEP=1` | Use the UCCL transport for DeepEP dispatch/combine (sets `PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND=DEEP_EP` + UCCL network env). Requires the `uccl` and `deep_ep` packages. |

> **Grouped GEMM keys.** Enable Turbo grouped GEMM for MoE with `use_turbo_grouped_gemm` (`--use_turbo_grouped_gemm True`). The older `use_turbo_grouped_mlp` alias has been **removed**—passing it now raises an assertion error (`use_turbo_grouped_mlp has been removed; please use use_turbo_grouped_gemm instead`).
>
> **Legacy path.** The legacy multi-stream grouped GEMM path is selected with `--moe_use_legacy_grouped_gemm True` (the scripts default `LEGACY_GG=True`). Turbo grouped GEMM is **incompatible** with the legacy path—set `--moe_use_legacy_grouped_gemm False` whenever `use_turbo_grouped_gemm` is enabled (Primus raises an error otherwise).

### Sync-free MoE stages

`turbo_sync_free_moe_stage` is a single knob with four levels (`0`–`3`, validated in `primus/backends/megatron/patches/args/rocm_arg_validation.py`). Each stage **auto-enables** a set of fusion flags:

| Level | Auto-enabled flags | Behavior |
|-------|--------------------|----------|
| `0` (default) | — | Disabled—standard baseline. |
| `1` | `moe_use_fused_router_with_aux_score`, `moe_permute_fusion` | Sync-free **Router** + **Permutation** fusion. |
| `2` | stage 1 + `use_turbo_deepep`, `use_turbo_grouped_gemm` | Adds sync-free **DeepEP** dispatch and **Turbo grouped GEMM**. |
| `3` | stage 2 + `use_turbo_fused_act_with_probs` | Full sync-free pipeline (adds fused activation). Per the blog this **uses significantly more GPU memory**—only enable with headroom. |

Requirements (enforced at startup):

- All stages require `--enable_primus_turbo True`.
- Stages `2` and `3` require Turbo grouped GEMM and are therefore **incompatible with `--moe_use_legacy_grouped_gemm True`**.

Practical guidance from the run scripts:

- **MI355X:** `--turbo_sync_free_moe_stage 1` (compatible with the default legacy grouped GEMM path, since stage 1 does not enable Turbo grouped GEMM).
- **MI300X / MI325X:** the example scripts suggest stage `2` (with `--moe_shared_expert_overlap False --moe_router_dtype fp32`). Because stage 2 auto-enables Turbo grouped GEMM, you must also set `--moe_use_legacy_grouped_gemm False`.

### Scheduling and memory features

- **1F1B A2A overlap**—interleaves micro-batch N's expert communication with micro-batch N-1's backward compute on top of interleaved-1F1B pipeline parallelism, hiding A2A behind compute while preserving the bubble rate and roughly the same peak memory.
- **Arbitrary pipeline partition**—manual stage layout instead of automatic even splits, to balance per-stage memory/compute. Use the Megatron-core `--pipeline_model_parallel_layout` flag (as the DeepSeek-V3 script does) or the Primus `decoder_pipeline_manual_split_list` config key (`primus/configs/modules/megatron/primus_megatron_module.yaml`).
- **Selective layer recompute**—recompute specific transformer layers with `--recompute_layer_ids 0,1,2,3` (keep `RECOMPUTE_LAYERS=0` so this is the only recompute control), or full block recompute via `--recompute_granularity full --recompute_method block --recompute_num_layers N`.
- **MoE expert-parallel comm overlap**—`overlap_moe_expert_parallel_comm: true` (`trainer_base.yaml`).

See [Performance tuning](./performance-tuning.md) for the full Primus-Turbo flag reference.

---

## 5. Model-specific tuning

### DeepSeek-V2-Lite (16B / 2.4B, 27 layers)

A compute/memory-efficient variant ideal for high-throughput pretraining. AMD Instinct's large HBM (192 GB on MI300X, 288 GB on MI355X) lets you push **micro-batch size (MBS)** high to maximize throughput.

Recommended optimization stack (matches `run_deepseek_v2_lite_pretrain_mi355x.sh`, where `MoE_Features=(3 4 5 6 7 8)`):

1. Manual GC for stable iteration time.
2. Loss fusion for the large vocabulary.
3. DeepEP for A2A.
4. Sync-free mode (stage 1 on MI355X) to remove D2H syncs.
5. NUMA binding for CPU affinity (`ENABLE_NUMA_BINDING=1` + `HSA_KERNARG_POOL_SIZE`).
6. MBS scaling using the memory freed by the above (the blog reports peak memory dropping from ~99.8% to ~84.3% at MBS=12, enabling MBS=14).

The script's default `MoE_Features=(3 4 5 6 7 8)` also enables feature `8` = UCCL-EP (see the feature table above). Default parallelism in the script: `TP=1 ETP=1 PP=1 EP=8 CP=1`, `MBS=14 GBS=896 SEQ=4096`.

### DeepSeek-V2 (236B / 21B, 60 layers)

Scale up with parallelism for max throughput across nodes. Recommended stack:

1. Manual GC, 2) Loss fusion, 3) DeepEP, 4) NUMA binding, 5) Sync-free mode, plus **interleaved pipeline parallelism (VPP)** to cut the pipeline bubble ratio. Enabling VPP (`--num_virtual_stages_per_pipeline_rank > 1`) also improves sync-free mode effectiveness.

Default parallelism in the script: `TP=1 ETP=1 PP=4 VPP=5 EP=8 CP=1` (interleaved PP), `SEQ=4096`.

### 1T+ parameter models (MoE-1T / MoE-2T, 96 layers)

Ultra-large training combines every advanced technique. Use **memory projection first**—at this scale **activations dominate memory**, not parameters/optimizer state.

Findings from the blog's projections (768–1024 GPUs):

- **Context parallelism (CP2)** roughly halves activation memory (~76 GB/GPU saved for 1T, ~131 GB/GPU for 2T)—the most effective single lever.
- **Increasing EP** (8→16) barely reduces memory but adds A2A time.
- **Increasing PP** (24→48) doesn't materially cut memory and raises pipeline bubbles + activation memory.

Suggested configs:

| Model | MI300X | MI355X |
|-------|--------|--------|
| MoE-1T | PP24 EP8 CP2 | PP24 EP8 (no checkpointing) |
| MoE-2T | PP24 EP16 CP2 | PP24 EP8 CP2 (benefits from larger DP) |

**Pipeline bubble at scale.** When the global batch is constrained, gradient-accumulation (GA) per iteration drops and the bubble ratio rises. Interleaved PP (VPP) mitigates this:

$$
\text{bubble ratio} = \frac{PP-1}{(PP-1) + GA \times VPP}
$$

For PP=16, GA=16: VPP=1 gives ~48% bubble; VPP=6 gives ~14%—a large efficiency win verified on a 64-node setup.

**Inter-node dispatch.** Profiling on a 2T/1024-GPU run showed A2A consuming **25–30%** of step time; DeepEP delivered roughly **1.05×–7.66×** end-to-end speedup over plain A2A and kept EP scaling nearly flat.

---

## 6. Quick checklist

1. **Profile first**—Torch Profiler + TraceLens; project memory before launching ultra-large runs.
2. **Turn on Turbo**—`--enable_primus_turbo True`, then grouped GEMM, attention, DeepEP as needed (requires the external `primus_turbo` package).
3. **Kill CPU syncs**—`--turbo_sync_free_moe_stage` (1 on MI355X, 2 on MI300X/MI325X).
4. **Stabilize + bind**—`--manual_gc True`, `ENABLE_NUMA_BINDING=1`, `HSA_KERNARG_POOL_SIZE=12582912`.
5. **Scale memory headroom into throughput**—raise MBS; use CP2 + selective recompute for 1T+; use VPP to cut pipeline bubbles.

---

## Related documentation

- [Performance tuning](./performance-tuning.md)—Primus-Turbo flags, HipBLASLt, precision, recompute.
- [Parallelism strategies](./parallelism-strategies.md) and [Parallelism configuration](./parallelism-configuration.md)—EP, PP, CP, VPP.
- [Collective operations](./collective-operations.md)—A2A and DeepEP context.
- [Profiling & observability](./profiling-and-observability.md) and [Projection](../02-user-guide/projection.md).
- Source blog: `examples/moe_package/README.md`.
