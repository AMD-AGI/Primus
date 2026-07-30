<!---
Copyright (c) 2025 Advanced Micro Devices, Inc. (AMD)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
--->

<!---
NOTE: This is a DRAFT of the MoE 2.0 blog. Each section carries an HTML-comment
"OWNER / FILL" note describing what the section owner still needs to supply
(final numbers, figures, benchmark tables). The prose is a starting point —
owners should verify all technical claims and replace every `[TODO: ...]` /
`[X]` placeholder and every image path under `imgs/` with the real asset before
publishing. All OWNER / FILL comments and placeholders must be removed before
release.

STRUCTURE: the post runs bottom-up through the training stack —
  foundation (framework) -> kernels -> scale (TTT) -> other regimes and backends -> tooling.
Every later result is measured on top of the framework baseline in the first
section, so that section should stay first. The megakernel is research-stage and
is explicitly labelled as a preview; keep that label until its numbers clear
review, so its placeholders do not sit unmarked next to measured results.
--->

# MoE Training Optimization with Primus

_Mixture-of-Experts (MoE) is now the default architecture for frontier-scale language models, and our users increasingly train large MoE models on AMD Instinct™ GPUs. This post walks through the MoE training optimizations we built in Primus in response to those workloads, working bottom-up through the stack: the general **Primus + Primus-Turbo** framework optimizations every MoE run inherits, kernel-level work (**low-precision expert GEMMs** and a fused MoE **megakernel**), **time-to-train** case studies spanning DeepSeek-V3 at up to 1024 GPUs and the GPT-OSS-20B MLPerf workflow, the different bottleneck regime of **small MoE models** benchmarked against **NVIDIA B200**, our **JAX (MaxText)** training path, and the **projection** tool that sizes a run before it starts. Both the Megatron-LM and JAX backends of Primus are covered. For the foundational optimizations this work builds on, see our earlier [MoE Training Best Practices on AMD GPU](https://rocm.blogs.amd.com/software-tools-optimization/primus-moe-package/README.html) post._

All feature demonstrations and benchmarking results in this guide are built on Primus. [Primus/Primus-LM](https://github.com/AMD-AGI/Primus) is a flexible, high-performance framework for large-scale foundation model training and inference on AMD GPUs. As the training-framework layer of the Primus ecosystem, Primus-LM works alongside [Primus-Turbo](https://github.com/AMD-AGI/Primus-Turbo) (high-performance operators) and [Primus-SaFE](https://github.com/AMD-AGI/Primus-SaFE) (stability and platform infrastructure) to deliver a scalable, production-ready solution for state-of-the-art large-model development.

---

## Background

The open-source MoE landscape has both broadened and deepened: models are larger and sparser, expert granularity is finer, and MoE has become the default architecture for frontier-scale language models. Combined with growing demand from real training workloads, this makes MoE training efficiency a first-order concern.

### How MoE models are changing

<!---
OWNER / FILL: TBD (Background owner). Verify the representative model list and the
architectural trend claims below; add citations/links where useful. Consider a
small table of the MoE models Primus now supports (DeepSeek-V3, Qwen3-235B-A22B,
Qwen3-30B-A3B, GLM, MiniMax, GPT-OSS-120B, Mixtral) with total/active params.
--->

Modern MoE models share a set of clear architectural trends:

- **Scaling total parameters while keeping activation sparse.** Total parameter counts continue to climb toward the trillion-parameter regime, while only a small fraction of parameters are activated per token. This decouples model capacity from per-token compute, but shifts the training bottleneck toward memory, routing, and communication.
- **Fine-grained experts.** Newer models use many small experts rather than a few large ones (for example, DeepSeek-V3's 256 routed experts with top-8 routing), increasing expert count and the number of fine-grained operators per MoE layer.
- **Shared plus routed experts.** A shared "always-on" expert combined with sparsely routed experts has become common, adding structure to the MoE layer that optimizations must respect.
- **Higher top-k and larger EP.** More experts activated per token and larger expert-parallel (EP) groups increase all-to-all traffic, making dispatch/combine communication a first-order cost.

Representative models in this generation include DeepSeek-V3, Qwen3-235B-A22B and Qwen3-30B-A3B, GLM, MiniMax, GPT-OSS, and Mixtral — all of which Primus supports today.

### Where the time goes, and what this post covers

Those trends mean MoE training efficiency is no longer about fast GEMMs alone. A modern MoE step is limited by several different things at once, and an optimization that attacks only one of them tends to expose the next. The table below maps each bottleneck to the work described in this post, so you can jump straight to what is limiting your own runs.

| Bottleneck | What we did about it | Section |
|---|---|---|
| **Expert GEMM throughput** — expert FFNs dominate FLOPs | FP8/MXFP8 grouped GEMM, FlyDSL kernels, quantized-weight caching | [Low-precision expert GEMMs](#low-precision-expert-gemms) |
| **All-to-all dispatch/combine** — grows with top-k and EP width | DeepEP dispatch, 1F1B all-to-all overlap, and tile-granularity fusion inside a single kernel | [Framework foundation](#the-foundation-primus--primus-turbo), [Megakernel](#research-preview-the-moe-megakernel) |
| **Memory ceiling** — sets micro-batch size and recompute cost | Precision-aware optimizer, pipeline layout, fine-grained recompute, ahead-of-time projection | [DeepSeek-V3 at scale](#time-to-train-deepseek-v3-at-256-1024-gpus), [Projection](#planning-before-you-train-primus-projection) |
| **End-to-end time-to-quality** — throughput alone does not determine TTT | Topology correction, lower data movement, full-path fusion, warm-up, and convergence-aware batch sizing | [GPT-OSS-20B time-to-train](#time-to-train-gpt-oss-20b) |
| **Host and launch overhead** — dominates when layers are small | Sync-free MoE, NUMA binding and launch tuning, pipeline warm-up | [Framework foundation](#the-foundation-primus--primus-turbo), [Small MoE models](#small-moe-models-a-different-bottleneck-regime) |

The [JAX (MaxText) path](#beyond-megatron-lm-the-jax-maxtext-path) sits outside that bottleneck grid, bringing the same grouped-GEMM and DeepEP primitives to a second backend.

### Results at a glance

<!---
OWNER / FILL: Overview owner. Replace [X] placeholders with the real headline
numbers once the per-model results below are finalized, and make sure this
paragraph matches the "Overall performance uplift" chart. Keep it to one
paragraph of end-to-end story + headline numbers. Add the asset at
imgs/moe_perf_overview.png — a bar chart of end-to-end training throughput uplift
across common MoE models (DeepSeek-V3, Qwen3-235B-A22B, Qwen3-30B-A3B, GPT-OSS,
Mixtral, ...). State the baseline, hardware (MI300 / MI355), and precision in the
caption.
--->

Taken together, these optimizations deliver end-to-end training speedups across the modern MoE model family on AMD Instinct MI300/MI355-series GPUs. Against an unoptimized baseline configuration, the combined kernel, communication, precision, and scheduling improvements yield up to **[X]×** higher training throughput on representative models such as DeepSeek-V3 and Qwen3-235B-A22B, while projection keeps configuration cost low by predicting memory and performance before a run starts.

![Figure 1: End-to-end MoE training throughput uplift across representative models on AMD Instinct MI355X](imgs/moe_perf_overview.png)

**Figure 1: End-to-end MoE training throughput uplift across representative models on AMD Instinct MI355X** _(placeholder — asset and numbers to be finalized)_

---

## The Foundation: Primus + Primus-Turbo

<!---
OWNER / FILL: Ruibin Zhang. Verify the list of general optimizations and add any
missing recent ones. The first blog already documents Turbo Grouped GEMM,
DeepEP, Sync-Free MoE, 1F1B A2A overlap, arbitrary pipeline partition, selective
recompute, loss fusion, CPU launch optimization, and manual GC — reference it
rather than repeating in full.
--->

Every result later in this post is measured on top of a common foundation, so it belongs first. Most MoE workloads benefit before any model-specific tuning from a set of general-purpose optimizations delivered through Primus and [Primus-Turbo](https://github.com/AMD-AGI/Primus-Turbo): Turbo grouped GEMM, DeepEP-accelerated dispatch, sync-free MoE, 1F1B all-to-all overlap, arbitrary pipeline partitioning, selective-layer recompute, loss fusion, NUMA binding and kernel-launch tuning, and manual GC. These are covered in detail in our earlier [MoE blog](https://rocm.blogs.amd.com/software-tools-optimization/primus-moe-package/README.html) and are default-on in every recipe here.

On top of that foundation, we hardened and extended the following for the current model generation.

**Performance**

- **Quantized-weight caching.** Caching quantized weights across multiple microbatches reduces quantization overhead in FP4 and FP8 training.
- **FlyDSL GEMM and grouped GEMM kernels.** Support for FlyDSL FP8 GEMM and grouped GEMM kernels improves performance over Triton implementations.
- **Single-parameter grouped linear.** Merging multiple experts' weights into a single contiguous tensor and applying a grouped quantization kernel reduces quantization overhead.
- **Precision-aware optimizer with BF16 states.** Storing master gradients and the Adam moment estimates in BF16 (`main_grads_dtype`, `exp_avg_dtype`, `exp_avg_sq_dtype`) meaningfully reduces optimizer memory and gradient-reduction cost, freeing memory headroom for larger micro-batches.
- **Fused cross-entropy and RoPE.** TE-backend cross-entropy loss fusion (`cross_entropy_fusion_impl: te`) and fused RoPE (`apply_rope_fusion`) cut memory and kernel overhead in the loss and attention paths.

**Usability**

- **Pipeline warm-up (`pp_warmup`).** A parallel forward+backward warm-up on every pipeline rank exercises all lazy-init paths (CUDA/HIP, TE, FP8, NCCL) concurrently, removing first-iteration stalls without changing numerics.
- **Faster process teardown.** An opt-in fast-exit path shaves wall-clock time from the tail of large runs.

![Figure 2: Additional Primus-Turbo optimizations improve throughput by 16.2% on Qwen3-235B-A22B, 9.7% on GPT-OSS 20B, and 5.8% on Qwen3-30B-A3B](imgs/general_opt_uplift.png)

**Figure 2: Incremental throughput uplift from additional Primus-Turbo optimizations.** Compared with the listed reference configurations, the additional Turbo and FlyDSL settings improve throughput by **16.2%** on Qwen3-235B-A22B, **9.7%** on GPT-OSS 20B, and **5.8%** on Qwen3-30B-A3B.

The specific model configurations and measured throughput are shown below.

| Model | GPUs | Precision | Parallelism (TP/PP/CP/EP/DP) | MBS | GBS | Seq Len | Key Config / Flags | Throughput (tokens/s) |
|---|---|---|---|---|---|---|---|---|
| Qwen3-235B-A22B (best configuration) | 32 | FP8-CS | 1/1/4/8/1 | 2 | 1024| 4096 | `turbo_sync_free_moe_stage: 1` | 4137.5 |
| Qwen3-235B-A22B (Turbo-accelerated) | 32 | FP8-CS | 1/1/4/8/1 | 2 | 1024| 4096 | `use_turbo_grouped_gemm: true`<br>`turbo_sync_free_moe_stage: 2`<br>`PRIMUS_TURBO_GROUPED_GEMM_BACKEND=flydsl` | 4809.1 |
| GPT-OSS 20B (MLPerf configuration) | 8 | FP8-CS | 1/1/1/1/8 | 2 | 16 | 4096 | `use_turbo_grouped_gemm: true`<br>`use_turbo_fused_act_with_probs: true`<br>`use_turbo_rms_norm: true` | 25660.1 |
| GPT-OSS 20B (Turbo-accelerated) | 8 | FP8-CS | 1/1/1/1/8 | 2 | 16 | 4096 | `use_turbo_grouped_gemm: true`<br>`use_turbo_fused_act_with_probs: true`<br>`use_turbo_rms_norm: true`<br>`PRIMUS_TURBO_GROUPED_GEMM_BACKEND=flydsl` | 28136.7 |
| Qwen3-30B-A3B (best configuration) | 8 | FP8-CS | 1/1/1/8/1 | 8 | 512 | 4096 | `turbo_sync_free_moe_stage: 1` | 26058.7 |
| Qwen3-30B-A3B (Turbo-accelerated) | 8 | FP8-CS | 1/1/1/8/1 | 8 | 512 | 4096 | `turbo_sync_free_moe_stage: 2`<br>`use_turbo_gemm: true`<br>`use_turbo_grouped_gemm: true`<br>`PRIMUS_TURBO_GROUPED_GEMM_BACKEND=flydsl` | 27581.3 |

---

## Kernel-Level Optimization

With the framework baseline in place, the next lever is the expert GEMM itself — first by making each GEMM cheaper (low precision), then by removing the kernel boundaries around it (the megakernel).

### Low-precision expert GEMMs

<!---
OWNER / FILL: Ruibin Zhang, Kyle Zhao. Confirm which FP8 recipes/results are
public. Add: (a) an accuracy/convergence statement for FP8 MoE training, (b) a
kernel-level FP8-vs-BF16 grouped-GEMM speedup figure, and (c) any end-to-end FP8
result. Keep the system-level break-even discussion (amax reduction, cast cost)
honest but framed around how Primus mitigates it.
--->

Low precision is the most direct lever on MoE training throughput, because expert GEMMs dominate compute. Primus supports FP8 training through Transformer Engine recipes with a Primus-Turbo operator overlay, covering **delayed**, **tensorwise (current) scaling**, **blockwise scaling**, and **MXFP8** recipes. For MoE specifically, the routed-expert path uses an FP8 **grouped GEMM** (`grouped_gemm_fp8`) with per-first-microbatch weight quantization and caching, and pads permuted tokens to the quantization block boundary so that dispatch, permutation, and expert GEMM all agree on the FP8 layout. These paths run on Primus-Turbo kernels supporting both FP8 tensorwise scaling and MXFP8 block scaling (`E4M3`, block size 32, `E8M0` scale). The FP8 GEMM and grouped-GEMM kernels behind the results in this section are authored in [FlyDSL](https://github.com/ROCm/FlyDSL).

Four things matter when running FP8 MoE training on AMD GPUs:

- **Accuracy is preserved.** Both the FP8 and MXFP8 expert GEMMs are numerically validated (including uneven-`M` grouped cases); at the training level the default tensorwise (current-scaling) recipe tracks BF16 convergence, while MXFP8's per-block scaling further contains quantization error on the expert GEMMs.
- **Recipe choice matters.** Tensorwise (current) scaling and block/MX scaling behave differently in both accuracy and speed on MI355X; Primus exposes the recipe as a single knob (`fp8_recipe`) so users can select the right trade-off.
- **Format selection.** On gfx950, using the OCP E4M3 format for expert GEMMs avoids costly up-conversions.
- **System-level break-even.** At the kernel level, FP8 expert GEMMs are substantially faster than BF16. End-to-end, the win depends on amortizing the surrounding quantization work — amax reduction, casting, and token-count synchronization. Primus reduces this overhead through weight-quantization caching, quantization-aware padding, and keeping token counts on-GPU, so the kernel speedup translates into end-to-end gains.

Figure 3 isolates that kernel-level payoff on MI355X alone, sweeping tokens-per-expert `M` over training-relevant sizes and reporting the FP8 (tensorwise) grouped-GEMM speedup over BF16 with quantization included in the FP8 timing.

<p align="center">
  <img src="imgs/low_precision_fp8_vs_bf16_grouped.png" alt="FP8 vs BF16 grouped GEMM speedup on MI355X" width="45%">
</p>

**Figure 3: Kernel-level FP8-vs-BF16 grouped-GEMM speedup on AMD Instinct MI355X, swept over tokens-per-expert `M` and averaged over the DeepSeek-V3 / Qwen3-235B-A22B / gpt-oss expert shapes; quantization is included in the FP8 timing.**

The forward speedup grows with `M` — from ~1.2x at `M`=2048 to ~1.6x at `M`=8192, as the GEMM gets large enough to hide the cast/amax cost that Primus already minimizes — while the backward is consistently ~1.5–1.7x. At training-relevant token counts the FP8 grouped-GEMM speedup is real and sizable.

To place these kernels against the industry reference, we benchmark end-to-end quantized dense and grouped GEMMs (quantization included in timing, correctness checked by output/gradient SNR) against the NVIDIA B200/GB200 TransformerEngine baseline, on representative Llama-style dense shapes and DeepSeek-V3 / Qwen3-235B-A22B / Kimi-K2 expert shapes. Across these shapes MI355X is broadly at parity: dense GEMM is essentially even in both precisions, and grouped GEMM leads on the forward pass (driven by the memory-bound `Down` projection) while backward is close to parity, with large-`N` `GateUP` weight-gradient the main remaining gap.

<table>
  <tr>
    <td><img src="imgs/low_precision_tensorwise_dense_b200.png" alt="FP8 tensorwise dense GEMM performance vs B200/GB200" width="100%"></td>
    <td><img src="imgs/low_precision_tensorwise_grouped_b200.png" alt="FP8 tensorwise grouped GEMM performance vs B200/GB200" width="100%"></td>
  </tr>
  <tr>
    <td><img src="imgs/low_precision_mxfp8_dense_b200.png" alt="MXFP8 dense GEMM performance vs B200/GB200" width="100%"></td>
    <td><img src="imgs/low_precision_mxfp8_grouped_b200.png" alt="MXFP8 grouped GEMM performance vs B200/GB200" width="100%"></td>
  </tr>
</table>

**Figure 4: FP8 tensorwise (top) and MXFP8 (bottom) GEMM throughput on AMD Instinct MI355X vs NVIDIA B200/GB200 (TE) — dense (left) and grouped (right), forward and backward, averaged over the tested shapes.**

The lesson is that low precision is not just a datatype switch: for MoE, layout and grouped scheduling matter as much as the quantization recipe. Removing the forward transpose tax and autotuning each grouped shape turns an apparent forward deficit into a forward lead, and MXFP8 layers finer-grained scaling on top of the same execution path while staying competitive with the B200/GB200 baseline.

### Research preview: the MoE megakernel

<!---
OWNER / FILL: Xiaoming Peng, Zhen Huang. This section describes research-stage
work (RocMoE / MonolithEP super-kernel). Before publishing:
  - Confirm which numbers are cleared for public release. The standalone GEMM
    roofline (near-peak MFMA) is safe and compelling; end-to-end fused-kernel
    speedups are still being finalized at training scale — keep them as
    placeholders ([X]) or mark clearly as preliminary until validated.
  - Decide how much AMD-specific implementation detail (DTOLDS/AGPR/wave
    specialization) to expose publicly.
  - Add figures: (a) fused single-kernel dataflow diagram, (b) timeline showing
    dispatch/GEMM/combine overlap, (c) a perf bar vs. the separate-kernel baseline.
KEEP the "research preview" framing until the numbers clear review.
--->

> **Research preview.** Unlike everything above, this is research-stage work rather than a shipping feature. The standalone GEMM results are measured; the fused end-to-end numbers are preliminary.

**Motivation.** Faster expert GEMMs eventually run into a structural limit. In a standard MoE layer, expert dispatch and combine are collective all-to-all operations while FC1/FC2 are grouped GEMMs. Because collective libraries are host-initiated and operate at kernel granularity, communication and expert compute execute in separate kernels and can only be overlapped coarsely across kernel boundaries. Profiling shows the MoE forward pass split roughly between all-to-all communication and expert compute, so the biggest remaining prize is to overlap the two *inside* a single kernel at fine granularity.

**Design.** The megakernel fuses the entire MoE forward path — dispatch (all-to-all) → FC1 (gate/up grouped GEMM) → SwiGLU → FC2 (down grouped GEMM) → combine (all-to-all) — into a **single persistent kernel**:

- **Role-specialized workgroups.** The persistent grid is partitioned into roles (dispatch, compute, combine), so communication workgroups can make progress while compute workgroups run MFMA GEMMs concurrently on the same device.
- **Tile-granularity overlap via arrival scoreboards.** Instead of a global barrier between dispatch and compute, per-block arrival flags let a compute workgroup begin FC1 on a tile the moment that tile's tokens have landed — hiding communication latency inside the GEMM.
- **Zero-permute token layout.** Received tokens are packed contiguously per expert, so the grouped GEMM indexes them directly with no separate permutation step.
- **Epilogue fusion.** SwiGLU is fused into the FC1 epilogue and FC2 output is written directly into the combine path, eliminating intermediate activation round-trips to HBM.

**AMD-specific engineering.** The design maps the pattern onto CDNA3/CDNA4 (gfx942/gfx950): direct-to-LDS asynchronous loads replace TMA, MFMA accumulators live in AGPRs, `__hip_atomic_*` release/acquire plus LDS signaling replaces mbarrier/cluster synchronization, wave specialization within a workgroup replaces warpgroup register partitioning, and XGMI/IPC peer transfers replace NVLink.

**Results.** The hand-tuned expert grouped-GEMM inner loop reaches near-peak MFMA utilization on MI355X, approaching the BF16 roofline for representative DeepSeek-V3 expert shapes. The fused single-kernel prototype overlaps dispatch/compute/combine to reduce MoE-layer forward time versus a separate-kernel baseline.

<!---
OWNER / FILL: Xiaoming Peng, Zhen Huang. Replace [X] below with cleared numbers,
or reword to a qualitative claim if numbers are not yet public.
--->

On representative DeepSeek-V3 expert shapes, the fused megakernel achieves up to **[X]×** speedup over the separate dispatch/GEMM/combine baseline for the MoE forward layer. _(preliminary — to be finalized)_

![Figure 5: Fused MoE megakernel — single persistent kernel overlapping dispatch, grouped GEMM, and combine](imgs/megakernel_dataflow.png)

**Figure 5: Fused MoE megakernel — single persistent kernel overlapping dispatch, grouped GEMM, and combine** _(placeholder — asset to be added)_

We are working to graduate this super-kernel into a feature-flagged Primus-Turbo operator, extend it to FP8/MXFP8 expert weights, and scale it beyond a single node.

---

## Time-to-Train: DeepSeek-V3 at 256-1024 GPUs

Kernels and framework features are means to an end. What users actually measure is time-to-train: how long a full run takes end to end at real scale. This section is the flagship case study — DeepSeek-V3 on the Megatron-LM backend, where the binding constraint is not GEMM throughput but **memory**, and where the job is to spend that memory in the right places.

The reference recipe is [`examples/moe_package/run_deepseek_v3_pretrain_mi355x.sh`](https://github.com/AMD-AGI/Primus/blob/main/examples/moe_package/run_deepseek_v3_pretrain_mi355x.sh):

| Setting | Value |
|---|---|
| Model | 61 layers (3 dense + 58 MoE), 256 routed experts, top-8, MLA |
| Parallelism | TP1 / PP16 / VPP2 / EP8 |
| Batch | MBS 2, sequence length 4096, global batch size 128 x nodes |
| Precision | FP8 |
| Measured scales | 32 to 128 nodes (256 to 1024 MI355X GPUs) |

The recipe also enables DeepEP dispatch (`turbo_deepep_num_cu: 80`), the distributed optimizer with overlapped gradient reduce and parameter gather, and the precision-aware optimizer, fused cross-entropy/RoPE, `pp_warmup`, manual GC, and NUMA binding described earlier.

Two conventions apply to every number below:

- The global batch size scales with node count, so the micro-batch count per iteration stays at 128 at every scale and only the data-parallel width grows.
- Iteration times are comparable only within a scale. All throughput ratios are against the baseline at the *same* scale, measured over the first few dozen iterations.

### Step 1: put the short pipeline stage where memory peaks

PP16 with VPP2 gives 32 virtual stages for 61 decoder layers, so three stages hold one layer instead of two. Two placements are forced — PP0's first stage also carries the embedding, PP15's last stage carries the loss — leaving exactly one free choice.

Megatron's default layout spends that free choice on PP15, which is the wrong rank:

- Under 1F1B, the earliest ranks hold the most in-flight activations and the last rank the fewest.
- The memory peak is therefore PP1, the first *fully loaded* rank.
- Measured on 32 nodes: PP1 runs at 88% of HBM while PP15 idles at 23% — a 64-point spread across ranks doing nominally equal work.

Primus exposes the layout as a string, so moving the free short stage from PP15 to PP1 is a one-line change:

```text
default:  Et|(tt|)*14,t|(tt|)*15,tL
tuned:    Et|t|(tt|)*29,tL
```

Both place all 61 decoder layers, and they differ on only two ranks:

| PP rank | Default (VPP0 / VPP1) | Tuned (VPP0 / VPP1) |
|---|---|---|
| 0 | `E` + 1 layer / 2 layers | `E` + 1 layer / 2 layers |
| 1 | 2 layers / 2 layers | **1 layer** / 2 layers |
| 2–14 | 2 layers / 2 layers | 2 layers / 2 layers |
| 15 | **1 layer** / 1 layer + `L` | 2 layers / 1 layer + `L` |

A flat memory profile is not the goal in itself. The headroom it frees on the peak ranks is what pays for the next step: less recompute.

### Step 2: recompute only where the pressure is

The default strategy (`--recompute_num_layers 1 --recompute_method block`) recomputes one layer in every virtual stage — 32 of the 61 layers — whether or not a given rank needs the memory. Primus instead accepts an explicit list of global layer IDs, so recompute lands exactly where it is needed:

```bash
--recompute_layer_ids "0,1,2,4,6,8,10,12,14" --recompute_granularity full
```

The list is chosen against the measured memory profile, following one loop:

1. Dump per-rank memory and pipeline timings (`--dump_pp_data`, visualized with [`tools/visualization/pp_vis/vis.py`](https://github.com/AMD-AGI/Primus/blob/main/tools/visualization/pp_vis)).
2. Find the ranks with unused headroom.
3. Drop their layers from the recompute list.
4. Repeat until the peak rank approaches the HBM ceiling.

**How we actually walk that loop.** Not by hand. We are building a tuning agent that searches the configuration space — legal parallelism combinations, pipeline layouts, and recompute sets — scoring each candidate with the Primus projection tool (covered below) as its oracle, so most of the search costs no cluster time. The agent is still under development, so both the layout string above and the layer-ID lists below are **semi-automated**: the agent narrows the space, and engineering judgement settles the final choice. We plan to open-source it once it is ready.

Figure 6 shows the endpoints of that loop at two scales.

<table>
  <tr>
    <td><img src="imgs/dsv3_mem_dist_32n.png" alt="Per-PP-rank memory usage on 32 nodes, default vs tuned" width="100%"></td>
  </tr>
  <tr>
    <td><img src="imgs/dsv3_mem_dist_128n.png" alt="Per-PP-rank memory usage on 128 nodes, default vs tuned" width="100%"></td>
  </tr>
</table>

**Figure 6: Per-pipeline-rank peak memory on 32 nodes (top) and 128 nodes (bottom), default configuration versus tuned layout plus selective recompute.** Flattening the profile converts idle HBM into a smaller recompute budget: the 32-node spread narrows from 64 points to 28, and the 128-node spread from 50 points to 36.

### Step 3: re-tune per scale, because the right answer moves

**The right number of recomputed layers is a property of the scale, not the model.** The global batch size grows with node count while the micro-batch count per iteration stays fixed, so a larger run has a wider data-parallel group. The distributed optimizer shards optimizer state and gradient buffers across that wider group more finely, handing static memory back to every GPU — and that returned memory buys fewer recomputed layers:

| Nodes | Data-parallel size | Recompute layer IDs | Layers recomputed |
|---|---|---|---|
| 32 | 16 | `0,1,2,4,6,8,10,12,14,16,34,36,38,40,50` | 15 of 61 |
| 64 | 32 | `0,1,2,4,6,8,10,12,14,16,34,36` | 12 of 61 |
| 128 | 64 | `0,1,2,4,6,8,10,12,14` | 9 of 61 |

The boundary is sharp rather than gradual. Trimming the 128-node list by a single ID, to `0,1,2,4,6,8,10,12`, still runs fine at 128 nodes but goes OOM at both 32 and 64 — the same eight layers that leave headroom on a wide data-parallel group overflow a narrow one. An explicit ID list makes this per-scale re-tuning cheap; a uniform layer count cannot express it at all.

### What it buys

Layout and recompute together shorten the step at both scales, and the win grows with scale because the recompute budget shrinks faster than the pipeline bubble grows:

| Nodes | Configuration | Iteration time (s) | Throughput vs baseline |
|---|---|---|---|
| 32 | Default layout, 1 recompute layer per virtual stage | 23.21 | 1.00x |
| 32 | Tuned layout + 15 recompute IDs | 22.59 | **1.028x** |
| 128 | Default layout, 1 recompute layer per virtual stage | 23.75 | 1.00x |
| 128 | Tuned layout + 9 recompute IDs | 21.68 | **1.095x** |

Figure 7 shows why, using the pipeline visualizer on 128 nodes.

![Figure 7: 128-node pipeline schedule, default configuration versus tuned layout and recompute IDs](imgs/dsv3_pp_schedule_128n.png)

**Figure 7: Pipeline schedule for one iteration on 128 nodes** — default configuration (top) versus tuned layout and recompute IDs (bottom). Forward chunks are blue, backward chunks green, bubbles grey; per-rank bubble percentage is annotated on the right.

Two things to read out of it:

- Bubble on the two boundary ranks drops from 30.6% and 35.1% to 22.0% and 11.7%, and the sampled iteration time drops from 22.6 s to 21.3 s.
- Several *interior* ranks report a higher bubble percentage after tuning. With nine layers recomputed instead of 32, the non-bubble work itself shrinks, so the same absolute bubble becomes a larger fraction of a shorter step. Iteration time, not bubble ratio, is the metric that matters.

**Scaling from 32 to 128 nodes.**

- **Fixed configuration:** 128 nodes retain 98.5% of the 32-node per-GPU throughput — a 1.5% loss over a 4x growth in GPU count.
- **Re-tuned per scale:** 128 nodes reach 104.1% of the 32-node reference. Not because communication got cheaper, but because the wider data-parallel group frees enough memory to cut recompute from 15 layers to 9, and the compute saved outweighs the extra collective cost.

![Figure 8: DeepSeek-V3 scaling from 32 to 128 nodes on AMD Instinct MI355X](imgs/dsv3_scaling.png)

**Figure 8: DeepSeek-V3 normalized per-GPU throughput from 32 to 128 nodes on AMD Instinct MI355X** — a single fixed configuration (left) versus recompute IDs re-selected per scale (right), both normalized to the 32-node result.

### What did not work

Two rejected configurations are worth recording, because they show the tuning is not simply "more freedom is better":

- **VPP1** costs 14–21% in step time even with its own layout and recompute tuning, and on this stack it required disabling gradient-reduce/all-gather overlap to run at all.
- **An aggressively non-uniform layout** (1–3 layers per virtual stage, chosen to equalize memory) is 21% slower than the VPP2 baseline. It does flatten memory, but it unbalances compute, and the pipeline then runs at the speed of its slowest stage.

Balanced compute per stage dominates. Memory balance is only worth chasing once compute is already even.

### 1700 steps without interruption

A short benchmark cannot show whether a 1024-GPU job holds together, so we ran the tuned 128-node configuration on real C4 data for 1700 consecutive steps — 10 hours 37 minutes:

- **No interruption.** 1700 steps back to back on 1024 GPUs: no restart, no failed rank, no operator intervention, and no skipped step.
- **Step time holds.** Median 22.08 s, with 93.2% of steps within ±2% of it.
- **No degradation over time.** Comparing the first tenth of the run against the last shows +1.8% drift across 10.6 hours.
- **Rare outliers, not a trend.** About 2% of steps run slower (the slowest is 89 s), consistent with periodic host-side and network interference.
- **Startup is bounded.** Steps 1 and 2 cost 307 s and 165 s for lazy initialization and kernel autotuning; step 3 onward is already at steady state.

The long-run median (22.08 s) sits about 2% above the short-benchmark step time quoted earlier (21.68 s). That gap is the drift above, which is why benchmark and long-run numbers are reported separately rather than mixed.

---

## Time-to-Train: GPT-OSS-20B

The DeepSeek-V3 study shows how memory and pipeline scheduling determine throughput at scale. GPT-OSS-20B exposes a different time-to-train problem: reaching a fixed quality target on one node means accounting for initialization and kernel warm-up, training steps, periodic evaluation, communication, input-pipeline work, and failed or restarted runs—not just maximizing steady-state tokens per second.

### GPT-OSS-20B optimization journey

This discussion is scoped to the validated single-node recipe; later multi-node development is intentionally excluded. The public Primus entry points are the [GPT-OSS-20B training config](https://github.com/AMD-AGI/Primus/blob/main/examples/mlperf/gpt_oss_20b/configs/MI355/gpt_oss_20B-FP8-mlperf-pretrain.yaml), [MI355X system configuration](https://github.com/AMD-AGI/Primus/blob/main/examples/mlperf/gpt_oss_20b/config_MI355X_1x8x1_tp1pp1ep1_gbs32.sh), and [outer timing script](https://github.com/AMD-AGI/Primus/blob/main/examples/mlperf/gpt_oss_20b/run_and_time.sh). The shell configuration deliberately overrides several YAML defaults; the effective recipe is:

| Setting | Value |
|---|---|
| Hardware | 1 node × 8 AMD Instinct MI355X GPUs |
| Model | 20B total parameters, approximately 3.6B active per token, 24 layers, 32 experts, top-4 routing |
| Attention | Alternating sliding-window and full attention, sequence length 8192 |
| Parallelism | TP1 / PP1 / EP1 / DP8 |
| Batch | MBS 4 / GBS 32 |
| Precision | E4M3 tensorwise FP8 for linear layers |
| Grouped-GEMM backend | Triton |
| Data | Tokenized C4 |
| Quality target | Validation log perplexity ≤ 3.34 |
| Evaluation | Every 12,288 training samples, over 1024 validation sequences |

The fixed quality and evaluation contract is the first optimization guardrail. Evaluation is expressed in *samples*, not iterations, so changing global batch size does not silently change evaluation frequency or thoroughness. Within an A/B test at a given scale, a performance change is retained only if it preserves the same optimizer schedule and reaches the same quality target.

The recipe evolved in four stages:

| Stage | Main change | Why it improves TTT |
|---|---|---|
| Initial baseline | EP8, DeepEP/flex dispatch, sync-free MoE stage 2, MBS 2, hybrid/delayed FP8 | Established a functional MoE baseline, but carried communication and synchronization machinery that was not optimal for this single-node model |
| Topology correction | EP1, all-to-all dispatcher, grouped GEMM, MBS 4, fused cross-entropy | Removed unnecessary expert-parallel communication and increased work per launch while preserving GBS 32 |
| Critical-path reduction | BF16 gradient reduction, eight DDP buckets, overlapped reduce/gather, identity-sort elimination, tensorwise FP8, tuned RMSNorm | Reduced communication bytes, tensor movement, CPU synchronization, and high-frequency kernel overhead |
| Final recipe hardening | Fused residual/RMSNorm, SwiGLU no-cat, fused router/activation, tuned attention/RoPE, RCCL parameter gather, offline GEMM selection, warm-up and log suppression | Hardened a reproducible single-node recipe without changing its quality contract |

**Choose parallelism for the workload, not for the architecture label.** The original EP8 path divided 32 experts across the eight GPUs and therefore required expert dispatch collectives. GPT-OSS-20B is small enough for each MI355X to hold the complete expert set, so the final single-node recipe uses EP1 and DP8. DeepEP and sync-free MoE are disabled (`use_turbo_deepep: false`, `turbo_sync_free_moe_stage: 0`), while grouped GEMM remains enabled locally. At TP1/EP1, the per-expert index can be an identity permutation; `MOE_SKIP_IDENTITY_SORT=1` detects that case and removes two otherwise redundant sort/copy paths. After the accompanying memory and kernel tuning, MBS rises from 2 to 4 without changing GBS.

**Reduce data movement before adding more compute kernels.** Gradient reduction runs in BF16, DDP uses multiple buckets, and gradient reduction overlaps parameter gathering. We also evaluated moving parameter shards through SDMA peer copies, but the v26.5 stack keeps the RCCL path by default after the SDMA path exposed a barrier-related regression; the experimental path should not be counted as part of the validated recipe.

**Optimize the complete token path.** E4M3 tensorwise FP8 replaces the initial hybrid/delayed recipe. The expert path combines grouped GEMM with fused router/activation work, residual-plus-RMSNorm fusion, and a SwiGLU backward path that writes gate and up gradients directly into their final layout instead of concatenating and splitting temporary tensors. For attention, GPT-OSS alternates sliding-window and full-attention layers; backend selection therefore treats dense/SWA and forward/backward shapes separately. Native SBHD execution removes layout-conversion transposes, the dataloader stops constructing and copying a dense `(B, 1, S, S)` CPU mask every step, and eligible gfx950 head-dimension-64 shapes use tuned forward/backward kernels.

**GBS-tuned recipes expose a statistical-efficiency trade-off.** The public [MLCommons RCP logs](https://github.com/mlcommons/training/tree/master/small_llm_moe_pretraining/primus/rcp_logs) contain 20 convergence runs for each of GBS 16, 32, and 64. These are reference convergence characterizations, not performance logs from the single-node FP8 recipe described above. They also tune learning rate and warm-up with GBS, so the comparison must not be read as a single-variable causal experiment. Using the first evaluation at or below the 3.34 target:

| GBS | LR / warm-up updates | Median samples to target | Mean ± std. samples | Range across 20 runs | Median optimizer updates |
|---:|---:|---:|---:|---:|---:|
| 16 | 4e-4 / 128 | 196,608 | 194,765 ± 8,243 | 184,320–208,896 | 12,288 |
| 32 | 8e-4 / 128 | 233,472 | 234,701 ± 7,873 | 221,184–245,760 | 7,296 |
| 64 | 1e-3 / 192 | 294,912 | 301,670 ± 15,168 | 282,624–331,776 | 4,608 |

Across these tuned RCP recipes, moving from GBS 16 to 32 raises the median number of samples needed for convergence by **18.8%**; moving from 32 to 64 raises it by another **26.3%**. GBS 64 therefore uses **50% more samples** than GBS 16 at the median, even though it reaches the target in fewer optimizer updates. Because evaluation occurs every 12,288 training samples, the observed counts are quantized to that interval: the median run reaches the target at the 16th, 19th, and 24th evaluation for GBS 16, 32, and 64, respectively.

The RCP files log `eval_samples` metadata of 1024, 2048, and 4096 for GBS 16, 32, and 64, whereas the validated single-node recipe explicitly fixes evaluation to 1024 sequences. We therefore use the RCP logs only for their training `samples_count` convergence points, not to compare evaluation cost or to claim FP8 convergence. The GBS 32 recipe is a system-level compromise: its larger micro-batch improves device utilization, but its extra training samples must still be paid for in TTT.

This relationship can be summarized as:

`TTT ≈ samples-to-target / sustained sample throughput + number-of-evaluations × evaluation time + initialization and warm-up time`

**Account for lifecycle overhead explicitly.** The recipe runs three synthetic warm-up steps before the MLPerf `RUN_START` event, preselects hipBLASLt solutions for recurring shapes, and suppresses high-frequency non-MLPerf logging. The outer `run_and_time.sh` timer starts before `torchrun`, so launch-to-finish time still includes process initialization and warm-up even though the MLPerf timed interval does not. We therefore treat these as two distinct metrics:

- **User-visible TTT:** process launch to the first evaluation that reaches 3.34, including initialization and warm-up.
- **MLPerf timed interval:** `RUN_START` to the qualifying evaluation, following the benchmark rules.

An MLPerf score requires 10 consecutive runs: discard the fastest and slowest, average the remaining eight `RUN_START`-to-`RUN_STOP` durations, and validate the run set with the RCP checker. The benchmark recipe disables checkpointing and final saving, so this score does not include the checkpoint/recovery overhead expected in a production training job.

The repository history records the major configuration evolution, but it does not contain the repeated baseline and final result logs needed to publish an aggregate TTT speedup. We therefore do not add component wins together or reuse the **9.7%** GPT-OSS throughput uplift from Figure 2: that A/B test uses sequence length 4096, MBS 2, GBS 16, and a FlyDSL backend, while the MLPerf time-to-quality recipe uses sequence length 8192, MBS 4, GBS 32, and a different validated kernel stack. An official TTT claim should report both timing boundaries above, the eight-run score and dispersion, and confirmation that the run set reaches validation log perplexity 3.34 and passes the RCP checker.

---

## Small MoE Models: a Different Bottleneck Regime

<!---
OWNER / FILL: Wei Huang. This work originated from MLPerf but the narrative should
center on the MoE training optimizations and the B200 comparison, not MLPerf
process. Before publishing:
  - Ensure the B200-vs-MI355X comparison is apples-to-apples (same model config,
    precision/FP8 recipe, patch set, multi-rank averaging). Only publish the
    comparison once it is clean.
  - Replace [X] step-time / throughput placeholders and add the comparison figure.
NOTE: the kernel-level MI355X-vs-B200 GEMM comparison lives in the low-precision
section; this section is the end-to-end training comparison. Keep the two clearly
scoped so the B200 reference does not read as duplicated.
--->

The GPT-OSS case study above also illustrates how small MoE models differ from trillion-parameter workloads: compute per layer is modest, so framework overhead, gradient reduction, and normalization/activation kernels dominate instead. On the same GPT-OSS-class MoE model (32 experts, top-4, 8K sequence length) and a single 8×MI355X node, we isolated a set of optimizations that are broadly applicable in this regime:

- **BF16 gradient reduction** (`grad_reduce_in_bf16`) — the single largest step-time win here, reducing communication volume and freeing significant memory.
- **Tuned normalization kernels** — a fast RMSNorm path that avoids regressions seen with generic implementations on this stack.
- **Precision-aware optimizer and memory tuning** — enabling larger micro-batches for higher hardware utilization.
- **Fused RoPE/attention and sync-free MoE tuning** — removing small-kernel and CPU-sync overhead.

Where the [low-precision section](#low-precision-expert-gemms) compared MI355X and B200 at the level of individual GEMM kernels, here we use NVIDIA B200 as the end-to-end reference point on the same small MoE training configuration.

<!---
OWNER / FILL: Wei Huang. Replace [X] with the finalized, apples-to-apples numbers
and add the comparison chart at imgs/b200_comparison.png.
--->

AMD Instinct MI355X reaches **[X]** ms/step (**[X]** TFLOP/s/GPU) versus **[X]** ms/step on NVIDIA B200. _(to be finalized as an apples-to-apples comparison)_

![Figure 9: Small MoE training step time — AMD Instinct MI355X vs NVIDIA B200](imgs/b200_comparison.png)

**Figure 9: Small MoE training step time — AMD Instinct MI355X vs NVIDIA B200** _(placeholder — asset and numbers to be finalized)_

---

## Beyond Megatron-LM: the JAX (MaxText) Path

<!---
OWNER / FILL: Liying Li. This section summarizes the now-published JAX dropless-MoE
work (grouped GEMM + DeepEP in Primus-Turbo's JAX front-end, integrated into the
ROCm MaxText fork). For the full treatment — the FFI / custom-VJP integration,
the memory-wall analysis, and the complete throughput/convergence tables — see
the dedicated blog linked at the end.
--->

Primus also supports MoE training on the JAX backend via [MaxText](https://github.com/ROCm/maxtext.git) (the ROCm fork), integrated through a thin backend adapter that drives MaxText's native training loop. On this path, MoE efficiency starts from MaxText's native controls — block-sparse/grouped expert matmul (`megablox` / `sparse_matmul`), expert capacity (`capacity_factor`), and expert parallelism across intra-node and inter-node mesh axes — combined with ROCm-tuned XLA and Transformer Engine settings (latency-hiding scheduler, collective-combine thresholds, hipBLASLt/CK attention). Primus provides ready-to-run MoE configs for models including DeepSeek-V2, Mixtral, Qwen3-30B-A3B, and Grok on both MI300X and MI355X.

**The dropless dilemma.** On the JAX/MaxText stack, MoE training has historically forced an unhappy choice. The default `dense_matmul` path makes expert shapes static by fixing a per-expert *capacity* (`capacity_factor`) and **dropping** the tokens that overflow — fast, but lossy. The faithful `sparse_matmul` path is **dropless** — every routed token reaches its expert via a ragged, sorted-by-expert grouped matmul — but in pure JAX it hits two memory walls: the built-in `jax.lax.ragged_dot` expert matmul OOMs at ~444 GiB even at a per-device batch size of 1, and because `jax.jit` traces static shapes, the `ragged_all_to_all` routing shuffle must allocate the worst case, OOMing at ~242 GiB on DeepSeek-V3 671B. Dropless was therefore infeasible at production scale.

**Two Primus-Turbo primitives brought to JAX.** [Primus-Turbo](https://github.com/AMD-AGI/Primus-Turbo) closes that gap by exposing two Composable Kernel (CK)-backed primitives as first-class JAX ops through the [XLA FFI](https://docs.jax.dev/en/latest/ffi.html) — with `custom_vjp` autodiff and `shard_map` sharding rules, so they compose cleanly with FSDP:

- **Grouped GEMM (GMM).** A single-launch kernel over the ragged, variable-`M` per-expert groups that the dropless expert FFN needs (plus its two backward grouped GEMMs, including the variable-`K` weight-gradient case). This removes the `ragged_dot` matmul wall and is what makes dropless trainable at all.
- **DeepEP dispatch/combine.** A token-aware expert-parallel all-to-all (dispatch sends each token to the rank owning its selected expert; combine reverses and reduces the results), over xGMI intranode and RDMA internode. Because its worst-case receive buffer is a static shape, the entire MoE forward traces **sync-free by default** under `jax.jit`, while leaner buffer management trims the transient footprint enough to claw back a step of batch size.

These are wired into MaxText behind two config flags — `use_turbo_grouped_gemm` and `use_turbo_deepep_dispatch` — with careful custom-VJP fan-out/fan-in, out-of-group masking, and a once-per-process `setup()` bootstrap, and with **zero overhead when the flags are off**, so the default graph is byte-for-byte unchanged.

**Results.** On DeepSeek-V3 671B across 8 nodes × 8 AMD Instinct MI355X (64 GPUs, sequence length 4096, FSDP=8, bf16), the grouped-GEMM + DeepEP dropless path is:

- **Feasible where pure JAX was not** — the grouped GEMM removes the ~444 GiB expert-matmul wall, and DeepEP's leaner all-to-all (~15% lower transient footprint) claws back a batch-size step, fitting per-device batch size 8–9 where the `ragged_all_to_all` dropless path OOMs at 8.
- **The fastest dropless option** — ~1180 tokens/s/device at per-device batch size 8, beating the `ragged_all_to_all` dropless path at every feasible batch size and reaching ~2× the throughput of a high-capacity (`capacity_factor=4`) dense config.
- **Numerically faithful and Pareto-superior to token dropping** — over a 2000-step C4 run its loss tracks the `ragged_all_to_all` dropless path to within 0.004 and converges below every capacity-factor dense config (final loss 5.003 vs 5.163 for the `capacity_factor=1.25` default — a 0.16-nat improvement), reaching lower loss at equal wall-clock even after paying the routing-imbalance throughput tax on real data.

<table>
  <tr>
    <td><img src="imgs/jax_moe_throughput.png" alt="Per-device TGS across dense-cf / sparse-gmm / sparse-gmm-deepep configs on DeepSeek-V3 671B" width="100%"></td>
    <td><img src="imgs/jax_moe_convergence.png" alt="C4 training-loss vs. step for the same configs" width="100%"></td>
  </tr>
</table>

**Figure 10: Dropless MoE on the JAX (MaxText) path on DeepSeek-V3 671B (8×8 MI355X, FSDP=8, bf16) — (left) per-device throughput (TGS) for the grouped-GEMM + DeepEP dropless config vs. capacity-factor dropping and the `ragged_all_to_all` dropless baseline; (right) C4 training-loss convergence for the same configs.**

For the full treatment — the FFI / `custom_vjp` integration, the fan-out/fan-in and out-of-group-masking correctness details, and the complete throughput and convergence tables — see the dedicated [Dropless MoE Training in JAX with Primus-Turbo](https://rocm.blogs.amd.com/software-tools-optimization/maxtext-dropless-moe/README.html) blog.

---

## Planning Before You Train: Primus Projection

The DeepSeek-V3 study above is, in effect, a search over configurations — and running that search on a cluster is expensive. At MoE scale, a misconfigured run can OOM after hours of queueing or leave much of the cluster underutilized. Primus ships a **projection** tool that answers "Will it fit?" and "How fast will it be?" before any GPU time is committed:

- **Memory** — analytical per-GPU estimation of parameters, optimizer state, and activations, including the `topk`-scaled MoE activation footprint that dominates at high expert counts.
- **Performance** — benchmarks representative layers on as few as one GPU, then projects to multi-node clusters using communication and pipeline-schedule models; a pure-CPU simulation mode requires no GPU at all.

Every published validation case lands within 10% of measured throughput, and the MoE case — 8-node Mixtral 8x22B at EP8 / PP4 / VPP2 — is within 1.4%. This is also the oracle behind the tuning agent mentioned earlier. For the full treatment, see [Primus Projection: Estimate Memory and Performance Before You Train](https://rocm.blogs.amd.com/software-tools-optimization/primus-projection/README.html).

---

## Future Outlook

<!--- OWNER / FILL: TBD — confirm the forward-looking items below with each owner. --->

Looking ahead, we are pursuing several directions:

- **Productionizing the MoE megakernel** as a feature-flagged Primus-Turbo operator, with FP8/MXFP8 expert weights and multi-node scaling.
- **Closing the FP8 end-to-end gap** by further reducing quantization and amax-reduction overhead so kernel-level FP8 speedups translate fully to end-to-end throughput.
- **Deeper communication/compute overlap** across dispatch, grouped GEMM, and pipeline schedules for the largest MoE models.
- **Broader backend parity**, bringing DeepEP/grouped-GEMM-class optimizations and more MoE models to the JAX (MaxText) path.
- **Wider model coverage** across the growing open-source MoE family.

---

## Acknowledgments

<!--- OWNER / FILL: TBD — finalize the team/individual acknowledgments before publishing (see the first MoE blog for the format: CK, aiter, AIG-Models, ROCm/DeepEP, rocSHMEM, mori teams, plus the Primus TAS team and contributors to this blog). --->

We thank the collaborating teams and individuals across the ROCm and Primus ecosystem — including the Composable Kernel, AITER, FlyDSL, ROCm/DeepEP, and MaxText teams, and the AMD AI Brain – Training at Scale (TAS) team — whose contributions made this work possible.

---

## Disclaimers

The estimates, projections, and benchmark numbers in this blog are intended for
engineering guidance. Results depend on hardware configuration, software
versions, model settings, and workload characteristics, and may change as these
evolve. Numbers should be independently reproduced on the target system before
being treated as official performance claims.

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED "AS IS" WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
