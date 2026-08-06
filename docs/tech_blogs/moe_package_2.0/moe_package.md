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
section, so that section should stay first. The megakernel numbers are measured
on a reduced-depth DeepSeek-V3 proxy; keep that scope stated wherever they appear.
--->

# MoE Training Optimization with Primus

_Mixture-of-Experts (MoE) has become the default architecture for frontier-scale language models, and our users increasingly train large MoE models on AMD Instinct™ GPUs. Driven by these real workload requirements — and by where the industry has recently concentrated its MoE optimization efforts — we have built a broad set of MoE training optimizations in Primus. This blog walks through that work: kernel-level work (a fused MoE **megakernel** and **low-precision operators**), general **Primus + Primus-Turbo** training optimizations, **time-to-train** improvements on large models such as DeepSeek-V3, our **JAX (MaxText)** MoE training path, and **performance projection**. Both the Megatron-LM and JAX backends of Primus are covered. For the foundational optimizations this work builds on, see our earlier [MoE Training Best Practices on AMD GPU](https://rocm.blogs.amd.com/software-tools-optimization/primus-moe-package/README.html) post._

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

Those trends mean MoE training efficiency is no longer about fast GEMMs alone. A modern MoE step is limited by four different things at once, and an optimization that attacks only one of them tends to expose the next. The table below maps each bottleneck to the work described in this post, so you can jump straight to what is limiting your own runs.

| Bottleneck | What we did about it | Section |
|---|---|---|
| **Expert GEMM throughput** — expert FFNs dominate FLOPs | FP8/MXFP8 grouped GEMM, FlyDSL kernels, quantized-weight caching | [Low-precision expert GEMMs](#low-precision-expert-gemms) |
| **All-to-all dispatch/combine** — grows with top-k and EP width | DeepEP dispatch, 1F1B all-to-all overlap, and tile-granularity fusion inside a single kernel | [Framework foundation](#the-foundation-primus--primus-turbo), [Megakernel](#the-moe-megakernel) |
| **Memory ceiling** — sets micro-batch size and recompute cost | Precision-aware optimizer, pipeline layout, fine-grained recompute, ahead-of-time projection | [DeepSeek-V3 at scale](#time-to-train-deepseek-v3-at-256-1024-gpus), [Projection](#planning-before-you-train-primus-projection) |
| **Host and launch overhead** — dominates when layers are small | Sync-free MoE, NUMA binding and launch tuning, pipeline warm-up | [Framework foundation](#the-foundation-primus--primus-turbo), [Small MoE models](#small-moe-models-a-different-bottleneck-regime) |

Two sections sit outside that grid: the [JAX (MaxText) path](#beyond-megatron-lm-the-jax-maxtext-path), which brings the same grouped-GEMM and DeepEP primitives to a second backend, and [Primus Projection](#planning-before-you-train-primus-projection), which answers "will it fit, and how fast?" before any cluster time is spent.

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

![Figure 1: Additional Primus-Turbo optimizations improve throughput by 16.2% on Qwen3-235B-A22B, 9.7% on GPT-OSS 20B, and 5.8% on Qwen3-30B-A3B](imgs/general_opt_uplift.png)

**Figure 1: Incremental throughput uplift from additional Primus-Turbo optimizations.** Compared with the listed reference configurations, the additional Turbo and FlyDSL settings improve throughput by **16.2%** on Qwen3-235B-A22B, **9.7%** on GPT-OSS 20B, and **5.8%** on Qwen3-30B-A3B.

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

Figure 2 isolates that kernel-level payoff on MI355X alone, sweeping tokens-per-expert `M` over training-relevant sizes and reporting the FP8 (tensorwise) grouped-GEMM speedup over BF16 with quantization included in the FP8 timing.

<p align="center">
  <img src="imgs/low_precision_fp8_vs_bf16_grouped.png" alt="FP8 vs BF16 grouped GEMM speedup on MI355X" width="45%">
</p>

**Figure 2: Kernel-level FP8-vs-BF16 grouped-GEMM speedup on AMD Instinct MI355X, swept over tokens-per-expert `M` and averaged over the DeepSeek-V3 / Qwen3-235B-A22B / gpt-oss expert shapes; quantization is included in the FP8 timing.**

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

**Figure 3: FP8 tensorwise (top) and MXFP8 (bottom) GEMM throughput on AMD Instinct MI355X vs NVIDIA B200/GB200 (TE) — dense (left) and grouped (right), forward and backward, averaged over the tested shapes.**

The lesson is that low precision is not just a datatype switch: for MoE, layout and grouped scheduling matter as much as the quantization recipe. Removing the forward transpose tax and autotuning each grouped shape turns an apparent forward deficit into a forward lead, and MXFP8 layers finer-grained scaling on top of the same execution path while staying competitive with the B200/GB200 baseline.

### The MoE megakernel

<!---
OWNER / FILL: Xiaoming Peng, Zhen Huang. Still open before publishing:
  - Confirm the end-to-end numbers below are cleared for public release. They are
    measured, not projected (8x MI355X, DeepSeek-V3 4 layers + MTP, EP8, gbs 512,
    50 steps) but they are a reduced-depth proxy, not a full 61-layer run.
  - Decide how much AMD-specific implementation detail (DTOLDS/AGPR/wave
    specialization) to expose publicly.
  - Figure 4 is a schematic drawn in code (.ab/gen_megakernel_diagram.py); stage
    widths are illustrative, not measured per-stage timings. Swap in a profiled
    timeline if one is cleared for release.
--->

Expert dispatch and combine are the two costs that make an MoE layer different from a dense one, and both of them are communication. This section is about removing the boundary between that communication and the expert compute it feeds.

#### How an MoE layer executes today

A single MoE layer is a chain of eight or so operators. The router scores each token and picks its top-k experts; a permutation gathers tokens into expert order; an all-to-all **dispatch** sends each token to the rank that owns its expert; two grouped GEMMs (**FC1** gate/up and **FC2** down) with a **SwiGLU** between them do the expert compute; an all-to-all **combine** returns the results; and a final unpermute-and-scale reduces the top-k contributions back onto each token.

Each of those stages is its own kernel. The top half of Figure 4 shows the consequence.

![Figure 4: MoE layer dataflow — separate kernels today versus MegaMoE's three fused kernels](imgs/megakernel_dataflow.png)

**Figure 4: MoE layer dataflow. Top: today, one kernel per stage, with an HBM round trip at every boundary. Bottom: MegaMoE, three kernels in which each all-to-all is fused into the grouped GEMM that feeds it, overlapping at tile granularity.** _(schematic; stage widths are illustrative, not measured per-stage timings)_

#### The problem: communication and compute cannot overlap

Two costs follow from that structure, and neither is addressable by making any individual kernel faster.

**Communication and compute are serialized.** Collective libraries are host-initiated and operate at kernel granularity: a dispatch kernel occupies the device, finishes, and only then does the FC1 kernel launch. Profiling shows the MoE forward pass split roughly evenly between all-to-all communication and expert compute, so with the two in separate kernels roughly half of the layer's time is spent with the matrix cores idle. Coarse overlap across kernel boundaries — running the next microbatch's communication behind this one's compute — recovers some of that, but it cannot overlap a tile's own dispatch with its own GEMM.

**Every kernel boundary is an HBM round trip.** The permutation writes a reordered copy of the tokens; FC1 writes its activations out for SwiGLU to read back; FC2 writes its output for the combine to read again. At DeepSeek-V3's expert granularity — 256 experts with top-8 routing — the layer is a long chain of comparatively small operators, and the traffic *between* them is on the same order as the traffic the GEMMs need to do their work.

The prize, then, is not a faster dispatch or a faster GEMM. It is to put them in the same kernel so that one can hide inside the other and the intermediates never reach HBM.

#### Our approach: fuse each all-to-all into the GEMM that feeds it

The critical move is not collapsing the layer into one kernel — it is putting each all-to-all *inside* the grouped GEMM that produces or consumes its data, so the communication has something to hide behind. **MegaMoE** does this in three kernels, shown in the bottom half of Figure 4:

1. **dispatch + FC1.** The incoming all-to-all and the FC1 grouped GEMM share one kernel. The CU grid is split between dispatch and compute roles, so tokens keep streaming in from peer ranks while the matrix cores work on tiles that have already landed.
2. **SwiGLU.** A small middle kernel that also quantizes its own output, so the MXFP8 cast never becomes a separate pass over the activations.
3. **FC2 + combine.** The FC2 grouped GEMM and the outgoing all-to-all share one kernel: the GEMM epilogue pushes each finished tile to its owning rank and the reduction happens there, so results leave as they are produced instead of after the whole GEMM completes.

The backward mirrors this by the dispatch/combine duality — `dispatch(dy)` fuses with the FC2 data-gradient GEMM, and the FC1 data-gradient fuses with the combine and reduction — plus two variable-K weight-gradient kernels.

Two properties make the overlap work. Within each fused kernel the CU grid is split by role, so communication workgroups make progress while compute workgroups run MFMA GEMMs on the same device. And instead of a global barrier between the two, per-tile arrival flags let compute start on a tile the moment it lands, and let a finished tile leave for the combine immediately — which is what turns communication latency into something the GEMM hides rather than waits on. The kernels are authored in [FlyDSL](https://github.com/ROCm/FlyDSL) and mapped onto CDNA3/CDNA4 (gfx942/gfx950).

For the user, all of this is one feature flag (`use_turbo_mega_moe`) plus a precision knob (`turbo_mega_moe_precision: bf16 | mxfp8`): MegaMoE replaces the whole Megatron MoE layer, with the router feeding the fused op directly.

#### Kernel-level performance

Measured on its own, on DeepSeek-V3 expert shapes (H=7168, I=2048, 256 experts, top-8, EP8, 8192 tokens per rank), the fused layer runs as follows:

| Pass | BF16 | MXFP8 | Speedup |
|---|---|---|---|
| Forward | 6.96 ms | 5.18 ms | 1.34× |
| Backward | 13.34 ms | 8.40 ms | 1.59× |
| **Forward + backward** | **19.94 ms** | **13.21 ms** | **1.51×** |

The backward benefits most, which matters because it is also the larger half of the layer. The ratio is unchanged under a deliberately imbalanced routing, so the speedup does not depend on experts receiving equal token counts.

#### End-to-end training performance

Kernel-level wins only matter if they survive a real training step. We train DeepSeek-V3 (4 layers + MTP, EP8 on one 8×MI355X node, global batch 512, 50 iterations) and swap only the MoE implementation: within each precision the two runs differ by exactly one Megatron argument, with attention, optimizer, data, and seed held fixed.

| Precision | MoE layer | ms / step | TFLOP/s per GPU | Speedup |
|---|---|---|---|---|
| BF16 | DeepEP dispatcher + grouped GEMM | 9540 | 841 | — |
| BF16 | **MegaMoE** | **8817** | **910** | **1.082×** |
| MXFP8 | DeepEP dispatcher + grouped GEMM | 8432 | 951 | — |
| MXFP8 | **MegaMoE** | **7508** | **1069** | **1.123×** |

![MegaMoE end-to-end step time and throughput on MI355X](imgs/megakernel_e2e_perf.png)

![MegaMoE training loss versus the DeepEP baseline](imgs/megakernel_e2e_loss.png)

**Figure 5: MegaMoE end to end on DeepSeek-V3 (4 layers + MTP), 8×MI355X, EP8, global batch 512. Top: steady-state step time and throughput, median of iterations 3–50. Bottom: training loss over 50 iterations.**

The step-level gain is necessarily smaller than the 1.51× measured on the kernel — the MoE layer is only part of a training step — but it survives the trip: the per-call saving, multiplied out over the layers and microbatches of a step, lands within 4% of the measured step-time delta. Notably, **fusion and low precision compound**: MegaMoE is worth more in MXFP8 (1.123×) than in BF16 (1.082×), because quantizing everything else raises the MoE's share of the step. Fusion also lowers peak memory — 142.4 GiB against 150.1 GiB in the MXFP8 pair — since the permuted token buffers are never materialized.

The loss curves in Figure 5 stay together over the whole run, so the fused path trains as the dispatcher path does; longer-horizon validation is ongoing. MegaMoE is currently EP-only (TP=1) and dropless, and network-wide MXFP8 on DeepSeek-V3 needs the Turbo GEMM path (`use_turbo_gemm`, `use_turbo_grouped_gemm`).

MegaMoE has graduated from a research prototype into a feature-flagged Primus-Turbo operator with MXFP8 expert weights and a full backward pass. We are now extending it beyond a single node and to the remaining MoE layer variants.

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

**What did not work.** Two rejected configurations are worth recording, because they show the tuning is not simply "more freedom is better":

- **VPP1** costs 14–21% in step time even with its own layout and recompute tuning, and on this stack it required disabling gradient-reduce/all-gather overlap to run at all.
- **An aggressively non-uniform layout** (1–3 layers per virtual stage, chosen to equalize memory) is 21% slower than the VPP2 baseline. It does flatten memory, but it unbalances compute, and the pipeline then runs at the speed of its slowest stage.

Balanced compute per stage dominates. Memory balance is only worth chasing once compute is already even.

Figure 7 shows why the tuned configuration is faster, using the pipeline visualizer on 128 nodes.

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

Everything above was constrained by memory and collective traffic. Small MoE models invert that: compute per layer is modest, so framework overhead, gradient reduction, and normalization/activation kernels dominate instead. Working from a GPT-OSS-class MoE model (32 experts, top-4, 8K sequence length) on a single 8×MI355X node, we tuned a set of optimizations that are broadly applicable in this regime:

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

- **Scaling the MoE megakernel beyond a single node**, and extending it to the MoE layer variants it does not yet cover (tensor parallelism, shared experts, non-SwiGLU activations).
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
