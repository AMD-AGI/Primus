# Primus technical blogs

Every published article about Primus, newest first. The last column points at the maintained
documentation or code for the same topic: a blog is frozen at its publication date, so where the two
disagree, the documentation wins. The [top-level README](https://github.com/AMD-AGI/Primus#-technical-blogs) lists only
the most recent handful.

## Published

| Date | Title | What it covers | In this repo |
|------|-------|----------------|--------------|
| 2026/09/03 | [Enabling DeepSeek-V4-Flash Training on AMD Instinct MI355X GPUs with Primus](https://rocm.blogs.amd.com/software-tools-optimization/primus-deepseek-v4/README.html) | DeepSeek-V4-Flash architecture, performance projection, kernel optimizations, and how to reproduce the runs | [`examples/deepseek-v4`](https://github.com/AMD-AGI/Primus/tree/main/examples/deepseek-v4) |
| 2026/08/12 | [Using ODC to Accelerate AMD SFT Training](https://rocm.blogs.amd.com/software-tools-optimization/odc-accelerate-training/README.html) | On-demand point-to-point communication replacing the FSDP all-gather / reduce-scatter, over rocSHMEM and MORI | [ODC FSDP2 patches](https://github.com/AMD-AGI/Primus/blob/main/primus/backends/megatron/patches/odc_torch_fsdp2_patches.py) |
| 2026/07/06 | [Primus Tuning Agent: Closing the Configuration-Search Loop](https://rocm.blogs.amd.com/software-tools-optimization/primus-tuning-agent/README.html) | LLM-driven search over parallelism, pipeline layout, and recompute sets, scored by projection instead of cluster time | [Tuning agent](../02-user-guide/tuning-agent.md) |
| 2026/06/10 | [Dropless MoE Training in JAX with Primus-Turbo](https://rocm.blogs.amd.com/software-tools-optimization/maxtext-dropless-moe/README.html) | Grouped GEMM brought into MaxText through JAX FFI and `custom_vjp`, with the fan-out / fan-in correctness details | [MaxText parameters](../03-configuration-reference/maxtext-parameters.md) |
| 2026/04/24 | [Primus Projection: Estimate Memory and Performance Before You Train](https://rocm.blogs.amd.com/software-tools-optimization/primus-projection/README.html) | Analytical memory estimation and benchmark-anchored throughput projection for multi-node runs | [Projection](../02-user-guide/projection.md) |
| 2026/02/23 | [Primus-Pipeline: A More Flexible and Scalable Pipeline Parallelism Implementation](https://rocm.blogs.amd.com/software-tools-optimization/primus-pipeline/README.html) | Zero-bubble schedules (zerobubble / zbv / v-half / v-min) alongside 1F1B and interleaved, in the Megatron-LM backend | [Parallelism strategies](../04-technical-guides/parallelism-strategies.md) |
| 2026/01/15 | [Deep Dive into Primus: High-Performance Training for Large Language Models](https://rocm.blogs.amd.com/software-tools-optimization/primus-deep-dive/README.html) | How the unified CLI and tuned backend presets reach peak dense-LLM throughput with minimal manual tuning | [Megatron-LM training](../02-user-guide/megatron-lm-training.md) |
| 2025/12/16 | [MoE Training Best Practices on AMD GPUs](https://rocm.blogs.amd.com/software-tools-optimization/primus-moe-package/README.html) | The foundational MoE optimizations: DeepEP dispatch, sync-free MoE, 1F1B all-to-all overlap, selective recompute | [MoE training deep-dive](../04-technical-guides/moe-training.md) |
| 2025/08/22 | [Primus: A Lightweight, Unified Training Framework for Large Models on AMD GPUs](https://rocm.blogs.amd.com/software-tools-optimization/primus/README.html) | The framework itself: YAML-driven configuration, multi-backend design, preflight validation, structured logging | [Project overview](../01-getting-started/overview.md) |

## Awaiting publication

| Date | Title | What it covers | In this repo |
|------|-------|----------------|--------------|
| 2026/08/21 | MoE Training Optimization with Primus | MegaMoE megakernel, FP8 / MXFP8 grouped GEMM, training-at-scale tuning | [MegaMoE fused MoE layer](../04-technical-guides/mega-moe.md) |

## Adding an article

Article sources are maintained in the ROCm Blogs repository, not here. Copies were vendored into this
repository once before and drifted out of sync with the published versions, down to stale author
lists, so this directory holds the index only. When an article goes live, add a row above and link to
[rocm.blogs.amd.com](https://rocm.blogs.amd.com/) rather than adding a copy. Add it to the
[top-level README](https://github.com/AMD-AGI/Primus#-technical-blogs) too if it belongs in the recent handful, dropping
the oldest entry there to keep that list short.
