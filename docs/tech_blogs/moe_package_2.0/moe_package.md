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
NOTE: This document is a work-in-progress skeleton for coordinating the MoE
training-optimization blog. Each section is marked with an owner and a TODO
placeholder (rendered as blockquotes). Please fill in your section and remove
the placeholder note when done. Owner tags are meant to be stripped before
publishing.
--->

# MoE Training Optimization with Primus

_This blog consolidates the MoE training-optimization progress we have made since the previous [Primus MoE package blog](TODO: link to the previous MoE package blog). It spans kernel-level work (megakernel, low-precision operators), general Primus + Primus-Turbo optimizations, time-to-train improvements on large models such as DeepSeek-V3, our JAX MoE training path, and performance projection. Both the Megatron-LM and JAX backends in Primus are covered._

## Background

> _TODO — Owner: TBD_

### MoE model trends

> _TODO: Analyze the development trend of MoE models (growth in parameter count with sparse activation, fine-grained experts, shared/routed experts, larger expert counts, representative models such as DeepSeek-V3 / Qwen3-235B / Mixtral). Motivate why MoE training efficiency matters._

### Directions for MoE training optimization

> _TODO: Lay out the optimization directions this blog covers — kernels, low-precision, parallelism/scheduling, recompute, communication (DeepEP / grouped GEMM), time-to-train, and framework backends (Megatron-LM & JAX). Use this to frame the rest of the blog._

## Optimization Overview

> _TODO: One-paragraph summary of the end-to-end story and the headline numbers._

### Overall performance uplift

> _TODO: Bar chart of end-to-end training performance uplift across common MoE models (e.g., DeepSeek-V3, Qwen3-235B, Mixtral, ...). Add the asset at `imgs/moe_perf_overview.png` and describe the baseline, hardware (MI300 / MI355), and the measured speedups._

## Performance Optimization Features

### General performance optimization: Primus + Primus-Turbo

> **Owner(s):** Ruibin Zhang
>
> _TODO: Describe the general-purpose training optimizations delivered through Primus and [Primus-Turbo](https://github.com/AMD-AGI/Primus-Turbo), and their impact on MoE workloads._

### Megakernel

> **Owner(s):** Xiaoming Peng, Zhen Huang
>
> _TODO: Introduce the megakernel work — motivation, design, and measured gains on MoE layers._

### Low-precision operator optimization

> **Owner(s):** Ruibin Zhang, Kyle Zhao
>
> _TODO: Cover low-precision (e.g., FP8) operator optimizations for MoE — which ops, accuracy considerations, and speedups._

### Benchmarking against B200

> **Owner(s):** Wei Huang
>
> _TODO: Present the small-model MoE training optimizations (work originating from MLPerf). No need to center the narrative on MLPerf itself — focus on the MoE training optimizations and the comparison against B200._

### Time-to-Train (TTT) oriented optimization

#### DeepSeek-V3 performance optimization

> **Owner(s):** Lihuan Zhang
>
> _TODO: End-to-end time-to-train optimization on DeepSeek-V3. Suggested sub-topics below._

- **Fine-grained recompute** — _TODO_
- **Scaling** — _TODO_
- **24-hour run** — _TODO_
- _..._

### JAX MoE training optimization

> **Owner(s):** Liying Li
>
> _TODO: MoE training optimization on the JAX backend._

- **JAX DeepEP / grouped GEMM** — _TODO_
- **DeepSeek-V3** — _TODO_

### Primus Projection

> **Owner(s):** Lihuan Zhang
>
> _TODO: Brief introduction to Primus Projection, with a link to Anshu's earlier projection blog (TODO: link). Keep it short and point readers to the existing material._

## Future Outlook

> _TODO: What's next — upcoming optimizations, models, and framework work._

## Acknowledgments

> _TODO: Thank the collaborating teams and individuals._

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
