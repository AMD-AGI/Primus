---
title: "Primus CLI: A Unified Entry Point for Training on ROCm"
date: "2026-01-23"
tags: ["ROCm", "LLM Training", "HPC", "Slurm", "Developer Tools"]
---

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

## Introduction

Training large-scale models on modern GPU clusters involves far more than model code alone.
Engineers routinely deal with environment validation, performance benchmarking, distributed launch configuration, and multi-node debugging—often before a single training step is executed.

As these workflows grow more complex, a recurring challenge emerges: **the lack of a consistent entry point**.
Different scripts are used for training, benchmarks, and diagnostics; execution semantics vary between local runs, containers, and Slurm jobs; and small environment differences can lead to non-reproducible behavior.

Primus CLI was designed to address this problem by providing **a unified entry point for training-related workflows on ROCm**, while preserving consistent execution semantics across environments.

This post focuses on the design ideas behind Primus CLI and how they translate into practical benefits for large-scale training workflows.

---

## The problem: fragmented execution paths

In many training setups, the workflow evolves organically:

- one script for local debugging  
- another wrapper for container execution  
- a separate Slurm launcher for multi-node runs  
- additional scripts for benchmarks and environment checks  

Over time, these scripts diverge. Flags drift, environment variables differ, and assumptions become implicit.
The result is a workflow that works, but is difficult to reason about, reproduce, or extend.

Primus CLI approaches this problem from a different angle: instead of adding another wrapper, it treats training, benchmarking, and diagnostics as variations of the same execution model.

---

## A unified entry, not a monolithic tool

At the surface, Primus CLI exposes a single command family with structured subcommands:

```bash
primus-cli <runtime> -- <task> [task options]
```

This structure allows different workflows—training, benchmarks, preflight checks—to share the same mental model, without forcing them into a single rigid pipeline.

Examples:

```bash
# Training
primus-cli direct -- train pretrain --config exp.yaml

# Benchmarks
primus-cli direct -- benchmark gemm -M 8192 -N 8192 -K 8192

# Environment inspection
primus-cli direct -- preflight --host --gpu --network
```

The goal is not to hide complexity, but to make it predictable.

---

## Why preserving the execution path matters

In large-scale training systems, many failures and performance regressions are not caused by model code itself, but by subtle differences in how jobs are launched across environments.

When local runs, container executions, and Slurm jobs follow different execution paths, debugging becomes guesswork. Performance numbers become difficult to compare, and fixes that work in one environment may not generalize to others.

By preserving a single execution path and varying only the runtime preparation layer, Primus CLI makes training behavior easier to reason about and results easier to trust.

---

## Preserving execution semantics across environments

A core design goal of Primus CLI is preserving the same execution semantics across environments.

Whether a job is launched locally, inside a container, or on a multi-node Slurm cluster, the task semantics remain the same. Only the runtime preparation layer changes.

```bash
# Local
primus-cli direct -- benchmark gemm -M 4096 -N 4096 -K 4096

# Container
primus-cli container --image rocm/primus:v25.10 -- benchmark gemm -M 4096 -N 4096 -K 4096

# Slurm
primus-cli slurm srun -N 2 -- benchmark gemm -M 16384 -N 16384 -K 16384
```

This consistency simplifies debugging and helps avoid the classic “works locally but not on the cluster” scenario.

---

## Designed for growth and experimentation

Training workflows rarely stay static. New benchmarks are added, diagnostics evolve, and fine-tuning or post-training stages become part of the workflow.

Primus CLI is designed to accommodate this evolution:

- new tasks can be added as subcommands  
- benchmarks can grow independently  
- workflow-specific logic can be composed without rewriting launchers  

This modular approach allows the CLI surface to expand while keeping the core execution model stable.

---

## The CLI as part of the training system

In practice, the CLI is not just a launcher—it is part of the training system itself.

Decisions made at the CLI layer affect reproducibility, debuggability, and how quickly new workflows can be introduced. Treating the CLI as a first-class component helps prevent training systems from accumulating brittle glue code over time.

---

## Python-first, HPC-aware

Primus CLI uses Python as the orchestration layer, which integrates naturally with configuration files, training frameworks, and developer tooling.

At the same time, it treats HPC realities—such as Slurm scheduling and multi-node execution—as first-class concerns rather than special cases.
The result is a workflow that feels natural in Python environments, yet remains practical on large clusters.

---

## What this enables in practice

In day-to-day usage, a unified entry point translates into tangible benefits:

- reproducibility across local, container, and cluster runs  
- simpler debugging due to consistent logs and execution paths  
- lower onboarding cost for new users and contributors  
- less glue code maintained outside the core workflow  

Teams can focus more on training behavior and performance, and less on maintaining launch scripts.

---

## Who this is for

Primus CLI is designed for teams who:

- run training workflows across multiple environments  
- care about reproducibility and debuggability at scale  
- want to evolve training pipelines without rewriting launch logic  

---

## Closing

Primus CLI aims to provide a reliable and consistent entry point for training workflows on ROCm.
By unifying training, benchmarking, and environment inspection under a single execution model, it reduces operational friction while remaining compatible with large-scale, Slurm-based HPC environments.
