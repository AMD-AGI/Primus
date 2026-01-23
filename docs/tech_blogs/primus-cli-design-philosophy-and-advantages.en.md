---
title: "Primus CLI: Design Philosophy and Advantages"
date: "2026-01-23"
tags: ["Primus", "CLI", "ROCm", "LLM Training", "HPC", "Slurm", "Developer Tools"]
---

## Why a unified CLI matters for large-scale training

As large-scale model training stacks evolve, one persistent problem remains: launching an experiment reliably is often harder than writing the training code. The complexity shows up in environment differences (local vs container vs Slurm), distributed settings, GPU/network topology, and a growing set of “side tasks” (benchmarks, preflight checks, diagnostics).

Primus CLI was built to address this problem by providing **a unified, consistent entry point** that consolidates training, benchmarking, and environment checks into one command structure—while keeping the execution path consistent across environments.

This post focuses on **design philosophy** and **practical advantages**. For usage details, see `docs/cli/README.md` and the full guide `docs/cli/PRIMUS-CLI-GUIDE.md`.

## Design principles

### 1) Unified entry, unified mental model

In many training codebases, training, benchmarks, and preflight checks are launched via different scripts with different flags and different environment assumptions. Primus CLI unifies those workflows under a single CLI that is organized via subcommands.

Examples:

```bash
primus-cli direct -- train posttrain --config exp.yaml
primus-cli direct -- benchmark gemm -M 8192 -N 8192 -K 8192 --dtype bf16
primus-cli direct -- preflight --host --gpu --network
```

Why this helps:

- One command family to remember
- Less duplicated “glue” logic across scripts
- Lower onboarding friction for new users

### 2) Preserved execution path across environments

A core goal of Primus CLI is to **keep the execution entry consistent across environments**. Whether you run locally, in a container, or via Slurm, Primus keeps the same task semantics and code path, only changing the runtime preparation layer.

Primus supports three execution modes (as documented in `docs/cli/README.md`):

- **Direct**: quick validation, local development
- **Container**: environment isolation and reproducibility
- **Slurm**: multi-node distributed execution on HPC clusters

The command structure stays stable:

```bash
# Local
primus-cli direct -- benchmark gemm -M 4096 -N 4096 -K 4096

# Container
primus-cli container --image rocm/primus:v25.10 -- benchmark gemm -M 4096 -N 4096 -K 4096

# Slurm
primus-cli slurm srun -N 2 -- benchmark gemm -M 16384 -N 16384 -K 16384
```

Why this helps:

- No diverging “local script” vs “cluster script”
- Easier debugging (same entry path, similar logs)
- Reduced environment pollution and fewer “works on my machine” issues

### 3) Modular and extensible by design

Primus CLI is designed to be extended without destabilizing the core. New tasks can be added as additional subcommands or suites, without rewriting the launcher or duplicating wrappers.

In practice, this keeps the CLI core stable while allowing the tooling surface to grow with new needs (new benchmarks, new diagnostics, new training workflows).

### 4) Python-first, Slurm-friendly

Primus uses Python as the orchestration language (good fit for YAML configs, framework integration, and tooling), while keeping Slurm workflows first-class. In `runner/`, the runtime-specific launchers encapsulate environment preparation and scheduling details; the task semantics remain consistent.

## How the architecture maps to the repository

At a high level, Primus CLI follows a three-layer structure:

### Runtime layer: direct / container / slurm

The `runner/` directory contains the entrypoints and launchers that implement environment-specific behavior while preserving the same task structure. For example:

- `runner/primus-cli`
- `runner/primus-cli-direct.sh`
- `runner/primus-cli-container.sh`
- `runner/primus-cli-slurm.sh`
- `runner/primus-cli-slurm-entry.sh`

### Hook / patch layer: workflow composition without intrusion

Training workflows often require pre/post steps (preflight checks, dependency installation, checkpoint preparation, hotfixes). Primus supports a hook/patch mechanism so these steps can be composed without modifying training code.

This also helps keep behavior consistent across environments, because hooks are executed as part of the same preserved entry path.

### Task execution layer: train / benchmark / preflight / analyze

The task layer implements what users care about: training, micro-benchmarks, preflight checks, and analysis tools. It stays focused on “what to do,” while the runtime layer focuses on “where/how to run.”

## Practical advantages

- **Lower cognitive load**: one command family for multiple workflows
- **Higher reproducibility**: stable semantics across local/container/Slurm
- **Better debuggability**: fewer divergent code paths, more consistent logs
- **Less glue code**: hooks/patches capture common pre/post steps
- **Safer extensibility**: add new capabilities without rewriting the core

## Example workflows

### Training

```bash
primus-cli direct -- train posttrain --config examples/megatron_bridge/configs/MI355X/qwen3_8b_sft_posttrain.yaml
```

### Benchmarks

```bash
primus-cli direct -- benchmark gemm -M 8192 -N 8192 -K 8192 --dtype bf16
primus-cli direct -- benchmark rccl --op allreduce --num-bytes 1048576
```

### Preflight checks

```bash
primus-cli direct -- preflight --host --gpu --network
```

## Roadmap (directional)

- More backends and workflow types (beyond current training backends)
- A more unified “training + fine-tuning” command surface
- Diagnostics and auto-tuning tools (topology, RCCL tuning, profiling/reporting)
- Curated reproducible examples (“recipes”) for popular models and clusters

## Closing

Primus CLI aims to be the most reliable entry point for AMD GPU training workflows by hiding environment complexity behind a unified interface—without sacrificing HPC realities like Slurm scheduling and multi-node debugging.

