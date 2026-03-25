# Primus overview

## Executive summary

**Primus (Primus-LM)** is a YAML-driven training framework for large-scale foundation model work on AMD GPUs (ROCm). It targets **machine learning engineers**, **researchers**, and **platform/operations teams** who need reproducible, multi-backend training pipelines on AMD Instinct hardware.

**Repository:** [https://github.com/AMD-AIG-AIMA/Primus](https://github.com/AMD-AIG-AIMA/Primus)

**License:** The project README states Apache License 2.0; the repository root `LICENSE` file is MIT. Treat licensing as project-specific and confirm with your compliance process before redistribution.

---

## What Primus provides

| Area | Description |
|------|-------------|
| **Multi-backend training** | One workflow surface over Megatron-LM, TorchTitan, JAX MaxText, Megatron Bridge, and HummingbirdXT. |
| **Unified CLI** | `primus-cli` with **direct** (bare metal), **container** (Docker/Podman), and **slurm** (cluster) execution modes. |
| **YAML-driven configuration** | Experiment, model, module, and platform presets composed from reusable fragments under `primus/configs/`. |
| **Benchmark suite** | Built-in benchmarks (for example GEMM) for quick hardware and stack validation. |
| **Preflight diagnostics** | Cluster-oriented checks for host, GPU, and network health before long jobs. |
| **Performance projection** | Tools to estimate memory use and throughput without occupying a full cluster. |

Workflows span **pretraining**, **post-training** (including SFT and LoRA), and **RL-oriented** pipelines, depending on backend and recipe support.

---

## Primus ecosystem (conceptual)

Primus-LM sits between stability/platform services and low-level operators:

```
                    +------------------+
                    |   Primus-SaFE    |
                    | (stability /     |
                    |  cluster mgmt)   |
                    +--------+---------+
                             |
                    +--------v---------+
                    |    Primus-LM     |
                    |  (this repo:     |
                    |   training)      |
                    +--------+---------+
                             |
                    +--------v---------+
                    |  Primus-Turbo    |
                    | (operators /     |
                    |  kernels)        |
                    +------------------+
```

- **Primus-SaFE**: Stability and fault-tolerance oriented cluster management (external component).
- **Primus-LM**: Training orchestration, backends, CLI, and configs (this repository).
- **Primus-Turbo**: High-performance operators (for example FlashAttention-style kernels, GEMM, collectives).

---

## Supported backends

| Backend | Typical use |
|---------|-------------|
| **Megatron-LM** | Broadest model coverage; default for many GPT-style and MoE recipes. |
| **TorchTitan** | PyTorch-native large-model training paths. |
| **JAX MaxText** | JAX/Flax training stacks aligned with MaxText. |
| **Megatron Bridge** | Post-training and bridge workflows on top of Megatron-related stacks. |
| **HummingbirdXT** | Additional integrated training path where enabled by your deployment. |

Backend choice is expressed in experiment YAML and resolved through Primus’s adapter layer (see [glossary](./glossary.md)).

---

## Supported hardware and stack

| Item | Requirement |
|------|----------------|
| **GPUs** | AMD Instinct **MI300X**, **MI325X**, **MI355X** |
| **Platform** | **ROCm** (version **≥ 7.0** recommended) |
| **Container image (reference)** | `docker.io/rocm/primus:v26.1` |

Exact kernel and driver packages should match AMD’s documentation for your GPU SKU and ROCm release.

---

## Key dependencies

Primus integrates with the following categories of software (non-exhaustive):

| Category | Examples |
|----------|----------|
| **Framework** | PyTorch (Megatron-LM, TorchTitan paths); JAX/Flax (MaxText path). |
| **AMD stack** | ROCm, RCCL, HipBLASLt (GEMM), GPU drivers. |
| **Execution** | Docker or Podman for container mode; Slurm for cluster mode. |
| **Observability & tooling** | **loguru** (logging), **Weights & Biases** (`wandb`) and other optional trackers (see `requirements.txt`). |

Install specifics are covered in [installation](./installation.md).

---

## Runtime model

At a high level, a run follows this pipeline:

1. **YAML experiment config** defines work group, modules, model preset, and overrides.
2. **`primus-cli`** selects execution mode (direct, container, or slurm) and forwards to the runner.
3. **Backend adapter** maps the resolved config to the target framework (Megatron-LM, TorchTitan, and so on).
4. **Distributed launch** typically uses **`torchrun`** (or the backend’s equivalent) to start workers across GPUs and nodes.

For CLI shape and options, see [quickstart](./quickstart.md) and [CLI reference](../02-user-guide/cli-reference.md).

---

## Repository layout (top level)

| Path | Role |
|------|------|
| `primus/` | Core library: configs, runtime, trainers, backend adapters, tools (including preflight). |
| `runner/` | CLI implementation, helpers, hooks, and launch glue. |
| `examples/` | End-to-end example YAML and recipes per backend and GPU SKU. |
| `docs/` | Upstream project documentation (architecture, CLI guides, backends). |
| `tests/` | Automated tests. |
| `tools/` | Auxiliary scripts and utilities. |
| `benchmark/` | Benchmark drivers and related assets. |
| `third_party/` | Vendored or submodule dependencies. |

---

## Next steps

- [Installation](./installation.md) — ROCm, Docker, pip, and Slurm setup.
- [Quickstart](./quickstart.md) — Run a minimal training job in minutes.
- [Glossary](./glossary.md) — Terms used across Primus docs.
