# Primus overview

## Executive summary

**Primus** is a YAML-driven training framework for large-scale foundation model work on AMD GPUs. It targets **machine learning engineers**, **researchers**, and **platform/operations teams** who need reproducible, multi-backend training pipelines on AMD Instinct™ hardware. Within the broader Primus ecosystem, this repository is the training component, sometimes referred to as **Primus-LM** (see [Primus ecosystem](#primus-ecosystem) below).

**Repository:** [https://github.com/AMD-AGI/Primus](https://github.com/AMD-AGI/Primus)

---

## What Primus provides

| Area | Description |
|------|-------------|
| **Multi-backend training** | One workflow surface over Megatron-LM, TorchTitan, JAX MaxText, Megatron Bridge, and HummingbirdXT. |
| **Unified CLI** | `primus-cli` with **direct** (bare metal), **container** (Docker/Podman), and **slurm** (cluster) execution modes. |
| **YAML-driven configuration** | Experiment, model, module, and platform presets composed from reusable fragments under `primus/configs/`. |
| **Benchmark suite** | Built-in benchmarks (for example, GEMM) for quick hardware and stack validation. |
| **Preflight diagnostics** | Cluster-oriented checks for host, GPU, and network health before long jobs. |
| **Performance projection** | Tools to estimate memory use and throughput without occupying a full cluster. |

Workflows span **pretraining** and **post-training** (including SFT and LoRA). Some Megatron configuration files expose RL-related parameters, but reinforcement-learning workflows are outside the scope of this documentation set: they are not part of the tested, supported paths described here, and this documentation does not cover how to run them.

---

## Primus ecosystem

The training component (Primus-LM) sits between the stability/platform services above it and the low-level operator libraries below it:

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

- **Primus-SaFE**: Stability and fault-tolerance oriented cluster management referenced by auxiliary tooling (not documented here).
- **Primus-LM**: Training orchestration, backends, CLI, and configurations (maintained in the [AMD-AGI/Primus](https://github.com/AMD-AGI/Primus/) repository).
- **Primus-Turbo**: High-performance operators (for example, FlashAttention-style kernels, GEMM, and collectives).

---

## Supported backends

| Backend | Typical use |
|---------|-------------|
| **Megatron-LM** | Broadest model coverage; default for many GPT-style and MoE recipes. |
| **TorchTitan** | PyTorch-native large-model training paths. |
| **JAX MaxText** | JAX/Flax training stacks aligned with MaxText. |
| **Megatron Bridge** | Post-training and bridge workflows on top of Megatron-related stacks. |
| **HummingbirdXT** | Additional integrated training path when enabled by your deployment. |

Backend choice is expressed in the configuration YAML and resolved through Primus’s adapter layer (see [glossary](./glossary.md)).

---

## Supported hardware and stack

| Item | Requirement |
|------|----------------|
| **GPUs** | AMD Instinct™ **MI300X**, **MI325X**, **MI355X** |
| **Platform** | **ROCm** (version **≥ 7.0** recommended) |
| **Container image (reference)** | `docker.io/rocm/primus:v26.4` |

Exact kernel and driver packages should match AMD’s documentation for your GPU SKU and ROCm release.

---

## Key dependencies

Primus depends on the following categories of software (the list is non-exhaustive):

| Category | Examples |
|----------|----------|
| **Framework** | PyTorch (Megatron-LM, TorchTitan paths); JAX/Flax (MaxText path). |
| **AMD stack** | ROCm, RCCL, HipBLASLt (GEMM), GPU drivers. |
| **Execution** | Docker or Podman for container mode; Slurm for cluster mode. |
| **Observability & tooling** | **loguru** (logging), **Weights & Biases** (`wandb`) and other optional trackers (see `requirements.txt`). |

Install specifics are covered in [Installation and setup](./installation.md).

---

## Runtime model

At a high level, a run follows this pipeline:

1. **YAML configuration** defines work group, modules, model preset, and overrides.
2. **`primus-cli`** selects execution mode (direct, container, or slurm) and forwards to the runner.
3. **Backend adapter** maps the resolved configuration to the target framework (Megatron-LM, TorchTitan, and so on).
4. **Distributed launch** typically uses **`torchrun`** (or the backend’s equivalent) to start workers across GPUs and nodes.

For CLI shape and options, see [Quickstart](./quickstart.md) and [CLI reference](../02-user-guide/cli-reference.md).

---

## Repository layout (top level)

The following table shows the top-level layout of the Primus project repository, [AMD-AGI/Primus](https://github.com/AMD-AGI/Primus/):

| Path | Role |
|------|------|
| `primus/` | Core library: configurations, runtime, trainers, backend adapters, tools (including preflight). |
| `runner/` | CLI implementation, helpers, hooks, and launch glue. |
| `examples/` | End-to-end example YAML and recipes per backend and GPU SKU. |
| `docs/` | Project documentation (this documentation set). |
| `tests/` | Automated tests. |
| `tools/` | Auxiliary scripts and utilities. |
| `benchmark/` | Benchmark drivers and related assets. |
| `third_party/` | Vendored or submodule dependencies. |

---

## Next steps

- [Installation and setup](./installation.md): ROCm, Docker, pip, and Slurm setup.
- [Quickstart](./quickstart.md): run a minimal training job in minutes.
- [Glossary](./glossary.md): terms used across Primus documentation.

---

## Licensing

Primus is distributed under the terms described in the project's `LICENSE` file and `README`. If you encounter differing license references between the `README` and the repository root `LICENSE` file, treat licensing as project-specific: confirm the intended terms with the maintainers and your own compliance process before redistributing.
