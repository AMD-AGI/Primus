# Primus Documentation

> **Primus** is a unified, high-performance training framework designed for large-scale foundation model training and inference on AMD GPUs.

---

## 🚀 Overview

Primus aims to provide a flexible, robust, and user-friendly entry point for LLM training and evaluation, with first-class support for distributed multi-node environments and advanced parallelism. The framework integrates industry best practices from open-source ecosystems (e.g., Megatron-LM, TorchTitan) and is optimized for ROCm/MI300/MI350 platforms.

---

## 🌟 Key Features

- **Unified CLI**: One entry point for training, benchmarking, and environment checking—single node or large-scale clusters.
- **Multi-Backend Support**: Seamless switching between Megatron, TorchTitan, and future backends via unified configuration.
- **Distributed & Scalable**: Native Slurm orchestration, multi-node/multi-GPU support, automatic environment setup.
- **Benchmark & Profiling Suite**: Built-in operator-level and end-to-end benchmarking tools for AMD platforms.
- **Flexible Configuration**: YAML/TOML-driven experiments, support for custom models, optimizer settings, and data loaders.
- **Best-in-Class AMD ROCm Optimization**: Tailored for MI300/MI350 and next-gen ROCm software stacks.

---

## 📖 Documentation Structure

- [Quickstart](./quickstart.md): Step-by-step guide to run your first Primus job
- [CLI Reference](./cli.md): Detailed reference manual for the unified command-line interface
- [Slurm & Container Usage](./usage/slurm_container.md): Distributed and container-based workflow
- [Experiment Configuration](./config/overview.md): YAML/TOML config, recommended patterns, and examples
- [Benchmark Suite](./benchmark/overview.md): GEMM, RCCL, end-to-end benchmarks and profiling
- [Supported Models](./models.md): Supported LLM architectures and feature matrix
- [Advanced Features](./advanced.md): Mixed precision, parallelism, optimization tricks
- [FAQ](./faq.md): Frequently asked questions and troubleshooting

---

## 🏗️ The Three Pillars of Primus

**Train / Preflight / Benchmark**
_Primus delivers a reproducible, high-performance LLM workflow with three core pillars:_

1. **Train**: Unified training and evaluation for large-scale models, with strong backend support.
2. **Preflight**: Cluster/environment sanity check and readiness verification—catch config, env, and cluster issues before launch.
3. **Benchmark**: Operator-level and end-to-end performance baselining, regression tracking, and hardware feature visualization.

> *“One CLI, one ecosystem, three pillars—aiming to make AMD an outstanding platform for LLM training.”*

---

## 🛠️ Contributing & Community

- [Contributing Guide](./contributing.md)
- [Report an Issue](https://github.com/amd/primus/issues)
- [Discussions](https://github.com/amd/primus/discussions)

Primus is an open-source project. We welcome feedback, issues, and contributions from the community!

---

## 📢 Stay Up To Date

- Latest news and releases: [Primus GitHub](https://github.com/amd/primus)
- ROCm blog series: [ROCm Official Blog](https://rocm.blogs.amd.com/)

---

_© 2025 Advanced Micro Devices, Inc. All rights reserved._
