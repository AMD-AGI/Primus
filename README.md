# Primus

**Primus/Primus-LM** is a flexible and high-performance training framework designed for large-scale foundation model training and inference on AMD GPUs. It supports **pretraining**, **posttraining**, and **reinforcement learning** workflows with multiple backends including [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), [TorchTitan](https://github.com/pytorch/torchtitan), and [JAX MaxText](https://github.com/google/maxtext), alongside ROCm-optimized components.

> **Part of the Primus Ecosystem**: Primus-LM is the training framework layer of the [Primus ecosystem](#-primus-ecosystem), working together with [Primus-Turbo](https://github.com/AMD-AGI/Primus-Turbo) (high-performance operators) and [Primus-SaFE](https://github.com/AMD-AGI/Primus-SaFE) (stability & platform).

---

## ✨ Key Features

- **🔄 Multi-Backend Support**: Seamlessly switch between Megatron-LM, TorchTitan, and other training frameworks
- **🚀 Unified CLI**: One command interface for local development, containers, and Slurm clusters ([Docs](./docs/README.md))
- **⚡ ROCm Optimized**: Deep integration with AMD ROCm stack and optimized kernels from Primus-Turbo
- **📦 Production Ready**: Battle-tested on large-scale training with hundreds of GPUs
- **🔌 Extensible Architecture**: Plugin-based design for easy integration of custom models and workflows
- **🛡️ Enterprise Features**: Built-in fault tolerance, checkpoint management, and monitoring

---

## ✅ Supported Models (high level)

- **Megatron-LM**: LLaMA2 / LLaMA3 / LLaMA4 families, DeepSeek-V2/V3, Mixtral-style MoE, and other GPT-style models
- **TorchTitan**: LLaMA3 / LLaMA4, DeepSeek-V3, and related decoder-only architectures
- **MaxText (JAX)**: LLaMA3.x and other MaxText-supported transformer models (subset; see MaxText docs for details)

For the full and up-to-date model matrix, see [Supported Models](./docs/backends/overview.md#supported-models).

---

## 🆕 What's New

- **[2025/12/17]** MoE Training Best Practices on AMD GPUs - [MoE Package Blog](https://rocm.blogs.amd.com/software-tools-optimization/primus-moe-package/README.html)
- **[2025/11/14]** 🎉 **Primus CLI 1.0 Released** - Unified command-line interface with comprehensive documentation
- **[2025/08/22]** Primus introduction [blog](https://rocm.blogs.amd.com/software-tools-optimization/primus/README.html)
- **[2025/06/18]** Added TorchTitan backend support
- **[2025/05/16]** Added benchmark suite for performance evaluation
- **[2025/04/18]** Added [Preflight](./primus/tools/preflight/README.md) cluster sanity checker
- **[2025/04/14]** Integrated HipBLASLt autotuning for optimized GPU kernel performance
- **[2025/04/09]** Extended support for LLaMA2, LLaMA3, DeepSeek-V2/V3 models
- **[2025/03/04]** Released Megatron trainer module

---

## 🚀 Setup & Deployment

Primus leverages AMD’s ROCm Docker images to provide a consistent, ready-to-run environment optimized for AMD GPUs. This eliminates manual dependency and environment configuration.

### Prerequisites

- AMD ROCm drivers (version ≥ 7.0 recommended)
- Docker (version ≥ 24.0) with ROCm support
- ROCm-compatible AMD GPUs (e.g., Instinct MI300 series)
- Proper permissions for Docker and GPU device access


### Quick Start with Primus CLI

1. **Pull the latest Docker image**

    ```bash
    docker pull docker.io/rocm/primus:v26.2
    ```

2. **Clone the repository**

    ```bash
    git clone --recurse-submodules https://github.com/AMD-AIG-AIMA/Primus.git
    cd Primus
    ```

3. **Run your first training**

    ```bash
    # Run training in container
    # NOTE: If your config downloads weights/tokenizer from Hugging Face Hub,
    #       you typically need to pass HF_TOKEN into the container.
    ./primus-cli container --image rocm/primus:v26.2 \
      --env HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" \
      -- train pretrain --config examples/megatron/configs/MI300X/llama2_7B-BF16-pretrain.yaml
    ```

For more detailed usage instructions, see the [CLI User Guide](./docs/cli/PRIMUS-CLI-GUIDE.md).

---

## 📚 Documentation

Comprehensive documentation is available in the [`docs/`](./docs/) directory:

- **[Quick Start Guide](./docs/quickstart.md)** - Get started in 5 minutes
- **[Primus CLI User Guide](./docs/cli/PRIMUS-CLI-GUIDE.md)** - Complete CLI reference and usage
- **[CLI Architecture](./docs/cli/CLI-ARCHITECTURE.md)** - Technical design and architecture
- **[Backend Patch Notes](./docs/backends/overview.md)** - Primus-specific backend arguments
- **[Full Documentation Index](./docs/README.md)** - Browse all available documentation

---

## 🌐 Primus Ecosystem

Primus-LM is part of a comprehensive ecosystem designed to provide end-to-end solutions for large model training on AMD GPUs:

### 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                   Primus-SaFE                       │
│         (Stability & Platform Layer)                │
│   Cluster Management | Fault Tolerance | Scheduling │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│                   Primus-LM                         │
│              (Training Framework)                   │
│    Megatron | TorchTitan | Unified CLI | Workflows  │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│                  Primus-Turbo                       │
│           (High-Performance Operators)              │
│  FlashAttention | GEMM | Collectives | GroupedGemm  │
│        AITER | CK | hipBLASLt | Triton              │
└─────────────────────────────────────────────────────┘
```

### 📦 Component Details

| Component | Role | Key Features | Repository |
|-----------|------|--------------|------------|
| **Primus (Primus-LM)** | Training Framework | Multi-backend support, unified CLI, production-ready workflows | [This repo](https://github.com/AMD-AGI/Primus) |
| **Primus-Turbo** | Performance Layer | Optimized kernels for attention, GEMM, communication, and more | [Primus-Turbo](https://github.com/AMD-AGI/Primus-Turbo) |
| **Primus-SaFE** | Platform Layer | Cluster orchestration, fault tolerance, topology-aware scheduling | [Primus-SaFE](https://github.com/AMD-AGI/Primus-SaFE) |

### 🔗 How They Work Together

1. **Primus-LM** provides the training framework and workflow orchestration
2. **Primus-Turbo** supplies highly optimized compute kernels for maximum performance
3. **Primus-SaFE** ensures stability and efficient resource utilization at scale

This separation of concerns allows each component to evolve independently while maintaining seamless integration.

---

## 📝 TODOs

- [ ] Add support for more model architectures and backends
- [ ] Expand documentation with more examples and tutorials

---

## 🙏 Upstream Optimizations

Primus builds on top of several ROCm-native operator libraries and compiler projects—we couldn’t reach current performance levels without them:

- [ROCm AITer](https://github.com/ROCm/aiter) – AI Tensor Engine kernels (elementwise, attention, KV-cache, fused MoE, etc.)
- [Composable Kernel](https://github.com/ROCm/composable_kernel) – performance-portable tensor operator generator for GEMM and convolutions
- [hipBLASLt](https://github.com/ROCm/hipBLASLt) – low-level BLAS Lt API with autotuning support for ROCm GPUs
- [ROCm Triton](https://github.com/ROCm/triton) – Python-first kernel compiler used for custom attention and MoE paths

If you rely on Primus, please consider starring or contributing to these projects as well—they are foundational to our stack.

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](./CONTRIBUTING.md) for details.

## 📄 License

Primus is released under the [Apache 2.0 License](./LICENSE).

---

**Built with ❤️ by AMD AI Brain - Training at Scale (TAS) Team**



I mainly work on Primus, AMD’s training framework for large-scale model training on AMD GPUs.

My work covers both framework development and performance optimization. On the framework side, I help build the core Primus components, such as the unified training entry, configuration system, backend integration, patching mechanism, preflight checks, benchmark tools, and Slurm/container launch flow.

On the optimization side, I focus on making training backends like Megatron and TorchTitan run better on AMD GPUs. This includes MoE training, FP8, distributed training, GEMM and RCCL benchmarking, communication-computation overlap, pipeline parallelism, and expert parallelism.

In short, my job is to make Primus easier to use, more reliable, and faster, so internal teams and customers can train large models more efficiently on AMD GPUs.

What I've been spending most of my time on lately. I've added a set of Cursor-agent skills that automate per-config performance tuning end-to-end. The goal is simple: **cut the engineer time spent on PoC tuning**, and make the tuning process itself controllable and reproducible — instead of relying on each engineer's tribal knowledge.

I'm a Primus engineer at AMD. On the **framework side** I help build the core components — unified CLI, config system, backend patching, preflight checks, benchmark suite, Slurm/container launch flow. On the **optimization side** I focus on making Megatron and TorchTitan run faster on AMD GPUs: MoE, FP8, distributed training, comm-compute overlap, pipeline / expert parallelism. **Most recently** I've been adding a set of Cursor-agent skills that automate per-config performance tuning end-to-end — to cut PoC tuning time and make the process controllable and reproducible, instead of relying on per-engineer tribal knowledge.


### A bit more detail

**Framework side** — concrete pieces I've contributed to:

- **`primus-cli` 1.0** (released Nov 2025): one CLI that drives local development, container runs, and Slurm cluster launches from the same command and the same YAML. Eliminates the per-environment shell-script sprawl users had before.
- **Configuration system + `primus-defaults`**: YAML-driven config with reusable feature bundles (Primus-Turbo, DeepEP, rope-fusion, bf16-precision-aware-optimizer, gradient-accumulation-fusion, …) that users opt into per-model, instead of re-deriving the flag combination every time.
- **Backend patching layer**: runtime patches applied at import time, keeping upstream Megatron-LM / TorchTitan / MaxText repositories unmodified. An upstream rebase does not force us to re-fork.
- **Preflight cluster sanity checker**: verifies idle SLURM nodes before training — Docker daemon health, NIC QoS / DCQCN, RDMA link state, GID table, RCCL all-reduce throughput — so bad nodes are caught up front instead of failing 20 minutes into a multi-node run.
- **Benchmark suite + HipBLASLt autotuning integration**: standardized perf-evaluation harness for GEMM, RCCL collectives, and end-to-end throughput regressions; HipBLASLt autotune wired into the training-time GEMM path.
- **Slurm + container launch flow**: container mount / env / `srun` plumbing, plus a developer-mode workflow that attaches to an existing SLURM allocation without manual pod bring-up.

**Optimization side** — what I work on per topic:

- **MoE on MI300X** (DeepSeek-V2 / V3, Mixtral): DeepEP integration, `turbo_deepep_num_cu` tuning per EP size (64–80 CUs is the sweet spot at ep=8), and the trade-off between DeepEP and the alltoall-dispatcher + `moe_shared_expert_overlap` path on single- vs multi-node shapes.
- **FP8**: Primus-Turbo FP8 GEMM / attention rollout across Megatron and TorchTitan, plus loss-stability regression coverage so FP8 doesn't silently break long runs.
- **Distributed training**: full DP / TP / PP / EP / CP / VPP knob coverage, legal `mbs` / `gbs` combinations, pipeline-bubble accounting.
- **GEMM + RCCL benchmarking**: hipBLASLt autotuning, RCCL collective profiling, kernel-time-based diagnosis of comm-bound vs compute-bound regimes.
- **Comm-compute overlap**: `overlap_grad_reduce`, `overlap_param_gather`, `turbo_deepep_use_comm_stream` — characterized that these matter most on multi-node; on single-node xGMI keeps `comm_ratio ≈ 0` so overlap features are flat there.
- **Per-feature integrations**: rope-fusion (which alone delivered +3.5 % on DeepSeek V2 Lite), grad-acc-fusion, bf16-precision-aware-optimizer (saves ~5 GB of optimizer state at near-zero throughput cost).

**Recent focus: Primus Pilot — agent-driven proactive tuning** (`pilot/`)

What I've been spending most of my time on lately. I've added a set of Cursor-agent skills that automate per-config performance tuning end-to-end. The goal is simple: **cut the engineer time spent on PoC tuning**, and make the tuning process itself controllable and reproducible — instead of relying on each engineer's tribal knowledge.

**Recent: Backend-gap dashboard** (`tools/backend_gap_report/`)

Shared engineering dashboard that tracks Megatron-LM / TorchTitan / Primus-Turbo drift against upstream and integrates weekly engineering reports into the same site. Each backend gap report is reproducible from git facts (commit gap, merge-base, dependency-file diff) instead of ad-hoc notes.
