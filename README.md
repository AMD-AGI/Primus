# Primus

**Primus/Primus-LM** is a flexible and high-performance training framework designed for large-scale foundation model training and inference on AMD GPUs. It supports **pretraining**, **posttraining**, and **reinforcement learning** workflows with multiple backends including [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), [TorchTitan](https://github.com/pytorch/torchtitan), and [JAX MaxText](https://github.com/google/maxtext), alongside ROCm-optimized components.

> **Part of the Primus Ecosystem**: Primus-LM is the training framework layer of the [Primus ecosystem](#-primus-ecosystem), working together with [Primus-Turbo](https://github.com/AMD-AGI/Primus-Turbo) (high-performance operators) and [Primus-SaFE](https://github.com/AMD-AGI/Primus-SaFE) (stability & platform).

---

## ✨ Key Features

- **🔄 Multi-Backend Support**: Seamlessly switch between Megatron-LM, TorchTitan, and other training frameworks
- **🚀 Unified CLI**: One command interface for local development, containers, and Slurm clusters
- **⚡ ROCm Optimized**: Deep integration with AMD ROCm stack and optimized kernels from Primus-Turbo
- **📦 Production Ready**: Battle-tested on large-scale training with hundreds of GPUs
- **🔌 Extensible Architecture**: Plugin-based design for easy integration of custom models and workflows
- **🛡️ Enterprise Features**: Built-in fault tolerance, checkpoint management, and monitoring

---

## 🆕 What's New

- **[2025/11/14]** 🎉 **Primus CLI 1.0 Released** - Unified command-line interface with comprehensive documentation
- **[2025/08/22]** Primus introduction [blog](https://rocm.blogs.amd.com/software-tools-optimization/primus/README.html)
- **[2025/06/18]** Added TorchTitan backend support
- **[2025/05/16]** Added benchmark suite for performance evaluation
- **[2025/04/18]** Added [Preflight](./tools/preflight/README.md) cluster sanity checker
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
    docker pull docker.io/rocm/primus:v25.9_gfx942
    ```

2. **Clone the repository**

    ```bash
    git clone --recurse-submodules https://github.com/AMD-AIG-AIMA/Primus.git
    cd Primus
    ```

3. **Run your first training**

    ```bash
    # Run training in container
    ./runner/primus-cli container --image rocm/primus:v25.9_gfx942 \
      -- train pretrain --config examples/megatron/configs/MI300X/llama2_7B-pretrain.yaml
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

- [ ] Support for Primus-RL (training/inference modules for RLHF, OnlineDPO, GRPO, etc.)
- [ ] Add support for more model architectures and backends
- [ ] Expand documentation with more examples and tutorials

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](./CONTRIBUTING.md) for details.

## 📄 License

Primus is released under the [Apache 2.0 License](./LICENSE).

---

**Built with ❤️ by AMD AI Brain - Training at Scale (TAS) Team**
