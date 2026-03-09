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
    docker pull docker.io/rocm/primus:v26.1
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
    ./primus-cli container --image rocm/primus:v26.1 \
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
