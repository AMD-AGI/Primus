# Primus

**Primus** is a flexible and high-performance training framework designed for large-scale foundation model training and inference on AMD GPUs. It supports **pretraining**, **posttraining**, and **reinforcement learning** workflows, and is compatible with multiple backends including [Megatron](https://github.com/NVIDIA/Megatron-LM) and [TorchTitan](https://github.com/pytorch/torchtitan).

## ✨ Key Features

- **🎯 Unified CLI**: Single command interface for all training workflows - from local development to multi-node production
- **🐳 Flexible Execution**: Seamlessly switch between direct, container, and Slurm execution modes
- **🚀 High Performance**: Optimized for AMD Instinct GPUs with ROCm, integrated with Primus-Turbo operators
- **📦 Multiple Backends**: Support for Megatron, TorchTitan, and more training frameworks
- **🔌 Extensible Design**: Plugin-based architecture for easy customization and extension
- **🔍 Smart Environment**: Auto-detect GPU models and apply optimized configurations
- **📊 Comprehensive Benchmarking**: Built-in GEMM, RCCL, and end-to-end performance evaluation

---

## 🆕 What's New

- **[2025/11/14]** 🎉 **Primus CLI 1.0 Released** - Unified command-line interface for training, benchmarking, and cluster management. See [CLI Documentation](./docs/cli/).
- **[2025/08/22]** Primus introduction [blog](https://rocm.blogs.amd.com/software-tools-optimization/primus/README.html).
- **[2025/06/18]** Added TorchTitan backend support.
- **[2025/05/16]** Added benchmark suite for performance evaluation across models and hardware.
- **[2025/04/18]** Added [Preflight](./tools/preflight/README.md) cluster sanity checker to verify environment readiness.
- **[2025/04/14]** Integrated HipblasLT autotuning for optimized GPU kernel performance.
- **[2025/04/09]** Extended support for LLaMA2, LLaMA3, DeepSeek-V2/V3 models in [Megatron model configs](https://github.com/AMD-AIG-AIMA/Primus/tree/main/primus/configs/models/megatron).
- **[2025/03/04]** Released Megatron trainer module for flexible and efficient large model training.

---

## 🧩 Primus Product Matrix

|    Module    | Role | Key Features | Dependencies / Integration |
|--------------|------|--------------|-----------------------------|
| [**Primus-LM**](https://github.com/AMD-AGI/Primus)         | End-to-end training framework | - Supports multiple training backends (Megatron, TorchTitan, etc.)<br>- Provides high-performance, scalable distributed training<br>- Deeply integrates with Primus-Turbo and Primus-SaFE | - Can invoke Primus-Turbo kernels and modules<br>- Runs on top of Primus-SaFE for stable scheduling |
| [**Primus-Turbo**](https://github.com/AMD-AGI/Primus-Turbo)         | High-performance operators & modules | - Provides common LLM training operators (FlashAttention, GEMM, Collectives, GroupedGemm, etc.)<br>- Modular design, directly pluggable into Primus-LM<br>- Optimized for different architectures and precisions | - Built on [**AITER**](https://github.com/ROCm/aiter), [**CK**](https://github.com/ROCm/composable_kernel), [**hipBLASLt**](https://github.com/ROCm/hipBLASLt), [**Triton**](https://github.com/ROCm/triton)  and other operator libraries<br>- Can be enabled via configuration inside Primus-LM |
| [**Primus-SaFE**](https://github.com/AMD-AGI/Primus-SaFE)         | Stability & platform layer | - Cluster sanity check and benchmarking<br>- Kubernets scheduling with topology awareness<br>- Fault tolerance<br>- Stability enhancements | - Building a training platform based on the K8s and Slurm ecosystem |

---

## 🚀 Setup & Deployment

Primus leverages AMD’s ROCm Docker images to provide a consistent, ready-to-run environment optimized for AMD GPUs. This eliminates manual dependency and environment configuration.

### Prerequisites

- AMD ROCm drivers (version ≥ 6.0 recommended)
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

3. **Run your first training** (choose one method)

    **Option A: Using Primus CLI (Recommended)**
    ```bash
    # Local development - direct execution
    primus-cli direct -- train pretrain --config examples/megatron/configs/MI300X/llama2_7B-pretrain.yaml

    # Or with container isolation
    primus-cli container --image rocm/primus:v25.9_gfx942 \
      -- train pretrain --config examples/megatron/configs/MI300X/llama2_7B-pretrain.yaml

    # Or on Slurm cluster (8 nodes)
    primus-cli slurm srun -N 8 -p gpu \
      -- train pretrain --config examples/megatron/configs/MI300X/llama2_7B-pretrain.yaml
    ```

    **Option B: Using legacy scripts**
    ```bash
    pip install -r requirements.txt
    EXP=examples/megatron/configs/MI300X/llama2_7B-pretrain.yaml bash ./examples/run_local_pretrain.sh
    ```

For more detailed usage instructions, see the [CLI User Guide](./docs/cli/PRIMUS-CLI-GUIDE.md) or [examples/README.md](./examples/README.md).

---

## 📚 Documentation

Primus provides comprehensive documentation to help you get started and master the framework:

### 🚀 Getting Started

- **[Quick Start Guide](./docs/quickstart.md)** - Get up and running in 5 minutes
- **[CLI User Guide](./docs/cli/PRIMUS-CLI-GUIDE.md)** ([中文版](./docs/cli/PRIMUS-CLI-GUIDE_CN.md)) - Complete command-line reference
- **[CLI Architecture](./docs/cli/CLI-ARCHITECTURE.md)** ([中文版](./docs/cli/CLI-ARCHITECTURE_CN.md)) - Design philosophy and internals

### 📖 User Guides

- **[Configuration Guide](./docs/configuration.md)** - YAML configuration, patterns, and best practices
- **[Slurm & Container Usage](./docs/slurm-container.md)** - Distributed training and containerization
- **[Experiment Management](./docs/experiments.md)** - Organizing and tracking your training runs

### 🔧 Technical References

- **[Benchmark Suite](./docs/benchmark.md)** - GEMM, RCCL, and end-to-end performance testing
- **[Supported Models](./docs/models.md)** - LLM architectures and feature compatibility matrix
- **[Advanced Features](./docs/advanced.md)** - Mixed precision, parallelism strategies, optimization techniques

### 💡 Help & Support

- **[FAQ](./docs/faq.md)** - Frequently asked questions and troubleshooting
- **[Examples](./examples/README.md)** - Real-world training examples and templates
- **[Preflight Tool](./tools/preflight/README.md)** - Cluster health check and validation

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

## 🙏 Acknowledgments

Primus builds upon the excellent work of:
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) by NVIDIA
- [TorchTitan](https://github.com/pytorch/torchtitan) by PyTorch Team
- [ROCm](https://github.com/ROCm) by AMD
- And many other open-source projects in the ML community

---

**Built with ❤️ by AMD AI Brain - Training at Scale (TAS) Team**
