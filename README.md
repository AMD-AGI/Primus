# Primus

**Primus** is a flexible and high-performance training framework designed for large-scale foundation model training and inference. It is designed to support **pretraining**, **posttraining**, and **reinforcement learning** workflows, and is compatible with multiple backends including [Megatron](https://github.com/NVIDIA/Megatron-LM) and ROCm-optimized components.

---

## 🆕 What's New

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

- AMD ROCm drivers (version ≥ 6.4 recommended)
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
- **[Full Documentation Index](./docs/README.md)** - Browse all available documentation

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
