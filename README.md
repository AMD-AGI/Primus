# 🚀 Primus: A Lightweight, Unified Training Framework for Large Models on AMD GPUs

Primus is a flexible and high-performance training framework designed for **large-scale foundation model pretraining, fine-tuning, and reinforcement learning (RLHF)** — optimized for **AMD Instinct GPUs** and **ROCm software stack**.

---

## ✨ Key Features

- 🔧 Unified CLI to train, benchmark, and validate on any cluster
- 🧠 Supports Megatron, TorchTitan backends
- 📦 Out-of-the-box multi-node support (Slurm & containers)
- 🚀 Integrated benchmarking suite (GEMM / RCCL / end-to-end)
- ⚡ **Primus Turbo**: ROCm-optimized custom kernels with caching & JIT for maximum performance
- 🎯 ROCm-optimized for MI300/MI350 with FP8/BF16/FP16 support


## 🆕 Recent Updates

- ⚡ **Primus Turbo**: ROCm-optimized kernels with JIT compilation and caching for maximum performance (2025/09)
- 🔧 **TorchTitan backend** support with native FP8 and GraphMode (2025/06)
- 📊 **Benchmark suite** covering GEMM, RCCL, and end-to-end training performance (2025/05)
- 🛠️ **Preflight CLI** for cluster environment validation (2025/04)
- 🚀 **HipBLASLt autotuning** integrated for optimized GEMM kernels (2025/04)
- 📚 Extended model configs for **LLaMA2/3** and **DeepSeek-V3** in Megatron (2025/04)
- 🧠 **Megatron backend** support, enabling seamless integration with Primus CLI and workflows (2025/03)

👉 Full release history → [CHANGELOG.md](./docs/CHANGELOG.md)

---

## 🧩 Primus Product Matrix

|    Module    | Role | Key Features | Dependencies / Integration |
|--------------|------|--------------|-----------------------------|
| [**Primus-LM**](https://github.com/AMD-AGI/Primus)         | End-to-end training framework | - Supports multiple training backends (Megatron, TorchTitan, etc.)<br>- Provides high-performance, scalable distributed training<br>- Deeply integrates with Turbo and Safe | - Can invoke Primus-Turbo kernels and modules<br>- Runs on top of Primus-Safe for stable scheduling |
| [**Primus-Turbo**](https://github.com/AMD-AGI/Primus-Turbo)         | High-performance operators & modules | - Provides common LLM training operators (FlashAttention, GEMM, Collectives, GroupedGemm, etc.)<br>- Modular design, directly pluggable into Primus-LM<br>- Optimized for different architectures and precisions | - Built on [**AITER**](https://github.com/ROCm/aiter), [**CK**](https://github.com/ROCm/composable_kernel), [**hipBLASLt**](https://github.com/ROCm/hipBLASLt), [**Triton**](https://github.com/ROCm/triton)  and other operator libraries<br>- Can be enabled via configuration inside Primus-LM |
| **Primus-SaFE** (Coming soon)         | Stability & platform layer | - Cluster sanity check and benchmarking<br>- Kubernets scheduling with topology awareness<br>- Fault tolerance<br>- Stability enhancements | - Building a training platform based on the K8s and Slurm ecosystem |

---

## 🚀 Setup & Deployment
Primus leverages **AMD ROCm Docker images** to provide a consistent, ready-to-run environment optimized for AMD GPUs.
This avoids manual dependency setup and ensures reproducibility across clusters.


### 🔧 Prerequisites

- AMD ROCm drivers (version ≥ 6.0 recommended)
- Docker (version ≥ 24.0) with ROCm support
- ROCm-compatible AMD GPUs (e.g., Instinct MI300 series)
- Proper permissions for Docker and GPU device access

### 🐳 Quick Start with AMD ROCm Docker Image

1. **Pull the latest ROCm Megatron image**

    ```bash
    docker pull docker.io/rocm/megatron-lm:v25.8_py310
    ```

2. **Clone the Primus repository**

    ```bash
    git clone --recurse-submodules https://github.com/AMD-AIG-AIMA/Primus.git
    cd Primus
    ```

3. **Install Primus (host or container)**

    ```bash
    pip install -e .
    ```

    > 💡 Use `-e` for editable mode if you plan to modify Primus source code.

4. **Run a pretraining job (Megatron backend example)**

    ```bash
    EXP=examples/megatron/configs/llama2_7B-pretrain.yaml \
    bash ./examples/run_local_pretrain.sh
    ```

---

## 📚 Full Documentation

Looking for training guides, config templates, and deployment tips?
👉 Visit our documentation: [`docs/index.md`](./docs/index.md)
Or jump directly to [Quickstart](./docs/quickstart.md) | [CLI](./docs/cli.md) | [Benchmark](./docs/benchmark/overview.md)

---

## 🤝 Contributing

We welcome community contributions!
Start here → [Contributing Guide](./docs/contributing.md)

---

## 📜 License

Apache 2.0 License © 2025 Advanced Micro Devices, Inc.
