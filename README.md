# Primus

**Primus** is a flexible and high-performance training framework designed for large-scale foundation model training and inference. It is designed to support **pretraining**, **posttraining**, and **reinforcement learning** workflows, and is compatible with multiple backends including [Megatron](https://github.com/NVIDIA/Megatron-LM) and ROCm-optimized components.

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

👉 Full release history → [CHANGELOG.md](./CHANGELOG.md)

---


## 🚀 Setup & Deployment

Primus leverages AMD’s ROCm Docker images to provide a consistent, ready-to-run environment optimized for AMD GPUs. This eliminates manual dependency and environment configuration.

### Prerequisites

- AMD ROCm drivers (version ≥ 6.0 recommended)
- Docker (version ≥ 24.0) with ROCm support
- ROCm-compatible AMD GPUs (e.g., Instinct MI300 series)
- Proper permissions for Docker and GPU device access

## 🐳 Quick Start with AMD ROCm Docker Image

1. **Pull the latest ROCm Megatron image**

    ```bash
    docker pull docker.io/rocm/megatron-lm:v25.8_py310
    ```

2. **Clone the Primus repository**

    ```bash
    git clone --recurse-submodules https://github.com/AMD-AIG-AIMA/Primus.git
    cd Primus
    ```

3. **Install dependencies (host or container)**

    ```bash
    pip install -e .
    ```

4. **Run pretraining (Megatron backend example)**

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
