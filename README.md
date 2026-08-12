# Primus

[![Primus CLI CI](https://github.com/AMD-AGI/Primus/actions/workflows/ci.yml/badge.svg)](https://github.com/AMD-AGI/Primus/actions/workflows/ci.yml)
[![Primus-CI-TAS](https://github.com/AMD-AGI/Primus/actions/workflows/ci.yaml/badge.svg)](https://github.com/AMD-AGI/Primus/actions/workflows/ci.yaml)
[![CodeQL](https://github.com/AMD-AGI/Primus/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/AMD-AGI/Primus/security/code-scanning)

**Primus/Primus-LM** is a flexible and high-performance training framework designed for large-scale foundation model training and inference on AMD GPUs. It supports **pretraining**, **posttraining**, and **reinforcement learning** workflows with multiple backends including [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), [TorchTitan](https://github.com/pytorch/torchtitan), and [JAX MaxText](https://github.com/google/maxtext), alongside ROCm-optimized components.

> **Part of the Primus Ecosystem**: Primus-LM is the training framework layer of the [Primus ecosystem](#-primus-ecosystem), working together with [Primus-Turbo](https://github.com/AMD-AGI/Primus-Turbo) (high-performance operators) and [Primus-SaFE](https://github.com/AMD-AGI/Primus-SaFE) (stability & platform).

---

## ✨ Key Features

- **🔄 Multi-Backend Support**: Seamlessly switch between Megatron-LM, TorchTitan, JAX MaxText, Megatron-Bridge, and the diffusion backend under one configuration system
- **🎯 Full Training Lifecycle**: Pretraining, SFT / LoRA post-training ([Docs](./docs/02-user-guide/posttraining.md)), and image / video diffusion training ([Docs](./docs/04-technical-guides/diffusion-models/README.md))
- **🚀 Unified CLI**: One command interface for local development, containers, and Slurm clusters ([Docs](./docs/02-user-guide/cli-reference.md))
- **⚡ ROCm Optimized**: Deep integration with AMD ROCm stack, Primus-Turbo kernels, DeepEP, fused [MegaMoE](./docs/04-technical-guides/mega-moe.md), and FP8 / MXFP8 / MXFP4 recipes
- **📐 Plan Before You Train**: [Projection](./docs/02-user-guide/projection.md) estimates parallelism and memory fit before you allocate a cluster, and the [tuning agent](./docs/02-user-guide/tuning-agent.md) searches configs automatically
- **📦 Production Ready**: Battle-tested on large-scale training with hundreds of GPUs; shipped as ROCm Docker images and a pip wheel
- **🔌 Extensible Architecture**: Patch-based backend integration for custom models and workflows ([Docs](./docs/06-developer-guide/extending-backends.md))
- **🛡️ Enterprise Features**: Built-in fault tolerance, checkpoint management, and monitoring with MLflow / TraceLens

---

## ✅ Supported Models (high level)

- **Megatron-LM**: LLaMA2 / LLaMA3.x / LLaMA4 families, DeepSeek-V2 / V3 / V4, Qwen2.5 and Qwen3 (dense and MoE), Mixtral, Grok, GPT-OSS 20B/120B, GLM, Kimi K2, MiniMax, LFM2, plus hybrid and linear-attention stacks (Mamba, Zebra-LLaMA with GDN / KDA)
- **TorchTitan**: LLaMA3.x / LLaMA4, DeepSeek-V3 (16B to 671B), and Qwen3 0.6B to 32B
- **MaxText (JAX)**: LLaMA2 / LLaMA3.x, DeepSeek-V2 16B, Mixtral-8x7B, Grok1, and Qwen3 14B / 30B-A3B (subset; see MaxText docs for details)
- **Megatron-Bridge**: SFT and LoRA post-training for Qwen3 8B/32B, LLaMA3.1 70B, Zebra-LLaMA, and Mamba
- **Diffusion**: Flux.1 (schnell / dev) text-to-image and Wan 2.1 / 2.2 text- and image-to-video

For the full and up-to-date model matrix, see [Supported Models](./docs/06-developer-guide/model-support-matrix.md).

---

## 🆕 What's New

- **[2026/07/29]** ⚡ **MegaMoE** - FlyDSL-based fused MoE layer that folds expert all-to-all into the grouped GEMMs, plus FP4 grouped GEMM support ([MegaMoE guide](./docs/04-technical-guides/mega-moe.md))
- **[2026/07/29]** Hybrid linear-attention models: Gated Delta Net (GDN) and Kimi Delta Attention (KDA) on Megatron-LM ([Hybrid models](./docs/04-technical-guides/hybrid-models/README.md))
- **[2026/07/22]** Backend upgrades: TorchTitan v0.2.2 (PyTorch 2.12) with GPT-OSS, and MaxText v26.5
- **[2026/07/17]** 🚀 **DeepSeek-V4 training support** - model definition, fused attention/MoE kernels, Muon optimizer, FP8/FP4 recipes, and a projection toolkit ([examples](./examples/deepseek-v4))
- **[2026/07/16]** 🎨 **Diffusion backend** - Flux.1 image and Wan video training with FP8/MXFP4, FSDP2, and Energon data pipelines ([Diffusion docs](./docs/04-technical-guides/diffusion-models/README.md))
- **[2026/07/14]** MLPerf Training 6.0 examples on MI355X: Llama2-70B LoRA, Llama3.1-8B, and GPT-OSS-20B ([examples](./examples/mlperf))
- **[2026/06/15]** [Tuning agent](./docs/02-user-guide/tuning-agent.md) with memory-aware benchmarking for automatic config search
- **[2026/06/08]** Primus is published as a pip wheel with a bundled `primus-cli` ([install](#install-as-a-python-package-pip))
- **[2026/01/22]** Post-training via Megatron-Bridge - SFT and LoRA workflows ([Post-training](./docs/02-user-guide/posttraining.md))
- **[2026/01/06]** MXFP4 low-precision training in the Megatron-LM backend, with MXFP8 recipes following in June
- **[2025/12/17]** MoE Training Best Practices on AMD GPUs - [MoE Package Blog](https://rocm.blogs.amd.com/software-tools-optimization/primus-moe-package/README.html)
- **[2025/11/14]** 🎉 **Primus CLI 1.0 Released** - Unified command-line interface with comprehensive documentation
- **[2025/08/22]** Primus introduction [blog](https://rocm.blogs.amd.com/software-tools-optimization/primus/README.html)

<details>
<summary>Earlier updates</summary>

- **[2025/06/18]** Added TorchTitan backend support
- **[2025/05/16]** Added benchmark suite for performance evaluation
- **[2025/04/18]** Added [Preflight](./primus/tools/preflight/README.md) cluster sanity checker
- **[2025/04/14]** Integrated HipBLASLt autotuning for optimized GPU kernel performance
- **[2025/04/09]** Extended support for LLaMA2, LLaMA3, DeepSeek-V2/V3 models
- **[2025/03/04]** Released Megatron trainer module

</details>

---

## 🚀 Setup & Deployment

Primus leverages AMD’s ROCm Docker images to provide a consistent, ready-to-run environment optimized for AMD GPUs. This eliminates manual dependency and environment configuration.  **It is recommended to use the AMD published training Docker images to run training.**

### Install as a Python package (pip)

Besides the Docker image, Primus can be installed as a wheel that bundles the `primus-cli` launcher.

```bash
# Option 1 — install directly from the Git repository (pin a tag, branch, or commit):
pip install "git+https://github.com/AMD-AGI/Primus.git@v26.2.0rc1"

# Option 2 — install a released wheel from the GitHub Pages index (recommended; pin the version):
pip install "primus==26.2.0rc1" --extra-index-url https://amd-agi.github.io/Primus/simple/
```

The wheel ships the `primus-cli` toolkit but not the heavy backend sources. Fetch the backend
sources pinned for that release (full source, including nested submodules) with:

```bash
primus-cli deps sync --dir ~/.cache/Primus/third_party
```

`primus-cli direct` also auto-runs `deps sync` on first use when the sources are missing
(set `PRIMUS_AUTO_DEPS_SYNC=0` to disable).

### Prerequisites

- AMD ROCm drivers (version ≥ 7.0 recommended)
- Docker (version ≥ 24.0) with ROCm support
- ROCm-compatible AMD GPUs (e.g., Instinct MI300 series)
- Proper permissions for Docker and GPU device access

### Quick Start with Primus CLI and AMD published training Docker images

#### Option 1: git clone this repository and run training in container (recommended)

1. **Pull the latest Docker image**

    Check the AMD published training Docker images here:

    - For Megatron-LM and TorchTitan backends: https://hub.docker.com/r/rocm/primus/tags
    - For MaxText backend: https://hub.docker.com/r/rocm/jax-training/tags

    ```bash
    # For Megatron-LM and TorchTitan backends
    docker pull rocm/primus:v26.4
    # For MaxText backend
    docker pull rocm/jax-training:maxtext-v26.4-jax0.9.1-te2.12.0
    ```

2. **Clone the repository**

    ```bash
    git clone --recurse-submodules https://github.com/AMD-AGI/Primus.git
    cd Primus
    # checkout the branch for the specific release
    git checkout release/v26.4
    git submodule update --init --recursive
    ```

3. **Run your first training**

    ```bash
    # Run training in container
    # NOTE: If your config downloads weights/tokenizer from Hugging Face Hub,
    #       you typically need to pass HF_TOKEN into the container.
    # Run in the Primus repository root directory
    ./primus-cli container --image rocm/primus:v26.4 \
      --env HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" \
      -- train pretrain --config examples/megatron/configs/MI300X/llama2_7B-BF16-pretrain.yaml
    ```

For more detailed usage instructions, see the [CLI User Guide](./docs/02-user-guide/cli-reference.md).

#### Option 2: wheel installation of Primus and run training in container

1. **Install Primus as a Python package**

    It is recommended to install Primus and other dependencies in a virtual environment.

    ```bash
    # Create a virtual environment
    python -m venv primus-env
    source primus-env/bin/activate
    # Install Primus
    pip install "primus==26.4.0" --no-deps --extra-index-url https://amd-agi.github.io/Primus/simple/

    ```

    >**Note**: this will only install the Primus CLI in your virtual environment under the `site-packages` directory, without other dependencies. The third party submodules will be downloaded on the first run. The complete dependencies and training software stack is provided in the AMD published training Docker images. You can use `primus-cli` to launch the training in container from any directory.

    >**Note**: If you don't want to use docker container to run training, and want to install the complete dependencies and training software stack on your host machine, please refer to the instruction: [Install training environment on your host machine](docs/01-getting-started/installation.md#bare-metal-host-setup). The automated installation script is under development and will be released soon.

2. **Run training in container using pip-installed Primus**

    ```bash
    primus-cli container --image rocm/primus:v26.4 \
    --env HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" \
    --volume /path/to/your/data:/data  -- --log_file /data/run.log \
    -- train pretrain --config /data/your/config.yaml
    ```

    > **Note**: The `--volume` option is used to mount the local data directory to the container. The `--log_file` option is used to save the training log to the local data directory. if `--log_file` is not specified, the training log will be saved to the primus installation directory (`site-packages/primus/logs` by default).

    If you install all the dependencies and training software stack on your host machine, you can run the training without using docker container.

    ```bash
    primus-cli direct -- train pretrain --config /path/to/your/config.yaml
    ```

---

## 📚 Documentation

Comprehensive documentation is available in the [`docs/`](./docs/) directory:

- **[Quick Start Guide](./docs/01-getting-started/quickstart.md)** - Get started in 5 minutes
- **[Primus CLI User Guide](./docs/02-user-guide/cli-reference.md)** - Complete CLI reference and usage
- **[CLI Architecture](./docs/06-developer-guide/cli-architecture.md)** - Technical design and architecture
- **[Backend Patch Notes](./docs/06-developer-guide/backend-patch-notes.md)** - Primus-specific backend arguments
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
