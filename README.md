# 🚀 Primus: A Lightweight, Unified Training Framework for Large Models on AMD GPUs

Primus is a flexible and high-performance training framework designed for **large-scale foundation model pretraining, fine-tuning, and reinforcement learning (RLHF)** — optimized for **AMD Instinct GPUs** and **ROCm software stack**.

---

## ✨ Key Features

- 🔧 Unified CLI to train, benchmark, and validate on any cluster
- 🧠 Supports Megatron, TorchTitan backends
- 📦 Out-of-the-box multi-node support (Slurm & containers)
- 🚀 Integrated benchmarking suite (GEMM / RCCL / end-to-end)
- 🎯 ROCm-optimized for MI300/MI350 with FP8/BF16/FP16 support

---

## 🆕 Recent Updates

- 🔧 Integrated **TorchTitan backend** with native FP8 and GraphMode (2025/06)
- 📊 Released **benchmark suite** for GEMM, RCCL, and end-to-end performance (2025/05)
- 🛠️ Added **Preflight CLI** for cluster environment validation (2025/04)
- 🚀 Enabled **HipBLASLt autotuning** for GEMM kernels on ROCm (2025/04)
- 📚 Extended model support: **LLaMA2/3**, **DeepSeek-V3** in Megatron configs (2025/04)
- 🧠 Introduced **Megatron trainer** with full TP/PP/EP support (2025/03)

👉 Full release history → [CHANGELOG.md](./CHANGELOG.md)

---

## 📦 Installation

```bash
git clone --recurse-submodules https://github.com/AMD-AIG-AIMA/Primus.git
cd Primus
pip install -r requirements.txt
pip install .
```

---

## ⚡ Quick Start

```bash
primus-cli direct -- train --config ./examples/configs/llama3_8B-pretrain.yaml
```

Or try benchmark:

```bash
primus-cli direct -- benchmark gemm --m 4096 --n 4096 --k 4096
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
