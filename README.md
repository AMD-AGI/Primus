# 🚀 Primus: Unified LLM Training on AMD GPUs

Primus is a flexible and high-performance training framework designed for **large-scale foundation model pretraining, fine-tuning, and reinforcement learning (RLHF)** — optimized for **AMD Instinct GPUs** and **ROCm software stack**.

---

## ✨ Key Features

- 🔧 Unified CLI to train, benchmark, and validate on any cluster
- 🧠 Supports Megatron, TorchTitan backends
- 📦 Out-of-the-box multi-node support (Slurm & containers)
- 🚀 Integrated benchmarking suite (GEMM / RCCL / end-to-end)
- 🎯 ROCm-optimized for MI300/MI350 with FP8/BF16/FP16 support

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
Or jump directly to [Quickstart](./docs/quickstart.md) | [CLI](./docs/cli.md) | [Benchmark](./docs/benchmark_overview.md)

---

## 🤝 Contributing

We welcome community contributions!
Start here → [Contributing Guide](./docs/contributing.md)

---

## 📜 License

Apache 2.0 License © 2025 Advanced Micro Devices, Inc.
