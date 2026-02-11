# Primus Documentation

Welcome to the Primus documentation! This guide will help you get started with training large-scale foundation models on AMD GPUs.

## 📚 Documentation Structure

### 🚀 Getting Started

Start here if you're new to Primus:

- **[Quick Start Guide](./quickstart.md)** - Get up and running in 5 minutes
- **[CLI User Guide](./cli/PRIMUS-CLI-GUIDE.md)** - Complete command-line reference
- **[CLI Architecture](./cli/CLI-ARCHITECTURE.md)** - Design philosophy and deep dive

### 📖 User Guides

Guides for common workflows and features:

- **[Configuration Guide](./configuration.md)** - YAML/TOML configuration, recommended patterns, and examples
- **[Slurm & Container Usage](./slurm-container.md)** - Distributed training and containerization workflows
- **[Experiment Management](./experiments.md)** - Organizing and tracking your training runs

### 🔧 Technical References

In-depth technical documentation:

- **[Performance Projection](./projection.md)** - Project training performance to multi-node configurations
- **[Preflight](./preflight.md)** - Cluster diagnostics (host/GPU/network info + perf tests)
- **[Benchmark Suite](./benchmark.md)** - GEMM, RCCL, end-to-end benchmarks and profiling
- **[Supported Models](./backends/overview.md#supported-models)** - Supported LLM architectures and feature compatibility matrix
- **[Advanced Features](./advanced.md)** - Mixed precision, parallelism strategies, optimization techniques
- **[Backend Patch Notes](./backends/overview.md)** - Primus-specific arguments for Megatron, TorchTitan, etc.
- **[Backend Extension Guide](./backends/extending-backends.md)** - How to add a new backend using the current adapter/trainer architecture
 - **[Megatron Model Extension Guide](./backends/adding-megatron-models.md)** - How to add a new Megatron model config
 - **[TorchTitan Model Extension Guide](./backends/adding-torchtitan-models.md)** - How to add a new TorchTitan model config

### 💡 Help & Support

Get help and find answers:

- **[FAQ](./faq.md)** - Frequently asked questions and troubleshooting
- **[Examples](../examples/README.md)** - Real-world training examples and templates
- **[Preflight Tool](../primus/tools/preflight/README.md)** - Cluster sanity checker to verify environment readiness

## 🎯 Quick Navigation by Use Case

### I want to...

- **Train a model locally** → [Quick Start](./quickstart.md) + [CLI User Guide](./cli/PRIMUS-CLI-GUIDE.md)
- **Run distributed training on Slurm** → [Slurm & Container Usage](./slurm-container.md)
- **Configure my training run** → [Configuration Guide](./configuration.md)
- **Project performance to multi-node** → [Performance Projection](./projection.md)
- **Benchmark performance** → [Benchmark Suite](./benchmark.md)
- **Understand the CLI design** → [CLI Architecture](./cli/CLI-ARCHITECTURE.md)
- **Troubleshoot issues** → [FAQ](./faq.md)

## 🔗 External Resources

- [Primus-Turbo](https://github.com/AMD-AGI/Primus-Turbo) - High-performance operators & modules
- [Primus-SaFE](https://github.com/AMD-AGI/Primus-SaFE) - Stability & platform layer
- [AMD ROCm Documentation](https://rocm.docs.amd.com/)
- [TorchTitan Documentation](https://github.com/pytorch/torchtitan)

---

**Need help?** Check the [FAQ](./faq.md) or open an issue on [GitHub](https://github.com/AMD-AIG-AIMA/Primus/issues).
