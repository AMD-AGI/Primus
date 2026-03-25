# Primus Documentation

Welcome to the Primus documentation! This guide will help you get started with training large-scale foundation models on AMD GPUs.

> **Comprehensive Documentation**: For the complete production documentation set, see [`production_docs/`](../production_docs/README.md). It includes configuration references, parallelism guides, environment variable documentation, and more.

## Documentation Structure

### Getting Started

Start here if you're new to Primus:

- **[Quick Start Guide](./quickstart.md)** - Get up and running in 5 minutes
- **[CLI User Guide](./cli/PRIMUS-CLI-GUIDE.md)** - Complete command-line reference
- **[CLI Architecture](./cli/CLI-ARCHITECTURE.md)** - Design philosophy and deep dive

### User Guides

Guides for common workflows and features:

- **[Configuration System](../production_docs/02-user-guide/configuration-system.md)** - YAML configuration, presets, overrides, and inheritance
- **[Deployment Guide](../production_docs/05-operations/deployment.md)** - Container, Slurm, and Kubernetes deployment

### Technical References

In-depth technical documentation:

- **[Post-Training Guide](./posttraining.md)** - Fine-tuning with SFT and LoRA using Primus CLI
- **[Performance Projection](./projection.md)** - Project training performance to multi-node configurations
- **[Preflight](./preflight.md)** - Cluster diagnostics (host/GPU/network info + perf tests)
- **[Benchmark Suite](./benchmark.md)** - GEMM, RCCL, end-to-end benchmarks and profiling
- **[Supported Models](./backends/overview.md#supported-models)** - Supported LLM architectures and feature compatibility matrix
- **[Backend Patch Notes](./backends/overview.md)** - Primus-specific arguments for Megatron, TorchTitan, etc.
- **[Backend Extension Guide](./backends/extending-backends.md)** - How to add a new backend
  - **[Megatron Model Extension Guide](./backends/adding-megatron-models.md)** - How to add a new Megatron model config
  - **[TorchTitan Model Extension Guide](./backends/adding-torchtitan-models.md)** - How to add a new TorchTitan model config

### Production Documentation

For comprehensive coverage, see the [Production Documentation](../production_docs/README.md):

- **[Configuration References](../production_docs/03-configuration-reference/megatron-parameters.md)** - Per-backend YAML parameter documentation
- **[Environment Variables](../production_docs/03-configuration-reference/environment-variables.md)** - Complete environment variable reference
- **[Parallelism Strategies](../production_docs/04-technical-guides/parallelism-strategies.md)** - Distributed training parallelism explained
- **[Performance Tuning](../production_docs/04-technical-guides/performance-tuning.md)** - HipBLASLt, Primus-Turbo, FP8, MoE optimization
- **[Troubleshooting](../production_docs/05-operations/troubleshooting.md)** - Common issues and solutions
- **[Architecture](../production_docs/06-developer-guide/architecture.md)** - System design and code architecture

### Help and Support

- **[Troubleshooting Guide](../production_docs/05-operations/troubleshooting.md)** - Common issues and solutions
- **[Examples](../examples/README.md)** - Real-world training examples and templates
- **[Preflight Tool](../primus/tools/preflight/README.md)** - Cluster sanity checker to verify environment readiness

## Quick Navigation by Use Case

### I want to...

- **Train a model locally** -- [Quick Start](./quickstart.md) + [CLI User Guide](./cli/PRIMUS-CLI-GUIDE.md)
- **Run distributed training on Slurm** -- [Deployment Guide](../production_docs/05-operations/deployment.md)
- **Configure my training run** -- [Configuration System](../production_docs/02-user-guide/configuration-system.md)
- **Look up YAML parameters** -- [Configuration References](../production_docs/03-configuration-reference/megatron-parameters.md)
- **Project performance to multi-node** -- [Performance Projection](./projection.md)
- **Benchmark performance** -- [Benchmark Suite](./benchmark.md)
- **Understand the CLI design** -- [CLI Architecture](./cli/CLI-ARCHITECTURE.md)
- **Troubleshoot issues** -- [Troubleshooting](../production_docs/05-operations/troubleshooting.md)

## External Resources

- [Primus-Turbo](https://github.com/AMD-AGI/Primus-Turbo) - High-performance operators and modules
- [Primus-SaFE](https://github.com/AMD-AGI/Primus-SaFE) - Stability and platform layer
- [AMD ROCm Documentation](https://rocm.docs.amd.com/)
- [TorchTitan Documentation](https://github.com/pytorch/torchtitan)

---

**Need help?** Open an issue on [GitHub](https://github.com/AMD-AIG-AIMA/Primus/issues).
