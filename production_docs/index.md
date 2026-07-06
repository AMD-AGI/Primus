# Primus Documentation

Production-grade documentation for the Primus large-scale foundation model training framework on AMD GPUs.

---

## Quick Navigation by Audience

| I am a... | Start here |
|-----------|------------|
| **New user** getting started | [Getting Started](./01-getting-started/overview.md) |
| **User** running training jobs | [User Guide](./02-user-guide/pretraining.md) |
| **User** writing YAML configs | [Configuration Reference](./03-configuration-reference/megatron-parameters.md) |
| **Engineer** tuning performance | [Technical Guides](./04-technical-guides/performance-tuning.md) |
| **Operator** deploying to production | [Operations](./05-operations/deployment.md) |
| **Contributor** to the codebase | [Developer Guide](./06-developer-guide/architecture.md) |

---

## Documentation Structure

### [01 - Getting Started](./01-getting-started/)

Start here if you are new to Primus.

- [Project Overview](./01-getting-started/overview.md) -- what Primus does, who it is for, key capabilities
- [Installation Guide](./01-getting-started/installation.md) -- prerequisites, Docker/bare-metal/Slurm setup
- [Quick Start](./01-getting-started/quickstart.md) -- first training run in 5 minutes
- [Glossary](./01-getting-started/glossary.md) -- terms, acronyms, and domain concepts

### [02 - User Guide](./02-user-guide/)

Core workflows and day-to-day usage.

- [CLI Reference](./02-user-guide/cli-reference.md) -- `primus-cli` modes, flags, and subcommands
- [Configuration System](./02-user-guide/configuration-system.md) -- YAML config model, presets, overrides, inheritance
- [Pretraining](./02-user-guide/pretraining.md) -- pretraining workflows per backend
- [Post-Training](./02-user-guide/posttraining.md) -- SFT and LoRA fine-tuning via Megatron Bridge
- [Benchmarking](./02-user-guide/benchmarking.md) -- GEMM, RCCL, and dense-GEMM benchmark suites
- [Preflight](./02-user-guide/preflight.md) -- cluster diagnostics and environment validation
- [Projection](./02-user-guide/projection.md) -- memory and performance projection tools

### [03 - Configuration Reference](./03-configuration-reference/)

Complete parameter references for each backend and all environment variables.

- [Megatron Parameters](./03-configuration-reference/megatron-parameters.md) -- all Megatron-LM backend YAML parameters
- [TorchTitan Parameters](./03-configuration-reference/torchtitan-parameters.md) -- all TorchTitan backend YAML parameters
- [MaxText Parameters](./03-configuration-reference/maxtext-parameters.md) -- all MaxText (JAX) backend YAML parameters
- [Megatron Bridge Parameters](./03-configuration-reference/megatron-bridge-parameters.md) -- all Megatron Bridge backend YAML parameters
- [Environment Variables](./03-configuration-reference/environment-variables.md) -- comprehensive reference for all environment variables

### [04 - Technical Guides](./04-technical-guides/)

Deep technical topics for advanced users.

- [Parallelism Strategies](./04-technical-guides/parallelism-strategies.md) -- DP, TP, PP, SP, CP, EP, FSDP explained
- [Parallelism Configuration](./04-technical-guides/parallelism-configuration.md) -- per-backend parallelism setup and batch size relationships
- [Collective Operations](./04-technical-guides/collective-operations.md) -- NCCL/RCCL operations and their role in each parallelism strategy
- [Performance Tuning](./04-technical-guides/performance-tuning.md) -- HipBLASLt, Primus-Turbo, FP8, MoE optimization
- [Data Preparation](./04-technical-guides/data-preparation.md) -- tokenization, data formats, mock data
- [Checkpoint Management](./04-technical-guides/checkpoint-management.md) -- formats, save/load, distributed checkpointing
- [Multi-Node Networking](./04-technical-guides/multi-node-networking.md) -- InfiniBand, RoCE, AINIC configuration

### [05 - Operations](./05-operations/)

Production deployment and operational guidance.

- [Deployment](./05-operations/deployment.md) -- container, Slurm, and Kubernetes deployment
- [Monitoring and Logging](./05-operations/monitoring-logging.md) -- WandB, TensorBoard, MLflow, Primus logging
- [Troubleshooting](./05-operations/troubleshooting.md) -- common failures, diagnostics, and fixes
- [Security](./05-operations/security.md) -- secrets handling, container security, dependencies

### [06 - Developer Guide](./06-developer-guide/)

For contributors and maintainers.

- [Architecture](./06-developer-guide/architecture.md) -- system design, runtime, backends, patch system
- [Contributing](./06-developer-guide/contributing.md) -- development setup, code style, PR process
- [Testing](./06-developer-guide/testing.md) -- test types, running tests, CI pipeline
- [Extending Backends](./06-developer-guide/extending-backends.md) -- adding new training backends
- [Adding Models](./06-developer-guide/adding-models.md) -- adding model configurations per backend
- [Model Support Matrix](./06-developer-guide/model-support-matrix.md) -- supported models per backend and GPU

### [Appendix](./appendix/)

- [Documentation Gaps](./appendix/gaps-and-verification.md) -- items needing maintainer verification

---

## Quick Use-Case Navigation

### I want to...

| Goal | Document |
|------|----------|
| Understand what Primus is | [Overview](./01-getting-started/overview.md) |
| Install Primus | [Installation](./01-getting-started/installation.md) |
| Run my first training | [Quick Start](./01-getting-started/quickstart.md) |
| Write a training YAML config | [Configuration System](./02-user-guide/configuration-system.md) |
| Look up a Megatron parameter | [Megatron Parameters](./03-configuration-reference/megatron-parameters.md) |
| Look up a TorchTitan parameter | [TorchTitan Parameters](./03-configuration-reference/torchtitan-parameters.md) |
| Look up an environment variable | [Environment Variables](./03-configuration-reference/environment-variables.md) |
| Understand parallelism strategies | [Parallelism Strategies](./04-technical-guides/parallelism-strategies.md) |
| Configure parallelism for my model | [Parallelism Configuration](./04-technical-guides/parallelism-configuration.md) |
| Tune training performance | [Performance Tuning](./04-technical-guides/performance-tuning.md) |
| Prepare training data | [Data Preparation](./04-technical-guides/data-preparation.md) |
| Deploy to a Slurm cluster | [Deployment](./05-operations/deployment.md) |
| Debug a training failure | [Troubleshooting](./05-operations/troubleshooting.md) |
| Contribute to Primus | [Contributing](./06-developer-guide/contributing.md) |
| Understand the code architecture | [Architecture](./06-developer-guide/architecture.md) |
| Add a new training backend | [Extending Backends](./06-developer-guide/extending-backends.md) |

---

## External Resources

- [Primus-Turbo](https://github.com/AMD-AGI/Primus-Turbo) -- high-performance operators and kernels
- [Primus-SaFE](https://github.com/AMD-AGI/Primus-SaFE) -- stability and platform layer
- [AMD ROCm Documentation](https://rocm.docs.amd.com/)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [TorchTitan](https://github.com/pytorch/torchtitan)
- [MaxText](https://github.com/google/maxtext)

---

**Need help?** Open an issue on [GitHub](https://github.com/AMD-AIG-AIMA/Primus/issues).
