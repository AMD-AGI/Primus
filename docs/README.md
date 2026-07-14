# Primus documentation

Production documentation for **Primus**, a large-scale foundation model training framework for AMD GPUs.

---

## Choose your starting point

| I am a... | Start here |
|-----------|------------|
| **New user** | [Getting started](./01-getting-started/overview.md) |
| **User** running training jobs | [User guide](./02-user-guide/pretraining.md) |
| **User** writing YAML configurations | [Configuration reference](./03-configuration-reference/megatron-parameters.md) |
| **Engineer** tuning performance | [Technical guides](./04-technical-guides/performance-tuning.md) |
| **Operator** deploying to production | [Operations](./05-operations/deployment.md) |
| **Contributor** to the codebase | [Developer guide](./06-developer-guide/architecture.md) |

---

## Documentation structure

### [Getting started](./01-getting-started/)

Start here if you are new to Primus.

- [Project overview](./01-getting-started/overview.md): what Primus does, who it is for, key capabilities
- [Installation guide](./01-getting-started/installation.md): prerequisites, Docker/bare-metal/Slurm setup
- [Quickstart](./01-getting-started/quickstart.md): first training run in 5 minutes
- [Glossary](./01-getting-started/glossary.md): terms, acronyms, and domain concepts

### [User guide](./02-user-guide/)

Core workflows and day-to-day usage.

- [CLI reference](./02-user-guide/cli-reference.md): `primus-cli` modes, flags, and subcommands
- [Configuration system](./02-user-guide/configuration-system.md): YAML configuration model, presets, overrides, inheritance
- [Pretraining](./02-user-guide/pretraining.md): pretraining **concepts**: backends, YAML structure, parallelism, configuration inventory
- [Backend training recipes](./02-user-guide/training-recipes.md): pretraining **commands**: copy-paste, GPU-arch-specific run commands
- [Post-training](./02-user-guide/posttraining.md): SFT and LoRA fine-tuning via Megatron Bridge
- [Benchmarking](./02-user-guide/benchmarking.md): GEMM, RCCL, and dense-GEMM benchmark suites
- [Preflight](./02-user-guide/preflight.md): cluster diagnostics and environment validation
- [Projection](./02-user-guide/projection.md): memory and performance projection tools
- [Tuning agent](./02-user-guide/tuning-agent.md): LLM-driven search for an optimal training configuration (uses projection as an oracle)
- [Primus tools](./02-user-guide/primus-tools.md): catalog of all Primus tools and ecosystem projects with how-to starting points

### [Configuration reference](./03-configuration-reference/)

Parameter references for Primus presets, backend-facing keys, and commonly used environment variables.

- [Megatron parameters](./03-configuration-reference/megatron-parameters.md): Megatron-LM backend YAML parameters and Primus overrides
- [TorchTitan parameters](./03-configuration-reference/torchtitan-parameters.md): Primus TorchTitan preset keys and common JobConfig fields
- [MaxText parameters](./03-configuration-reference/maxtext-parameters.md): Primus MaxText overlay defaults and common fields
- [Megatron Bridge parameters](./03-configuration-reference/megatron-bridge-parameters.md): Megatron Bridge recipe, SFT, and pretraining fields surfaced through Primus
- [Environment variables](./03-configuration-reference/environment-variables.md): practical reference for commonly encountered environment variables

### [Technical guides](./04-technical-guides/)

Deep technical topics for advanced users.

- [Parallelism strategies](./04-technical-guides/parallelism-strategies.md): DP, TP, PP, SP, CP, EP, FSDP explained
- [Parallelism configuration](./04-technical-guides/parallelism-configuration.md): per-backend parallelism setup and batch size relationships
- [Collective operations](./04-technical-guides/collective-operations.md): NCCL/RCCL operations and their role in each parallelism strategy
- [Performance tuning](./04-technical-guides/performance-tuning.md): HipBLASLt, Primus-Turbo, FP8, MoE optimization
- [MoE training deep-dive](./04-technical-guides/moe-training.md): bottlenecks and Primus-Turbo optimizations for Mixture-of-Experts models
- [Data preparation](./04-technical-guides/data-preparation.md): tokenization, data formats, mock data
- [Checkpoint management](./04-technical-guides/checkpoint-management.md): formats, save/load, distributed checkpointing
- [Multi-node networking](./04-technical-guides/multi-node-networking.md): InfiniBand, RoCE, AINIC configuration
- [Profiling and observability](./04-technical-guides/profiling-and-observability.md): Torch profiler, TraceLens, memory snapshots, projection, pp_vis
- [Logging and experiment tracking](./04-technical-guides/logging-and-experiment-tracking.md): TensorBoard, WandB, MLflow setup per backend
- [Fault tolerance and elastic training](./04-technical-guides/fault-tolerance-and-elastic-training.md): graceful exit, auto-resume, in-process restart, torchft
- [Determinism and reproducibility](./04-technical-guides/determinism-and-reproducibility.md): deterministic mode, seeds, trade-offs
- [Diffusion models](./04-technical-guides/diffusion-models/README.md): Flux diffusion architecture, data pipeline, and FP8 / MXFP4 training
- [Native SFT and LoRA](./04-technical-guides/native-sft-lora.md): Megatron-native SFT/LoRA runbook (BF16 / FP8 / FP4), no Megatron-Bridge dependency

### [Operations](./05-operations/)

Production deployment and operational guidance.

- [Deployment](./05-operations/deployment.md): container, Slurm, and Kubernetes deployment
- [Monitoring and logging](./05-operations/monitoring-logging.md): WandB, TensorBoard, MLflow, Primus logging
- [Troubleshooting](./05-operations/troubleshooting.md): common failures, diagnostics, and fixes
- [Security](./05-operations/security.md): secrets handling, container security, dependencies

### [Developer guide](./06-developer-guide/)

For contributors and maintainers.

- [Architecture](./06-developer-guide/architecture.md): system design, runtime, backends, patch system
- [Contributing](./06-developer-guide/contributing.md): development setup, code style, PR process
- [Testing](./06-developer-guide/testing.md): test types, running tests, CI pipeline
- [Extending backends](./06-developer-guide/extending-backends.md): adding new training backends
- [Adding models](./06-developer-guide/adding-models.md): adding model configurations per backend
- [Model support matrix](./06-developer-guide/model-support-matrix.md): supported models per backend and GPU
- [CLI architecture](./06-developer-guide/cli-architecture.md): CLI internals: subcommand discovery, dispatch, and launch wrappers
- [Backend patch notes](./06-developer-guide/backend-patch-notes.md): Primus-specific backend arguments and the files they patch
- [Tooling](./06-developer-guide/tooling.md): auxiliary analysis, benchmarking, visualization, and diagnostics tools

---

## Common use cases

### I want to...

| Goal | Document |
|------|----------|
| Understand what Primus is | [Overview](./01-getting-started/overview.md) |
| Browse all Primus tools | [Primus tools](./02-user-guide/primus-tools.md) |
| Install Primus | [Installation](./01-getting-started/installation.md) |
| Run my first training | [Quickstart](./01-getting-started/quickstart.md) |
| Get an exact run command for my model/GPU | [Backend training recipes](./02-user-guide/training-recipes.md) |
| Write a training YAML configuration | [Configuration system](./02-user-guide/configuration-system.md) |
| Look up a Megatron parameter | [Megatron parameters](./03-configuration-reference/megatron-parameters.md) |
| Look up a TorchTitan parameter | [TorchTitan parameters](./03-configuration-reference/torchtitan-parameters.md) |
| Look up an environment variable | [Environment variables](./03-configuration-reference/environment-variables.md) |
| Understand parallelism strategies | [Parallelism strategies](./04-technical-guides/parallelism-strategies.md) |
| Configure parallelism for my model | [Parallelism configuration](./04-technical-guides/parallelism-configuration.md) |
| Tune training performance | [Performance tuning](./04-technical-guides/performance-tuning.md) |
| Train a Mixture-of-Experts model | [MoE training deep-dive](./04-technical-guides/moe-training.md) |
| Train a diffusion (Flux) model | [Diffusion models](./04-technical-guides/diffusion-models/README.md) |
| Fine-tune with native SFT / LoRA | [Native SFT and LoRA](./04-technical-guides/native-sft-lora.md) |
| Auto-tune my training configuration | [Tuning agent](./02-user-guide/tuning-agent.md) |
| Profile a training run | [Profiling and observability](./04-technical-guides/profiling-and-observability.md) |
| Track experiments (WandB/MLflow/TensorBoard) | [Logging and experiment tracking](./04-technical-guides/logging-and-experiment-tracking.md) |
| Survive node failures on long runs | [Fault tolerance and elastic training](./04-technical-guides/fault-tolerance-and-elastic-training.md) |
| Reproduce results bit-for-bit | [Determinism and reproducibility](./04-technical-guides/determinism-and-reproducibility.md) |
| Prepare training data | [Data preparation](./04-technical-guides/data-preparation.md) |
| Deploy to a Slurm cluster | [Deployment](./05-operations/deployment.md) |
| Debug a training failure | [Troubleshooting](./05-operations/troubleshooting.md) |
| Contribute to Primus | [Contributing](./06-developer-guide/contributing.md) |
| Understand the code architecture | [Architecture](./06-developer-guide/architecture.md) |
| Add a new training backend | [Extending backends](./06-developer-guide/extending-backends.md) |

---

## External resources

- [Primus-Turbo](https://github.com/AMD-AGI/Primus-Turbo): high-performance operators and kernels
- [Primus-SaFE](https://github.com/AMD-AGI/Primus-SaFE): external stability/platform layer; this repository does not include a production integration guide
- [AMD ROCm documentation](https://rocm.docs.amd.com/)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [TorchTitan](https://github.com/pytorch/torchtitan)
- [MaxText](https://github.com/AI-Hypercomputer/maxtext)

---

**Need help?** Open an issue on [GitHub](https://github.com/AMD-AGI/Primus/issues).
