# Glossary

Alphabetical reference for terms used in Primus documentation and configuration. Cross-links point to other production docs where applicable.

---

### AINIC

**AMD Infinity NIC** — AMD’s high-performance network interface for multi-node GPU communication.

---

### Backend

A training framework integrated into Primus (for example **Megatron-LM**, **TorchTitan**, **MaxText**, **Megatron Bridge**, **HummingbirdXT**).

---

### BackendAdapter

Abstract class in Primus that connects a backend: discovery of setup paths, config conversion, and trainer loading.

---

### BackendRegistry

Registry mapping backend names to adapter classes, often with **lazy import** to avoid loading unused frameworks.

---

### BaseTrainer

Abstract trainer defining the lifecycle: **setup** → **init** → **train** → **cleanup**.

---

### BF16 / FP16 / FP8 / FP4

Floating-point precisions: **Brain Float 16**, **IEEE half**, **8-bit float**, and **4-bit float** training or inference formats (exact support depends on backend and hardware).

---

### CP (Context Parallelism)

Parallelism that **splits the sequence dimension** across devices for long-context training.

---

### DP (Data Parallelism)

Replicates the model across GPUs; each rank processes **different data batches**.

---

### DeepEP

**Deep Expert Parallelism** — Primus-Turbo’s acceleration path for **MoE token dispatch** and related expert-parallel work.

---

### EP (Expert Parallelism)

Distributes **Mixture-of-Experts** expert networks across devices.

---

### Experiment Config

Top-level **YAML** describing `work_group`, **modules**, and **overrides** for a training run.

---

### FSDP

**Fully Sharded Data Parallel** — shards parameters, gradients, and optimizer states across devices (PyTorch FSDP and similar concepts per backend).

---

### GBS (Global Batch Size)

Total **batch size across all data-parallel ranks** for one optimizer step (may combine micro-batching and gradient accumulation).

---

### Gradient Accumulation

Accumulates gradients over **multiple micro-batches** before an optimizer update.

---

### HipBLASLt

AMD’s high-performance **BLAS** library with **autotuning** for GEMM and related kernels.

---

### Hook

Shell or Python scripts under `runner/helpers/hooks/` executed at defined **lifecycle** points.

---

### LoRA

**Low-Rank Adaptation** — parameter-efficient fine-tuning that trains small adapter matrices.

---

### MBS (Micro Batch Size)

Batch size **per GPU** (per rank) for **one forward/backward pass** within a gradient-accumulation window.

---

### MLA (Multi-Latent Attention)

Compressed **KV-cache** attention architecture used in models such as DeepSeek.

---

### MoE (Mixture of Experts)

Architecture with **multiple expert** sub-networks and a **router** that assigns tokens to experts.

---

### Model Config

YAML **preset** describing architecture (hidden size, layers, attention heads, and so on).

---

### Module Config

YAML **preset** for training behavior: learning rate, batch sizes, optimizer, schedules.

---

### NCCL / RCCL

**NVIDIA Collective Communications Library** / **ROCm** equivalent — libraries for **GPU collective** operations in distributed training.

---

### PP (Pipeline Parallelism)

Splits **model layers** into **stages** on different devices.

---

### Patch

Runtime **monkey-patch** registered in **PatchRegistry** and applied at a named training phase.

---

### PatchRegistry

Registry of **phase-aware** patches (for example `build_args`, `setup`, `before_train`, `after_train`).

---

### Platform Config

YAML describing **cluster environment** mappings (for example `platform_azure.yaml`): env vars, paths, and scheduler hints.

---

### Preflight

Cluster **diagnostic** tooling that checks host, GPU, network, and baseline performance before long jobs. See `primus/tools/preflight/` in the repository.

---

### Preset

Reusable YAML fragment under `primus/configs/` (**module**, **model**, or **platform**).

---

### PrimusRuntime

Core **orchestrator**: loads configuration, resolves the backend, applies patches, and drives the **trainer lifecycle**.

---

### Primus-SaFE

**Stability and Fault-tolerance Engine** — ecosystem component for **cluster management** and resilience (external to this repo).

---

### Primus-Turbo

High-performance **operator** library (for example FlashAttention-style kernels, GEMM, collectives, grouped GEMM).

---

### Projection

Tools that **estimate memory** and **training performance** without requiring a full production cluster.

---

### ROCm

**Radeon Open Compute** — AMD’s GPU computing platform (drivers, compilers, libraries).

---

### SFT (Supervised Fine-Tuning)

Supervised fine-tuning that typically **updates all** (or a defined subset of) model parameters, as opposed to adapter-only methods.

---

### SP (Sequence Parallelism)

Parallelism that extends tensor-parallel regions to **non-TP** parts of the model to **reduce activation memory**.

---

### TP (Tensor Parallelism)

Splits **layer weights** across GPUs within a node (or defined process group).

---

### Transformer Engine (TE)

Library stack for **FP8** and related training optimizations (availability depends on backend and build).

---

### VPP (Virtual Pipeline Parallelism)

**Interleaved** pipeline parallelism with **multiple virtual stages** per device to improve utilization.

---

### Zero-Bubble

Pipeline scheduling that **reduces or eliminates pipeline bubbles** (idle time between micro-batches).

---

## See also

- [Overview](./overview.md)
- [Configuration system](../02-user-guide/configuration-system.md)
