# Projection

Primus includes projection tools that estimate **memory requirements** and **training performance** for large-scale distributed training without requiring the full target cluster. Two projection modes are available:

| Mode | Command | What it does |
|------|---------|--------------|
| **Memory** | `projection memory` | Estimates per-GPU memory (parameters, optimizer, activations) using analytical formulas |
| **Performance** | `projection performance` | Benchmarks layers on 1 node, then projects training time to multi-node clusters |

- **User-facing entry**: `primus-cli … -- projection {memory,performance} [options]`
- **Implementation entrypoint**: `primus/cli/subcommands/projection.py`
- **Core logic**:
  - Memory: `primus/core/projection/memory_projection/projection.py`
  - Performance: `primus/core/projection/performance_projection/projection.py`

This allows you to estimate training performance on larger clusters without actually running on them.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Command Syntax](#command-syntax)
3. [Memory Projection](#memory-projection)
   - [Overview](#memory-overview)
   - [Architecture](#memory-architecture)
   - [Parameter Estimation](#parameter-estimation)
   - [Param + Optimizer Memory](#param--optimizer-memory)
   - [Activation Memory](#activation-memory)
   - [Pipeline Schedule Memory Scaling](#pipeline-schedule-memory-scaling)
   - [Recomputation Support](#recomputation-support)
   - [Memory Formulas Reference](#memory-formulas-reference)
4. [Performance Projection](#performance-projection)
   - [Overview](#performance-overview)
   - [Profiling Modes](#profiling-modes)
   - [Simulation Backends](#simulation-backends)
   - [How It Works](#how-it-works)
   - [Scaling Mechanisms](#scaling-mechanisms)
   - [Communication Modeling](#communication-modeling)
   - [Pipeline Schedule Simulator](#pipeline-schedule-simulator)
   - [Overall Performance Prediction Flow](#overall-performance-prediction-flow)
5. [Example Output](#example-output)
6. [Assumptions and Limitations](#assumptions-and-limitations)
7. [Tips](#tips)

---

## Quick Start

### Memory Projection

Estimate per-GPU memory for a model configuration (no GPU needed for estimation, but the CLI currently requires torch distributed init):

```bash
export NNODES=1
export HSA_NO_SCRATCH_RECLAIM=1

bash runner/primus-cli direct --script primus/cli/main.py -- \
    projection memory \
    --config examples/megatron/configs/MI300X/deepseek_v2_lite-BF16-pretrain.yaml
```

### Training Projection (default)

Run a basic training performance projection for the minimum required nodes:

```bash
export NNODES=1
export HSA_NO_SCRATCH_RECLAIM=1

bash runner/primus-cli direct --script primus/cli/main.py -- \
    projection performance \
    --config examples/megatron/configs/MI300X/deepseek_v2_lite-BF16-pretrain.yaml
```

Project training performance to a specific number of nodes:

```bash
export NNODES=1
export HSA_NO_SCRATCH_RECLAIM=1

bash runner/primus-cli direct --script primus/cli/main.py -- \
    projection performance \
    --config examples/megatron/configs/MI300X/deepseek_v2_lite-BF16-pretrain.yaml \
    --target-nodes 4
```

Benchmark on a single GPU and project to multi-node (sub-node benchmarking):

```bash
export NNODES=1
export GPUS_PER_NODE=8
export HSA_NO_SCRATCH_RECLAIM=1

bash runner/primus-cli direct --script primus/cli/main.py -- \
    projection performance \
    --config examples/megatron/configs/MI300X/deepseek_v2_lite-BF16-pretrain.yaml \
    --benchmark-gpus 1 \
    --target-nodes 4
```

---

## Command Syntax

```bash
primus-cli [global-options] <mode> [mode-args] -- projection {memory,performance} [options]
```

### Common Options

| Option | Type | Description |
|--------|------|-------------|
| `--config` | string | Path to the Primus YAML configuration file (required) |

### Performance-Only Options

| Option | Type | Description |
|--------|------|-------------|
| `--target-nodes` | int | Target number of nodes for projection. Defaults to minimum required by parallelism config |
| `--benchmark-gpus` | int | Number of GPUs to use for benchmarking. Enables sub-node benchmarking when set lower than `GPUS_PER_NODE`. Parallelism dimensions (PP, EP, TP) are automatically reduced to fit and restored analytically during projection. Defaults to `GPUS_PER_NODE` |
| `--hardware-config` | string | Path to YAML file with custom hardware parameters for communication modeling |
| `--profiling-mode` | string | `benchmark` (default), `simulate` for pure analytical (no GPU), or `both` for side-by-side comparison |

### Parallelism Overrides

You can override parallelism settings from the config file using environment variables or CLI overrides:

```bash
# Using environment variables
export PRIMUS_TP=1
export PRIMUS_PP=3
export PRIMUS_EP=8

bash runner/primus-cli direct --script primus/cli/main.py -- \
    projection performance \
    --config examples/megatron/configs/MI300X/deepseek_v2_lite-BF16-pretrain.yaml \
    --target-nodes 6
```

---

## Memory Projection

<a name="memory-overview"></a>
### Overview

The memory projection estimates **per-GPU memory** usage by analytically computing:

1. **Parameter memory** — model weights stored on this GPU
2. **Optimizer state memory** — optimizer first/second moments, sharded across DP ranks
3. **Activation memory** — intermediate tensors stored for the backward pass

It uses a hierarchical profiler system that mirrors the model's module structure, computing each component's contribution bottom-up.

<a name="memory-architecture"></a>
### Architecture

```
LanguageModelProfiler
├── EmbeddingProfiler              — vocab embeddings (stage 0 only)
├── DenseTransformerLayerProfiler  — for non-MoE layers
│   ├── LayerNormProfiler (×3)     — pre-attn, pre-MLP, post-MLP
│   ├── AttentionProfiler          — QKV projections + attention
│   ├── ResidualAddProfiler (×2)   — skip connections
│   └── DenseMLPProfiler           — standard SwiGLU/FFN
├── MoETransformerLayerProfiler    — for MoE layers
│   ├── LayerNormProfiler (×3)
│   ├── AttentionProfiler
│   ├── ResidualAddProfiler (×2)
│   ├── RouterProfiler             — expert routing logits
│   └── MoEMLPProfiler            — routed experts + shared expert
├── LayerNormProfiler              — final layer norm (last stage only)
├── OutputLayerProfiler            — language model head (last stage only)
└── LossProfiler                   — cross-entropy loss (last stage only)
```

Each profiler implements two key methods:
- `estimated_num_params(rank)` — parameter count (total if `rank=None`, per-GPU if rank given)
- `estimated_activation_memory(batch_size, seq_len)` — activation bytes for one microbatch

<a name="parameter-estimation"></a>
### Parameter Estimation

Parameters are computed per component and summed across all layers assigned to this GPU's pipeline stage.

#### Layer Assignment

Layers are distributed across `PP × VPP` virtual stages. Each physical PP rank hosts `VPP` virtual stages in an interleaved pattern:

```
PP rank 0 → virtual stages 0, PP, 2×PP, ...
PP rank 1 → virtual stages 1, PP+1, 2×PP+1, ...
```

#### Per-Component Parameter Formulas

| Component | Formula | Notes |
|-----------|---------|-------|
| **Embedding** | `V × H` | `V` = padded vocab size, `H` = hidden size |
| **LayerNorm** | `2 × H` | gamma + beta per LayerNorm |
| **Attention (standard)** | `2 × H² × (1 + G/A) × P` | `A` = num heads, `G` = num KV groups, `P` = proj ratio |
| **Attention (MLA)** | `q_term + kv_term + pos + out` | DeepSeek-style multi-latent attention |
| **Dense MLP (SwiGLU)** | `3 × H × FFN` | gate, up, down projections |
| **Dense MLP (standard)** | `2 × H × FFN` | up, down projections |
| **Router** | `H × NE` | `NE` = number of experts |
| **MoE MLP** | `NE/EP × n_proj × H × FFN_e + shared` | Expert params sharded by EP |
| **Output Layer** | `V × H` | May share weights with embedding |

Where:
- `H` = `hidden_size`
- `V` = `padded_vocab_size`
- `FFN` = `ffn_hidden_size` (dense MLP intermediate dimension)
- `FFN_e` = `moe_ffn_hidden_size` (per-expert intermediate dimension)
- `NE` = `num_experts`
- `EP` = `expert_model_parallel_size`
- `n_proj` = 3 for SwiGLU, 2 for standard FFN

### Param + Optimizer Memory

The total bytes per parameter depends on the training precision and optimizer sharding:

```
bytes_per_param = weight_bytes + gradient_bytes + optimizer_bytes

Where:
  weight_bytes   = 2      (BF16 weights)
  gradient_bytes = 2      (BF16 gradients)
  optimizer_bytes = 10/DP  (FP32 master weights + Adam m + Adam v, sharded across DP)
                 = (2 + 4 + 4) / DP
```

**DP calculation:**

```
DP = world_size / (EP × PP)
```

Note: TP and CP are not divided out because all TP/CP ranks within a DP group share the same optimizer partition.

**Total param + optimizer memory per GPU:**

```
param_optimizer_memory = params_on_this_gpu × bytes_per_param
```

### Activation Memory

Activation memory is the memory needed to store intermediate tensors for the backward pass. Each component estimates what it stores for backward.

#### Base Tensor (sbh)

The fundamental building block is the hidden state tensor:

```
sbh = MBS × (S / TP / CP) × H × 2 bytes (BF16)
```

Where `MBS` = micro batch size, `S` = sequence length.

#### Per-Component Activation Formulas

##### LayerNorm
Stores its input for backward:
```
act = sbh = MBS × S/(TP×CP) × H × 2
```

##### Residual Add
Stores the residual for backward:
```
act = sbh
```

##### Router
Stores hidden states for routing weight gradients:
```
act = sbh
```

##### Attention (standard, Flash Attention)

Stores Q, K, V, output, and logsumexp for Flash Attention backward:

```python
tokens_per_rank = MBS × S / (TP × CP)

# activation width = Q + K + V + output + softmax stats
Q_width     = kv_channels × num_heads                    # e.g. 128 × 64 = 8192
KV_width    = kv_channels × num_kv_groups                # e.g. 128 × 1 = 128 (MQA)
output_width = hidden_size                                # 8192
softmax_width = Q_width  (with Flash Attention)           # 8192

total_width = Q_width + 2×KV_width + output_width + softmax_width
act = tokens_per_rank × total_width × 2  (BF16)
```

For MQA with 64 heads and 1 KV group: `Q(256MB) + K(4MB) + V(4MB) + O(256MB) + LSE(4MB) ≈ 0.51 GB`

##### Dense MLP (SwiGLU)

For the SwiGLU computation `output = down_proj(silu(gate_proj(x)) ⊙ up_proj(x))`, stores:

```python
tokens = MBS × S / (TP × CP)

# SwiGLU stores gate, up, and hidden (silu×up) for backward
intermediate = 2 × tokens × FFN × 2      # gate_proj + up_proj outputs (BF16)
activation   = tokens × FFN × 2           # silu(gate) ⊙ up (input to down_proj)
output       = tokens × H × 2             # down_proj output

act = intermediate + activation + output
    = tokens × (3×FFN + H) × 2
```

##### MoE MLP

For MoE, each token is routed to `topk` experts, duplicating the activation:

```python
tokens = MBS × S / (TP × CP)
topk_tokens = tokens × topk               # total token-expert pairs

# Routed experts: same SwiGLU structure per token-expert pair
routed_act = topk_tokens × (3×FFN_e + H) × 2

# Shared expert (if configured): processes ALL tokens
shared_act = tokens × (3×FFN_e + H) × 2   # same SwiGLU, one copy

act = routed_act + shared_act
    = tokens × (topk + N_shared) × (H + 3×FFN_e) × 2
```

Where:
- `topk` = `moe_router_topk` (experts activated per token)
- `FFN_e` = `moe_ffn_hidden_size` (per-expert FFN intermediate dimension)
- `N_shared` = 1 if `moe_shared_expert_intermediate_size` is set, else 0

**Example (MoE 4.5T, MBS=4, S=16384, CP=4, H=8192, FFN_e=2048, topk=36):**
```
tokens = 4 × 16384/4 = 16,384
MoE MLP = 16,384 × (36+1) × (8192 + 3×2048) × 2 = 16.19 GB
```

##### Full Transformer Layer (without recompute)

For a MoE layer, the total is the sum of all components:

```
layer_act = 3×LayerNorm + Attention + 2×ResidualAdd + Router + MoE_MLP
          = 3×sbh + attn_act + 2×sbh + sbh + moe_mlp_act
          = 6×sbh + attn_act + moe_mlp_act
```

For a dense layer: same but with Dense MLP instead of Router + MoE MLP.

##### Full Layer Activation Summary

| Component | Formula | Typical Size (MoE 4.5T) |
|-----------|---------|------------------------|
| LayerNorm (×3) | `3 × sbh` | 0.75 GB |
| Residual Add (×2) | `2 × sbh` | 0.50 GB |
| Router | `sbh` | 0.25 GB |
| Attention (Flash, MQA) | `tokens × (Q+2KV+O+softmax) × 2` | 0.51 GB |
| MoE MLP (SwiGLU) | `tokens × (topk+1) × (H+3×FFN_e) × 2` | 16.19 GB |
| **Full MoE layer** | **sum** | **18.20 GB** |
| With full recompute | `sbh` (checkpoint only) | 0.25 GB |

<a name="pipeline-schedule-memory-scaling"></a>
### Pipeline Schedule Memory Scaling

With pipeline parallelism, multiple microbatches are in-flight simultaneously, each requiring stored activations.

#### 1F1B Schedule

In the 1F1B (one-forward-one-backward) schedule, the first pipeline stage (rank 0) accumulates `PP` microbatches during the warmup phase before starting any backward passes. This means peak activation memory requires storing activations for `PP` microbatches.

```
base_activation = sum of per-layer activations across all layers on this rank
peak_activation = base_activation × PP
```

#### VPP (Virtual Pipeline Parallelism) Overhead

With interleaved scheduling (VPP > 1), there is a small memory overhead because more microbatches can be partially in-flight:

```
interleaved_penalty = 1 + (PP - 1) / (PP × VPP)
```

For VPP=1: penalty = 1 + (PP-1)/PP (significant overhead)
For VPP=20: penalty = 1 + (PP-1)/(PP×20) ≈ 1.04 (nearly negligible)

#### Gradient Accumulation Saving

When gradient accumulation (GA) steps are fewer than PP stages, the pipeline isn't fully filled:

```python
GA = GBS / (MBS × DP)
ga_saving = 1.0 if GA >= PP else GA / PP
```

#### Final Activation Memory Formula

```
total_activation = base_activation × PP × interleaved_penalty × ga_saving
```

<a name="recomputation-support"></a>
### Recomputation Support

Activation recomputation trades compute for memory by discarding intermediate activations during forward and recomputing them during backward.

#### Full Recompute (`recompute_granularity="full"`)

When a layer is fully recomputed, only the **input tensor** is stored as a checkpoint:

```
recomputed_layer_act = sbh = MBS × S/(TP×CP) × H × 2 bytes
```

This is dramatically smaller than the full activation. For example, an MoE layer drops from ~18 GB to 0.25 GB.

#### Partial Recompute

The `recompute_num_layers` setting controls how many layers per VPP stage are recomputed:

```python
for each layer on this rank:
    local_idx = layer_index % layers_per_vpp_stage
    if recompute_granularity == "full" and local_idx < recompute_num_layers:
        act += input_act_per_layer   # just sbh (checkpoint)
    else:
        act += full_layer_act        # all intermediates
```

#### With Recompute: Total Memory

```
total_with_recompute = (L/PP × sbh) × PP × interleaved_penalty × ga_saving
                     + recompute_working_memory (1 layer's full activation, temporary)
                     + embedding_act (stage 0 only)
```

The recompute working memory is transient — only one layer's full intermediates exist at a time during backward.

### Memory Formulas Reference

Summary of all memory components for one GPU:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Total GPU Memory                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. Parameters (BF16)                                               │
│     = params_on_rank × 2 bytes                                      │
│                                                                      │
│  2. Gradients (BF16)                                                │
│     = params_on_rank × 2 bytes                                      │
│                                                                      │
│  3. Optimizer States (FP32, sharded across DP)                      │
│     = params_on_rank × 10 / DP bytes                                │
│     (master weights: 2B + Adam m: 4B + Adam v: 4B)                  │
│                                                                      │
│  4. Activations                                                     │
│     = Σ(per-layer act) × PP × VPP_penalty × GA_saving              │
│     + embedding/output activations (stage-dependent)                 │
│                                                                      │
│  5. Transient buffers (not in projection)                           │
│     - A2A dispatch/combine buffers                                   │
│     - Communication scratch space                                    │
│     - PyTorch allocator fragmentation                                │
│                                                                      │
│  Total = (1) + (2) + (3) + (4)                                     │
│  Reported as: Param+Optimizer + Activation = Projected Total        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Performance Projection

<a name="performance-overview"></a>
### Overview

The performance projection tool uses a **hybrid benchmark-then-project** approach:

1. **Benchmarks** transformer layers on a configurable number of GPUs (from a single GPU up to a full node) to measure forward/backward pass times
2. **Simulates** pipeline parallelism scheduling (including zero-bubble optimization)
3. **Projects** performance to multi-node configurations by modeling:
   - Data Parallelism (DP) scaling
   - Gradient AllReduce communication overhead (training only)
   - Expert Parallelism (EP) All-to-All communication overhead
   - Inter-node vs intra-node communication differences

#### Why Hybrid? Measure What You Can, Simulate What You Can't

The core design principle is: **measure what you can, simulate what you can't**. When the benchmark configuration can accommodate a parallelism dimension, both its compute and communication are captured in real measurements. Only communication that falls outside the benchmark scope — inter-node traffic, or overhead from parallelism dimensions that were reduced to fit — is estimated analytically. This hybrid approach yields more accurate results than pure analytical modeling for several reasons:

- **Immunity to the Peak-vs-MAF gap**: Modern GPUs throttle clock frequency under sustained workloads. The gap between peak FLOPs and Max-Achievable FLOPs (MAF) can be 44–70% on current hardware (see [Understanding Peak, Max-Achievable & Delivered FLOPs](https://rocm.blogs.amd.com/software-tools-optimization/Understanding_Peak_and_Max-Achievable_FLOPS/README.html)). Benchmarking captures the actual operating frequency; analytical models must assume or estimate it.
- **Robustness to software stack changes**: Each ROCm update (hipBLASLt GEMM improvements, CK attention kernels, Triton retiling) shifts achievable performance. Benchmarking automatically reflects the current stack; analytical models require recalibration.
- **Accurate grouped GEMM for MoE**: Grouped GEMM performance depends on expert count, topk routing, token distribution, and kernel implementation — factors that are difficult to model analytically. Benchmarking measures actual execution directly.
- **Framework overhead included**: PyTorch dispatch latency, memory allocator behavior, kernel launch overhead, and stream synchronization are captured in benchmarks but absent from analytical models.
- **Communication is measured when possible**: When a parallelism dimension fits within the benchmark GPUs (e.g., TP AllReduce within a node, EP All-to-All within the benchmark config), its communication is captured in the benchmark. Only communication outside the benchmark scope falls back to analytical models. For those cases, collectives are more analytically tractable than compute — dominated by bandwidth and latency with predictable message sizes. Pipeline scheduling follows deterministic rules that can be simulated exactly.
- **Transparent validation**: `--profiling-mode both` runs benchmark and simulation side by side, letting users quantify the analytical accuracy gap and decide when pure simulation (for no-GPU capacity planning) can be trusted.

| Factor | Pure Analytical | Hybrid (Primus) |
|--------|----------------|-----------------|
| GPU frequency under load | Must assume or estimate | Captured via real measurement |
| Software stack changes | Requires recalibration | Automatically reflects current performance |
| Grouped GEMM for MoE | Difficult to model | Measured directly |
| Framework overhead | Not captured | Included in benchmarks |
| Communication modeling | Must model analytically | Measured when within benchmark scope; analytical fallback for the rest |
| No-GPU capacity planning | Supported | Supported via `--profiling-mode simulate` |
| Cross-validation | Not available | `--profiling-mode both` |

### Profiling Modes

The `--profiling-mode` flag controls how per-layer compute times are obtained:

| Mode | GPU Required | What it does |
|------|-------------|--------------|
| `benchmark` (default) | **Yes** | Runs real GPU kernels on 1 node and measures forward/backward times with CUDA events |
| `simulate` | **No** | Uses **Origami** (GEMM) and **SDPA Simulator** (attention) analytical backends to predict layer times — no GPU or model instantiation needed |
| `both` | **Yes** | Runs both benchmark and simulation side-by-side, prints a comparison table, and uses benchmark results for the actual projection |

When you **don't have access to a GPU** (e.g., capacity planning on a laptop), use `--profiling-mode simulate`. The simulation backends analytically predict kernel execution times based on matrix dimensions, data types, and hardware characteristics.

### Simulation Backends

When `--profiling-mode simulate` is selected, layer compute times are predicted analytically using two simulation backends:

#### Origami (GEMM Backend)

[Origami](https://github.com/ROCm/rocm-libraries/tree/develop/shared/origami/python) is an open-source tool (part of the ROCm ecosystem) that provides a fast, analytical, deterministic methodology to select optimal GEMM configurations (such as tile size) for out-of-the-box GEMM performance on AMD GPUs. It also provides an analytical performance model for GEMM kernels — Primus uses this analytical modeling capability to predict GEMM execution times without running on actual hardware.

- **Used for**: All GEMM operations — attention projections (Q, K, V, O), MLP (gate, up, down), MoE expert GEMMs, embedding, and output layers
- **How it works**: Models the GPU's Compute Units (CUs), memory hierarchy, and tile-level execution to predict GEMM duration given matrix dimensions, data type, and hardware characteristics
- **Built-in hardware profiles**: Supports architectures like MI300X, MI325X, MI355X, etc. with pre-configured CU counts, HBM bandwidth, and peak TFLOPS
- **GPU arch override**: Use `--gpu-arch` (or `PRIMUS_GPU_ARCH` env var) to select a target architecture
- **Clock override**: Use `--gpu-clock-mhz` to scale compute throughput for different clock frequencies

Installation:
```bash
pip install git+https://github.com/ROCm/rocm-libraries.git#subdirectory=shared/origami/python
```

#### SDPA Simulator (Attention Backend)

The SDPA simulator models Flash Attention v3 (FAv3) kernel execution analytically using Origami's 1-CU tile-level model.

- **Used for**: Scaled Dot-Product Attention (SDPA) — both forward and backward passes
- **How it works**: Models FAv3 as a fused kernel where QKᵀ, softmax, and PV are sequential within each workgroup. Total time = (per-tile-QKᵀ + per-tile-PV) × num_waves. This captures wave quantization and per-tile efficiency without needing an empirical `compute_efficiency` parameter.
- **Backward pass**: Also models the dQ atomic overhead from `buffer_atomic_add_f32` accumulation across KV-workgroups
- **Built-in hardware profiles**: Same architecture support as Origami (e.g. MI300X, MI325X, MI355X)
- **Requires Origami**: The SDPA simulator uses Origami internally for tile-level GEMM simulation

#### Simulation Quick Start

Run a full training projection without any GPU:

```bash
# No GPU required — runs entirely on CPU
bash runner/primus-cli direct --script primus/cli/main.py -- \
    projection performance \
    --config examples/megatron/configs/MI300X/deepseek_v2_lite-BF16-pretrain.yaml \
    --profiling-mode simulate \
    --target-nodes 4
```

Target a specific GPU architecture:

```bash
bash runner/primus-cli direct --script primus/cli/main.py -- \
    projection performance \
    --config examples/megatron/configs/MI300X/deepseek_v2_lite-BF16-pretrain.yaml \
    --profiling-mode simulate --gpu-arch mi355x \
    --target-nodes 4
```

Compare simulation accuracy against projection with single node benchmarking:

```bash
bash runner/primus-cli direct --script primus/cli/main.py -- \
    projection performance \
    --config examples/megatron/configs/MI300X/deepseek_v2_lite-BF16-pretrain.yaml \
    --profiling-mode both \
    --target-nodes 4
```

#### Sub-Node Benchmarking

Benchmark on fewer GPUs than a full node and project to multi-node. For example, benchmark on 1 GPU and project to 4 nodes:

```bash
export NNODES=1
export GPUS_PER_NODE=8

bash runner/primus-cli direct --script primus/cli/main.py -- \
    projection performance \
    --config examples/megatron/configs/MI300X/deepseek_v2_lite-BF16-pretrain.yaml \
    --benchmark-gpus 1 \
    --target-nodes 4
```

The tool automatically reduces PP, EP, and if necessary TP to fit on the benchmark GPU count, then restores the full target configuration analytically during projection. This is useful when only a fraction of a node is available for profiling.

### How It Works

#### 1. Configuration Reduction

If the target parallelism configuration requires more GPUs than are available for benchmarking, the tool automatically reduces the config to fit. The number of benchmark GPUs defaults to `GPUS_PER_NODE` (full-node benchmarking), but can be set to any value via `--benchmark-gpus` — including as low as a single GPU — to enable **sub-node benchmarking**.

Parallelism dimensions are reduced in a fixed priority order:

1. **Pipeline Parallelism (PP)**: Reduced to 1 first (easiest to add back — PP overhead is estimated analytically via the pipeline schedule simulator)
2. **Expert Parallelism (EP)**: Reduced next if the config still doesn't fit (MoE compute stays accurate because experts-per-rank is preserved; only All-to-All communication is replaced analytically for the target EP)
3. **Tensor Parallelism (TP)**: Reduced last if PP=1 and EP reduction were not sufficient (compute is scaled by `benchmark_tp / target_tp` and TP AllReduce overhead is added analytically)

CP is kept unchanged throughout. After benchmarking, all reduced dimensions are restored analytically during projection.

For example, with `--benchmark-gpus 1` on a config that requires TP=8, PP=4, EP=8, the tool would benchmark on a single GPU with PP=1, EP=1, TP=1, then analytically reconstruct the full target configuration's performance.

#### 2. Layer Benchmarking

The tool benchmarks each transformer layer type on the available benchmark GPUs:
- Dense attention layers
- MoE (Mixture of Experts) layers
- Measures forward and backward pass times separately
- Also benchmarks embedding and output layers

#### 3. Pipeline Simulation

For PP > 1, the tool simulates the pipeline schedule to account for:
- Pipeline bubble overhead
- Microbatch interleaving
- Zero-bubble scheduling (if enabled)

#### 4. Data Parallel Scaling

The projection models how performance scales with additional nodes:

```
Projected Time = (Base Time / DP_scaling_factor) + Communication Overheads
```

### Scaling Mechanisms

#### Tensor Parallelism (TP)

**What it does**: Splits individual layer weights across GPUs within a node.

**How it's modeled**: When benchmarking runs with the **same** TP as the target configuration, TP communication (AllReduce after each layer) is already captured in the benchmark — no additional modeling is needed. However, if the benchmark GPU count is too small to accommodate the target TP (e.g., benchmarking on 1 GPU for a TP=8 config), TP is reduced during benchmarking and the projection analytically compensates:
1. **Compute scaling**: Per-GPU compute is scaled by `benchmark_tp / target_tp` (each GPU processes a proportionally smaller slice with higher TP).
2. **TP AllReduce delta**: The AllReduce overhead difference between benchmark TP and target TP is estimated analytically and added to each layer's time. When GPU-measured AllReduce data is available, it is used to anchor the analytical model for better accuracy.

**Communication**: AllReduce within TP group (typically intra-node, fast).

#### Pipeline Parallelism (PP)

**What it does**: Distributes layers across pipeline stages. Each stage processes microbatches in sequence.

**How it's modeled**: 
- PP is the first dimension reduced during configuration reduction — it is always set to 1 for benchmarking when the config doesn't fit on the available GPUs
- A **pipeline scheduler simulator** (`simulator.py`) reconstructs the full pipeline schedule for the target PP
- Simulates the actual 1F1B or zero-bubble schedule with proper send/receive synchronization
- Accounts for pipeline bubble overhead and microbatch interleaving

#### Expert Parallelism (EP)

**What it does**: Distributes MoE experts across GPUs. Each GPU holds a subset of experts.

**How it's modeled**:
- If EP spans multiple nodes, the tool estimates **inter-node All-to-All overhead**
- Compares All-to-All time for benchmark EP (intra-node) vs target EP (potentially inter-node)
- Adds the overhead difference to each MoE layer's forward/backward time

**Communication**: All-to-All for token dispatch (before expert computation) and combine (after).

```
All-to-All Message Size = tokens × hidden_size × top_k × 2 (BF16)
```

#### Data Parallelism (DP)

**What it does**: Replicates the model across DP groups. Each group processes different data batches.

**How it's modeled**:
- DP provides **linear speedup** by processing more batches in parallel
- Scaling factor = `target_DP / baseline_DP`

**Communication**: Gradient AllReduce across all DP ranks.

```
Gradient AllReduce Size = num_params × 4 (FP32 gradients)
```

**Optimization**: If `overlap_grad_reduce=True` (default), gradient AllReduce is overlapped with backward computation and not on the critical path.

#### Context Parallelism (CP)

**What it does**: Splits sequence length across GPUs for long-context training.

**How it interacts with other dimensions**: CP behaves differently for dense and MoE models:

- **Dense models**: CP is an independent parallelism axis. It directly multiplies the GPU count:
  ```
  Total GPUs = TP × PP × CP × DP
  ```
  Each CP rank holds a different chunk of the sequence and they communicate via AllGather/AllToAll during attention.

- **MoE models (with Parallel Folding)**: CP is **folded into** EP — the CP ranks are a subset of the EP ranks. The constraints are `CP ≤ EP` and `EP % CP == 0`. CP does **not** add extra GPUs beyond EP:
  ```
  Total GPUs = TP × PP × EP × DP
  ```
  Within the EP group, CP determines how many of the EP ranks share context-parallel work on attention. The remaining `EP / CP` factor provides inner data-parallel streams for the attention layers. For the MoE FFN layers, all EP ranks participate in expert parallelism as usual.

  For example, with EP=8 and CP=4: the 8 EP ranks form 2 groups of 4 for context-parallel attention, giving 2 inner-DP streams. Traditional (unfolded) parallelism would require `EP × CP = 32` GPUs; with folding it requires only `EP = 8`.

**How it's modeled**: CP affects the GPU topology for communication routing. Currently included in minimum GPU requirements calculation.

### Communication Modeling

When a parallelism dimension fits within the benchmark GPU count, its communication is **measured** as part of the benchmark — no analytical modeling needed. For communication that falls outside the benchmark scope (e.g., inter-node collectives, or overhead from parallelism dimensions that were reduced to fit the benchmark), the tool uses analytical models to estimate collective communication times:

| Collective | Used By | Model |
|------------|---------|-------|
| AllReduce | TP, DP (gradients) | Best of Ring/Hypercube/Bruck/Single-shot |
| All-to-All | EP (MoE dispatch/combine) | Pairwise exchange, accounts for topology |
| P2P Send/Recv | PP (activations) | Point-to-point latency + bandwidth |

Communication times differ significantly based on:
- **Intra-node**: Fast (e.g., NVLink, UALink, xGMI)
- **Inter-node**: Slower (e.g., InfiniBand, RoCE)

Custom hardware parameters can be provided via `--hardware-config <yaml>`. See [`mi300x.yaml`](../examples/hardware_configs/mi300x.yaml) for the format.

Example usage:
```bash
primus projection performance --config <model_config> --hardware-config <hardware_config.yaml>
```

#### Key Concepts

##### Minimum Nodes Required

The minimum nodes required depends on the model type:

**Dense models** (no expert parallelism):
```
Min GPUs  = TP × PP × CP
Min Nodes = ceil(Min GPUs / GPUs_per_node)
```

**MoE models** (with MoE Parallel Folding, where CP is folded into EP):
```
Min GPUs  = TP × PP × EP          (CP ≤ EP, folded in)
Min Nodes = ceil(Min GPUs / GPUs_per_node)
```

##### Scaling Behavior

- **DP scaling**: Linear speedup. Doubling DP halves iteration time (minus communication overhead).
- **PP scaling**: Happens in multiples of pipeline replicas. With PP=3, you need 3, 6, 9... nodes to increase scaling.
- **EP scaling**: Divides the experts across EP ranks. For MoE models, EP also subsumes the CP dimension.

### Pipeline Schedule Simulator

The pipeline simulator (`simulator.py`) simulates the execution of pipeline parallelism schedules to calculate step time and bubble ratio.

#### Schedule Algorithms

| Algorithm | Description | Use Case |
|-----------|-------------|----------|
| **1F1B** | Standard one-forward-one-backward | Default pipeline schedule |
| **Interleaved 1F1B** | Multiple chunks per rank (VPP > 1) | Reduced bubble ratio |
| **Zero-Bubble** | Splits backward into B + W | Minimal bubble overhead |

#### Zero-Bubble Scheduling

Zero-bubble minimizes pipeline bubbles by separating the backward pass:
- **B (Input Gradient):** Compute gradients w.r.t. input activations
- **W (Weight Gradient):** Compute gradients w.r.t. weights

This allows more flexible scheduling because W doesn't depend on receiving gradients from the next stage. By default, backward time is split 50/50 between B and W.

Two implementations are available:
1. **Simple Zero-Bubble Simulator** — basic F-B-W pattern with warmup/steady/cooldown phases
2. **Megatron ILP-Based Scheduler** — graph-based schedule optimization with memory-aware scheduling using Megatron's actual zero-bubble scheduler

#### P2P Communication in Pipeline Simulation

The pipeline simulator uses a **fixed small constant** (~0.1 ms) for P2P communication, NOT the analytical `sendrecv` model. This is because:
1. P2P communication is typically **overlapped with compute** in optimized schedules
2. The simulator focuses on **schedule ordering and bubble calculation**
3. P2P time is **small relative to F/B/W times** for large models

However, when the benchmark PP differs from the target PP (e.g., benchmark PP=1, target PP=6), the **analytical `sendrecv` model** is used to estimate the PP communication overhead that was not captured in the benchmark:

```
PP overhead = additional_stages × 2 (fwd+bwd) × microbatches × sendrecv(activation_size)
```

P2P communication becomes significant when PP stages span nodes (inter-node P2P has 10-100× higher latency than intra-node).

### Overall Performance Prediction Flow

```
1. Load Configuration
   └── Parse YAML config, extract parallelism settings

2. Configuration Reduction (if config doesn't fit on benchmark GPUs)
   ├── Reduce PP to 1
   ├── If still doesn't fit, reduce EP (preserving experts-per-rank)
   ├── If still doesn't fit, reduce TP (last resort)
   └── Benchmark GPU count controlled by --benchmark-gpus (default: GPUS_PER_NODE)

3. Layer Benchmarking (on available benchmark GPUs)
   ├── Limit layers (1 dense + 1 MoE for efficiency)
   └── Benchmark forward + backward times

4. Extrapolate to Full Model
   ├── Multiply per-layer times by total layer count
   └── If TP was reduced: scale compute by benchmark_tp/target_tp, add TP AllReduce delta

5. Pipeline Schedule Simulation (if PP > 1)
   ├── Build chunk time matrix (per-rank, per-chunk)
   ├── Select scheduler (1F1B, Interleaved, Zero-Bubble)
   └── Simulate to get step_time_ms and bubble_ratio

6. Add Communication Overhead (if config was reduced)
   ├── PP overhead: P2P communication between stages
   ├── EP overhead: Additional All-to-All for larger EP
   └── TP overhead: AllReduce delta (if TP was reduced)

7. Multinode Scaling Projection
   ├── Calculate DP scaling factor
   ├── Scale compute time: projected = base × (min_dp / target_dp)
   ├── Add gradient AllReduce (if not overlapped)
   └── Report projected iteration time and tokens/s
```

#### Performance Formula

```
Projected_Time = Base_Time × (Min_DP / Target_DP) + Communication_Overhead

Where:
  Base_Time             = Pipeline simulation time (includes PP bubbles)
  Min_DP                = DP at minimum node configuration
  Target_DP             = DP at target node configuration
  Communication_Overhead = Gradient AllReduce (if not overlapped)
                         + MoE All-to-All overhead (if EP spans nodes)
```

#### Example Calculation

**Configuration:** DeepSeek V2 Lite — TP=1, PP=3, EP=8, CP=1 — GBS=640, MBS=4, Seq=4096

```
Step 1: Minimum Nodes
  GPUs required = 1 × 3 × 8 × 1 = 24 GPUs = 3 nodes
  Min DP = 24 / (1 × 3 × 1) = 8

Step 2: Target Configuration (6 nodes)
  Total GPUs = 48
  Target DP = 48 / (1 × 3 × 1) = 16
  DP Scaling = 16 / 8 = 2×

Step 3: Projected Time
  Base_Time (from pipeline simulation) = 5500 ms
  Projected_Time = 5500 × (8 / 16) = 2750 ms
  Tokens/s = (640 × 4096) / 2.750 = 953,018 tokens/s
```

---

## Example Output

### Memory Projection

```
====================================================================================================
[Primus:Projection] Component-wise Profiling Results (Rank 0):
====================================================================================================

  Total Number of Parameters: 15.654321 Billion (15,654,321,024)

  [embedding]
    Params: 0.819200 Billion (819,200,000)
    Activation Memory: 0.2500 GB

    [dense_transformer_layer]
      Params: 0.302000 Billion (302,000,000)
      Activation Memory: 2.1250 GB

      [layer_norm]
        Params: 0.000016 Billion (16,384)
        Activation Memory: 0.2500 GB

      [self_attention]
        Params: 0.134218 Billion (134,217,728)
        Activation Memory: 0.5100 GB

      [mlp]
        Params: 0.167772 Billion (167,772,160)
        Activation Memory: 0.8650 GB

    [moe_transformer_layer]
      Params: 1.001400 Billion (1,001,400,000)
      Activation Memory: 18.2000 GB

====================================================================================================
[Primus:Projection] Memory Projection Summary on Rank 0:
  Params: 20.850000 Billion (20,850,000,000)
  Param+Optimizer Memory: 83.7400 GB
  Activation Memory (per batch size 4, seq len 16384): 36.7500 GB
  Projected Total Memory: 120.4900 GB
====================================================================================================
```

### Performance Projection (Training)

```
====================================================================================================
[Primus:Performance Projection] Configuration Summary:
  Benchmark Config: PP=1, EP=8, TP=1, CP=1, DP=1 (1 node)
  Target Config: PP=1, EP=8, TP=1, CP=1, DP=4 (4 nodes)
  Benchmark Microbatches: 160 (global_batch=640, micro_batch=4, benchmark_dp=1)

====================================================================================================
Multinode Scaling Projection Results
====================================================================================================

📊 Parallelism: TP=1, PP=1, EP=8, CP=1

🎯 Target Configuration (4 nodes):
   Nodes: 4, GPUs: 32
   TP=1, PP=1, EP=8, CP=1, DP=4
   DP Scaling Factor: 4.0x
   Iteration Time: 4012.456 ms
   Tokens/s: 649,123

📡 Communication Breakdown:
   gradient_allreduce: 45.123 ms (message: 1024.00 MB) [OVERLAPPED]
   moe_a2a_fwd: 230.500 ms (message: 64.00 MB, 26 layers × 8.866 ms/layer)
   moe_a2a_bwd: 230.500 ms (message: 64.00 MB, 26 layers × 8.866 ms/layer)

   Total Communication (critical path): 461.000 ms

====================================================================================================
```

---

## Assumptions and Limitations

### Assumptions

1. **DP Weak Scaling** — DP scaling assumes weak scaling (constant micro-batch size per GPU); the DP gradient AllReduce overhead is modeled analytically
2. **Communication Model** — Bandwidth efficiency uses a constant factor; inter-node communication uses switch topology; all NICs are used in parallel for inter-node traffic
3. **Pipeline Bubbles** — B/W split is 50/50 for zero-bubble scheduling; P2P communication time is small relative to compute
4. **Gradient AllReduce** — By default overlapped with compute (`overlap_grad_reduce=True`); if disabled, added to critical path
5. **MoE All-to-All** — All-to-All time scales with EP size; per-peer latency overhead is constant

### Limitations

1. **Benchmark Accuracy with Reduced Parallelism** — Benchmarking with reduced PP/EP/TP may not capture all behaviors (e.g., GEMM efficiency differences at different TP levels); layer timing variance is assumed uniform
2. **Communication Contention** — Model doesn't account for network contention; assumes dedicated bandwidth per collective
3. **Memory Pressure** — Memory impact on performance not fully modeled; activation recomputation overhead not considered in performance
4. **Hardware Heterogeneity** — Assumes homogeneous nodes; GPU frequency variations not modeled

---

## Tips

- **Start with memory projection**: Run `projection memory` first to verify your model fits in GPU memory before benchmarking.
- **Benchmark with what you have**: Use `--benchmark-gpus` to run benchmarks on any number of GPUs (even 1) and project to multi-node. The tool handles parallelism downscaling (PP → EP → TP) and analytical upscaling automatically.
- **Understand scaling limits**: DP scaling is limited by `global_batch_size / micro_batch_size`. If you run out of microbatches, adding more nodes won't help.
- **Check minimum nodes**: If your config requires multiple nodes (e.g., PP=4 with 8 GPUs/node), the performance projection will automatically reduce PP for benchmarking.
- **Pipeline scaling**: With PP > 1, layers don't need to divide evenly across stages. The tool distributes remainder layers to the first stages (e.g., 61 layers with PP=4 → [16, 15, 15, 15]). You can also supply explicit per-stage layer counts via `decoder_first_pipeline_num_layers`, `decoder_last_pipeline_num_layers`, or `pipeline_model_parallel_layout`.
- **Recomputation trade-off**: Full recompute dramatically reduces activation memory (e.g., 18 GB → 0.25 GB per MoE layer) at the cost of ~33% more compute.
- **MoE activation dominance**: For MoE models, the MoE MLP activation (scaled by `topk`) typically dominates the per-layer activation budget. Consider recomputation if memory is tight.
- **No GPU? Use simulation**: With `--profiling-mode simulate`, you can run performance projection entirely on CPU using the Origami GEMM and SDPA analytical backends. This is useful for capacity planning without GPU access.
- **Validate simulation accuracy**: Use `--profiling-mode both` to run projection via single node benchmarking and simulation side-by-side and compare the results.

## Related Documentation

- [Benchmark Suite](benchmark.md) - For microbenchmarking individual operations
- [Quickstart Guide](quickstart.md) - Getting started with Primus
