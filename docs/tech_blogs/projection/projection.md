<!---
Copyright (c) 2025 Advanced Micro Devices, Inc. (AMD)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
--->

# Primus Projection: Estimate memory and performance before you train

Planning a large-scale distributed training run is expensive — both in time and in GPU hours. Misconfiguring parallelism settings can lead to out-of-memory crashes, underutilized hardware, or unexpectedly long training times. Primus includes a **projection** tool that lets you answer two critical questions *before* committing to a full-scale run: **"Will it fit?"** and **"How fast will it be?"**

This blog covers the design and capabilities of the Primus projection tool, which provides analytical **memory estimation** and **performance projection** for large-scale LLM training across multi-node GPU clusters.

## Quick Start

### Memory Projection

Estimate per-GPU memory for a model configuration:

```bash
export NNODES=1
export HSA_NO_SCRATCH_RECLAIM=1

bash runner/primus-cli direct --script primus/cli/main.py -- \
    projection memory \
    --config examples/megatron/configs/MI300X/deepseek_v2_lite-BF16-pretrain.yaml
```

### Performance Projection (Benchmark Mode)

Run a training performance projection using single-node GPU benchmarking:

```bash
export NNODES=1
export HSA_NO_SCRATCH_RECLAIM=1

bash runner/primus-cli direct --script primus/cli/main.py -- \
    projection performance \
    --config examples/megatron/configs/MI300X/deepseek_v2_lite-BF16-pretrain.yaml \
    --target-nodes 4
```

### Performance Projection (Simulation Mode — No GPU Required)

Run a full training projection entirely on CPU using analytical backends:

```bash
bash runner/primus-cli direct --script primus/cli/main.py -- \
    projection performance \
    --config examples/megatron/configs/MI300X/deepseek_v2_lite-BF16-pretrain.yaml \
    --profiling-mode simulate \
    --target-nodes 4
```

Target a specific GPU architecture for simulation:

```bash
bash runner/primus-cli direct --script primus/cli/main.py -- \
    projection performance \
    --config examples/megatron/configs/MI300X/deepseek_v2_lite-BF16-pretrain.yaml \
    --profiling-mode simulate --gpu-arch mi355x \
    --target-nodes 4
```

### Sub-Node Benchmarking (Benchmark on Fewer GPUs)

Benchmark on a single GPU and project to multi-node:

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

### Compare Benchmark vs. Simulation

Validate simulation accuracy against real hardware:

```bash
bash runner/primus-cli direct --script primus/cli/main.py -- \
    projection performance \
    --config examples/megatron/configs/MI300X/deepseek_v2_lite-BF16-pretrain.yaml \
    --profiling-mode both \
    --target-nodes 4
```

### Parallelism Overrides

Override parallelism settings from the config file:

```bash
export PRIMUS_TP=1
export PRIMUS_PP=3
export PRIMUS_EP=8

bash runner/primus-cli direct --script primus/cli/main.py -- \
    projection performance \
    --config examples/megatron/configs/MI300X/deepseek_v2_lite-BF16-pretrain.yaml \
    --target-nodes 6
```

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

### Performance Projection

```
====================================================================================================
[Primus:Performance Projection] Configuration Summary:
  Benchmark Config: PP=1, EP=8, TP=1, CP=1, DP=1 (1 node)
  Target Config: PP=1, EP=8, TP=1, CP=1, DP=4 (4 nodes)

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

## Background

Training large language models (LLMs) requires careful orchestration of multiple parallelism strategies — Tensor Parallelism (TP), Pipeline Parallelism (PP), Expert Parallelism (EP), Context Parallelism (CP), and Data Parallelism (DP). Each strategy trades off memory, compute, and communication differently. The combinatorial space of possible configurations makes it impractical to try every setup on actual hardware.

Existing approaches often rely on rules of thumb or trial-and-error: launch a full training run, observe whether it OOMs, measure throughput, tweak settings, and repeat. For models at the 100B+ parameter scale running across dozens of nodes, each iteration of this loop can waste hours of cluster time.

Primus projection takes a different approach: **estimate first, then run**. The memory projection uses analytical formulas derived from model architecture and parallelism configuration to predict per-GPU memory usage. The performance projection benchmarks representative layers on a configurable number of GPUs — from a single GPU up to a full node — and then analytically projects performance to arbitrary multi-node configurations using communication models and pipeline schedule simulation.

## Primus Projection

The projection tool operates in two modes, each targeting a different stage of the planning process.

| Mode | Command | What it does |
|------|---------|--------------|
| **Memory** | `projection memory` | Estimates per-GPU memory (parameters, optimizer, activations) using analytical formulas |
| **Performance** | `projection performance` | Benchmarks layers on a configurable number of GPUs, then projects training time to multi-node clusters |

Our main contributions can be summarized as follows:

- Provide a **hierarchical memory profiler** that mirrors the model's module structure — computing parameter counts, optimizer state, and activation memory per component, then aggregating bottom-up to produce accurate per-GPU memory estimates.

- Implement a **multi-mode performance projection** engine supporting GPU benchmarking, pure analytical simulation (via [Origami](https://github.com/ROCm/rocm-libraries/tree/develop/shared/origami/python) and SDPA simulators), and side-by-side comparison — enabling capacity planning even without GPU access.

- Provide **communication modeling** that captures collectives (AllReduce, All-to-All, P2P) in the benchmark when they fit within the available GPUs, and falls back to analytical models — accounting for intra-node vs. inter-node topology differences — for communication outside the benchmark scope.

- Integrate a **pipeline schedule simulator** supporting 1F1B, interleaved, and zero-bubble schedules to precisely account for pipeline bubble overhead in projected performance.


### Memory Projection

The memory projection estimates **per-GPU memory** by analytically computing three components: parameter memory, optimizer state memory, and activation memory. It uses a hierarchical profiler system that mirrors the model's module structure.

#### Hierarchical Profiler Architecture

Each component of the model has a corresponding profiler that computes its contribution:

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
│   └── MoEMLPProfiler             — routed experts + shared expert
├── LayerNormProfiler              — final layer norm (last stage only)
├── OutputLayerProfiler            — language model head (last stage only)
└── LossProfiler                   — cross-entropy loss (last stage only)
```

Each profiler implements two key methods:
- `estimated_num_params(rank)` — parameter count (total if `rank=None`, per-GPU if rank given)
- `estimated_activation_memory(batch_size, seq_len)` — activation bytes for one microbatch

#### Parameter and Optimizer Memory

Parameters are computed per component and summed across all layers assigned to a GPU's pipeline stage. The total bytes per parameter depend on training precision and optimizer sharding:

```
bytes_per_param = weight_bytes + gradient_bytes + optimizer_bytes

Where:
  weight_bytes    = 2      (BF16 weights)
  gradient_bytes  = 2      (BF16 gradients)
  optimizer_bytes = 10/DP  (FP32 master weights + Adam m + Adam v, sharded across DP)
                  = (2 + 4 + 4) / DP
```

#### Activation Memory

Activation memory is the memory needed to store intermediate tensors for the backward pass. The dominant contributor in MoE models is the MoE MLP, where each token is routed to `topk` experts, multiplying the activation footprint:

```
MoE MLP activation = tokens × (topk + N_shared) × (H + 3×FFN_e) × 2 bytes
```

For a large MoE model (MBS=4, S=16384, CP=4, H=8192, FFN_e=2048, topk=36), a single MoE layer's activation is ~16.19 GB — compared to ~0.51 GB for attention.

#### Pipeline Schedule Memory Scaling

With pipeline parallelism, multiple microbatches are in-flight simultaneously. The peak activation memory depends on the pipeline schedule:

```
total_activation = base_activation × PP × interleaved_penalty × ga_saving
```

Where `interleaved_penalty = 1 + (PP - 1) / (PP × VPP)` accounts for VPP overhead, and `ga_saving` handles cases where gradient accumulation steps are fewer than PP stages.

#### Recomputation Support

Activation recomputation trades compute for memory. With full recompute, a MoE layer's stored activation drops from ~18 GB to just `sbh` ≈ 0.25 GB (only the input checkpoint). The profiler models partial recomputation as well, where only a subset of layers per VPP stage are recomputed.


### Performance Projection

The performance projection tool estimates training throughput on multi-node clusters by benchmarking on a configurable number of GPUs — measuring both compute and communication where possible — and using analytical models only for what falls outside the benchmark scope. A key design goal is flexibility: you can benchmark on as few GPUs as you have available — even a single GPU — and project performance to arbitrary multi-node target configurations.

#### Flexible Benchmarking

The `--benchmark-gpus` flag controls how many GPUs are used for the benchmarking phase. By default, benchmarking uses a full node (`GPUS_PER_NODE` GPUs). When set to a lower value, the tool enables **sub-node benchmarking**: parallelism dimensions are automatically reduced to fit the benchmark GPU count (in the order PP → EP → TP), and the differences are analytically compensated during projection.

This means you can benchmark on 1 GPU and project to hundreds of nodes — the tool handles the parallelism rescaling and communication overhead modeling automatically.

#### Why a Hybrid Approach?

A natural question is: why not model everything analytically? The key insight behind Primus's design is that **the hardest part of performance prediction — what FLOPs the hardware actually delivers under a real workload — is precisely the part that is easiest to measure and hardest to model**. The guiding principle is simple: **measure what you can, simulate what you can't**. When the benchmark GPUs can accommodate a parallelism dimension (e.g., TP AllReduce within a node, EP All-to-All within the benchmark config), the communication is measured alongside compute. Only communication that falls outside the benchmark scope — inter-node traffic, or overhead from parallelism dimensions that had to be reduced — is estimated analytically. This hybrid approach achieves higher accuracy with less calibration effort than a purely analytical methodology.

Here are the main reasons the hybrid approach outperforms pure analytical modeling:

**1. Immunity to the Peak-vs-MAF Gap.** Modern GPUs dynamically adjust clock frequency during kernel execution to stay within their power envelope. The theoretical peak FLOPs use the maximum boost clock, but under sustained dense matrix workloads (which dominate LLM training), the GPU throttles to a lower operating frequency. The resulting Max-Achievable FLOPs (MAF) can be significantly lower than the peak — preliminary estimates suggest a 44–70% gap on current-generation hardware (see [Understanding Peak, Max-Achievable & Delivered FLOPs](https://rocm.blogs.amd.com/software-tools-optimization/Understanding_Peak_and_Max-Achievable_FLOPS/README.html)). A pure analytical or roofline-based model typically uses peak FLOPs (or a manually tuned efficiency factor) in its denominator, leading to systematically inflated predictions. By benchmarking real layers, Primus captures the actual operating frequency under the specific workload's power profile. Crucially, this frequency behavior is workload-dependent — different layer types (attention vs. MLP vs. MoE expert GEMMs) have different compute densities and therefore different sustained frequencies. Benchmarking captures this per-layer variation automatically.

**2. Measure What You Can, Simulate What You Can't.** The hybrid approach maximizes measurement coverage: when the benchmark configuration can accommodate a parallelism dimension, its communication is captured in the benchmark alongside compute. For example, if you benchmark on 8 GPUs with TP=8, the TP AllReduce is measured on real hardware. If you benchmark with EP=4, the EP All-to-All dispatch/combine is measured. Only the communication that *cannot* be captured — because the target configuration requires more GPUs than the benchmark (e.g., inter-node DP AllReduce, PP P2P across nodes, or communication for parallelism dimensions that were reduced to fit the benchmark) — is estimated analytically. Communication modeling itself is not trivial: real collective performance depends on topology, congestion, message sizes, and protocol choices. But for the components that must be modeled, communication collectives are more analytically tractable than compute kernels — their performance is dominated by bandwidth and latency with predictable message sizes. Pipeline scheduling follows deterministic rules that can be simulated exactly given per-stage compute times. Per-layer compute, on the other hand, depends on kernel implementation quality, memory system behavior, frequency throttling, operator fusion, and framework overhead — factors that are difficult to model analytically with high fidelity.

**3. Robustness to Software Stack Variations.** Every update to the ROCm software stack — a new hipBLASLt release with improved GEMM kernels, a CK attention kernel update, a Triton recompile with different tiling — shifts the achievable performance for each operation. A pure analytical model would require recalibration against each software version. The benchmark approach is inherently version-aware: it runs the actual kernels installed on the system and measures what they deliver, automatically reflecting the current software stack's performance.

**4. Captures Real Grouped GEMM Behavior for MoE Models.** For MoE models, the routed expert computation uses grouped GEMMs, whose performance is significantly harder to model analytically. The achieved efficiency depends on the number of experts and topk routing, token-to-expert distribution (affecting padding and load imbalance), the specific grouped GEMM kernel implementation (CK, hipBLASLt, AITER), and wave quantization effects from non-uniform sub-problem sizes. Grouped GEMMs often operate well below the roofline due to the overhead of managing many small sub-problems. Benchmarking sidesteps this by measuring actual grouped GEMM execution on representative problem shapes.

**5. Accounts for Framework and Runtime Overhead.** Real training iterations include overhead beyond raw kernel execution: PyTorch dispatch latency, memory allocator behavior, kernel launch overhead, CUDA/HIP stream synchronization, and torch.compile optimization effects. These overheads are present in benchmark measurements but absent from an analytical model that only considers the mathematical operations. For large models with many layers, these per-kernel overheads accumulate and can represent a non-trivial fraction of iteration time.

**6. Transparent Validation via Side-by-Side Comparison.** Primus offers a `--profiling-mode both` option that runs the benchmark and the pure analytical simulation (Origami + SDPA simulator) side by side and prints a comparison table. This allows users to quantify exactly how much accuracy the analytical path sacrifices, validate that the analytical backends remain calibrated as hardware and software evolve, and make informed decisions about when the pure simulation mode can be trusted for capacity planning.

The following table summarizes the comparison:

| Factor | Pure Analytical | Hybrid (Primus) |
|--------|----------------|-----------------|
| GPU frequency under load | Must assume or estimate; prone to overestimation | Captured automatically via real measurement |
| Software stack changes | Requires manual recalibration | Automatically reflects current kernel performance |
| Grouped GEMM for MoE | Difficult to model accurately | Measured directly with real kernels |
| Framework overhead | Not captured | Included in benchmark measurements |
| Communication modeling | Must model analytically | Measured when within benchmark scope; analytical fallback for the rest |
| Pipeline scheduling | Can simulate well | Same — uses pipeline schedule simulator |
| No-GPU capacity planning | Supported | Supported via `--profiling-mode simulate` fallback |
| Cross-validation | Not available | `--profiling-mode both` for side-by-side comparison |

#### Profiling Modes

The tool supports three profiling modes, making it usable across different environments:

| Mode | GPU Required | What it does |
|------|-------------|--------------|
| `benchmark` (default) | **Yes** | Runs real GPU kernels on the benchmark GPUs and measures forward/backward times |
| `simulate` | **No** | Uses Origami (GEMM) and SDPA Simulator (attention) analytical backends — no GPU needed |
| `both` | **Yes** | Runs both side-by-side for accuracy comparison |

When you **don't have access to a GPU** (e.g., capacity planning on a laptop), the `simulate` mode enables full training projection entirely on CPU.

#### Simulation Backends

Two analytical backends power the simulation mode:

**Origami (GEMM Backend)** — [Origami](https://github.com/ROCm/rocm-libraries/tree/develop/shared/origami/python) is an open-source tool (part of the ROCm ecosystem) that provides analytical performance modeling for GEMM kernels on AMD GPUs. Primus uses Origami to predict execution times for all GEMM operations — attention projections (Q, K, V, O), MLP (gate, up, down), MoE expert GEMMs, embedding, and output layers. It models the GPU's Compute Units (CUs), memory hierarchy, and tile-level execution, with built-in hardware profiles for architectures like MI300X, MI325X, and MI355X.

**SDPA Simulator (Attention Backend)** — Models Flash Attention v3 (FAv3) kernel execution analytically using Origami's 1-CU tile-level model. Flash Attention is a fused kernel where QKᵀ, softmax, and PV execute sequentially within each workgroup. The simulator computes `total_time = (per-tile-QKᵀ + per-tile-PV) × num_waves`, capturing wave quantization and per-tile efficiency. It also models the backward dQ atomic overhead from `buffer_atomic_add_f32` accumulation across KV-workgroups.

#### How It Works

The performance projection follows a multi-step pipeline:

1. **Configuration Reduction** — If the target parallelism configuration requires more GPUs than are available for benchmarking, the tool automatically reduces the config to fit. Parallelism dimensions are reduced in a fixed priority order:
   - **PP → 1** first (easiest to add back — overhead estimated via the pipeline schedule simulator)
   - **EP reduced** next if still doesn't fit (MoE compute stays accurate since experts-per-rank is preserved; only A2A is replaced analytically)
   - **TP reduced** last if PP and EP reduction were not sufficient (compute is scaled by `benchmark_tp / target_tp` and TP AllReduce overhead is added analytically)

   The benchmark GPU count is controlled by `--benchmark-gpus` (defaults to `GPUS_PER_NODE`). This enables benchmarking on as few as 1 GPU and projecting to multi-node configurations.

2. **Layer Benchmarking** — The tool benchmarks each transformer layer type (dense attention layers, MoE layers) on the available benchmark GPUs by measuring forward and backward pass times separately. Only representative layers are benchmarked (1 dense + 1 MoE) for efficiency.

3. **Extrapolation to Full Model** — Per-layer times are multiplied by the total layer count, accounting for the model's MoE layer frequency pattern. If TP was reduced, compute times are scaled by `benchmark_tp / target_tp` and the TP AllReduce overhead delta is added.

4. **Pipeline Simulation** — For PP > 1, the pipeline schedule simulator reconstructs the full 1F1B, interleaved, or zero-bubble schedule with proper send/receive synchronization, calculating step time and bubble ratio.

5. **Communication Overhead** — Communication that was already captured in the benchmark (e.g., intra-node TP AllReduce, EP All-to-All within the benchmark config) is included in the measured times. For communication outside the benchmark scope — inter-node DP gradient AllReduce, P2P for pipeline stages, or overhead from reduced parallelism dimensions — analytical models estimate the cost, differentiating between intra-node and inter-node links.

6. **Multi-node Scaling** — The final projection combines all components:

```
Projected_Time = Base_Time × (Min_DP / Target_DP) + Communication_Overhead
```

#### Communication Modeling

When a parallelism dimension fits within the benchmark GPU count, its communication is **measured** as part of the benchmark — no analytical modeling needed. For communication that falls outside the benchmark scope (e.g., inter-node collectives, or overhead from reduced parallelism dimensions), the tool uses analytical models, selecting the best algorithm for each:

| Collective | Used By | Model |
|------------|---------|-------|
| AllReduce | TP, DP (gradients) | Best of Ring/Hypercube/Bruck/Single-shot |
| All-to-All | EP (MoE dispatch/combine) | Pairwise exchange, accounts for topology |
| P2P Send/Recv | PP (activations) | Point-to-point latency + bandwidth |

Communication times differ significantly based on whether they are **intra-node** (fast, e.g., xGMI/NVLink) or **inter-node** (slower, e.g., InfiniBand/RoCE). These hardware parameters are customizable — users can provide a hardware configuration file via `--hardware-config` to match their specific cluster topology. See [`custom_hardware_example.yaml`](../../examples/hardware_configs/custom_hardware_example.yaml) for the format.

#### Pipeline Schedule Simulator

The pipeline simulator supports multiple scheduling algorithms:

| Algorithm | Description |
|-----------|-------------|
| **1F1B** | Standard one-forward-one-backward schedule |
| **Interleaved 1F1B** | Multiple virtual chunks per rank (VPP > 1) for reduced bubble ratio |
| **Zero-Bubble** | Separates backward into B (input gradient) + W (weight gradient) for minimal bubbles |

Zero-bubble scheduling minimizes pipeline bubbles by splitting the backward pass. B computes gradients w.r.t. input activations and W computes gradients w.r.t. weights. Since W doesn't depend on receiving gradients from the next stage, it enables more flexible scheduling. The Primus pipeline simulator supports both a simple F-B-W pattern and Megatron's ILP-based memory-aware scheduler. For a deep dive into how these pipeline scheduling algorithms are designed and implemented in Primus, see the [Primus-pipeline blog](../primus_pipeline/primus_pipeline.md).


### Parallelism Modeling

The projection tool models how each parallelism dimension affects performance:

**Tensor Parallelism (TP)** — Splits individual layer weights across GPUs within a node. When benchmarking runs at the same TP as the target, TP communication (AllReduce after each layer) is already captured in the benchmark. However, when the benchmark GPU count is too small to accommodate the target TP (e.g., benchmarking on 1 GPU for a TP=8 config), TP is reduced during benchmarking as the last resort (after PP and EP). The projection then compensates by: (1) scaling per-GPU compute by `benchmark_tp / target_tp`, and (2) adding the TP AllReduce overhead delta analytically. When GPU-measured AllReduce data is available, it anchors the analytical model for better accuracy.

**Pipeline Parallelism (PP)** — Distributes layers across pipeline stages. PP is always the first dimension reduced during benchmarking (set to 1). A pipeline scheduler simulator then reconstructs the full target schedule analytically, accounting for bubble overhead and microbatch interleaving.

**Expert Parallelism (EP)** — Distributes MoE experts across GPUs. EP is the second dimension reduced if needed. The tool estimates inter-node All-to-All overhead by comparing the A2A time for benchmark EP (intra-node) vs. target EP (potentially inter-node). MoE compute stays accurate because experts-per-rank is preserved during EP rescaling.

**Context Parallelism (CP)** — For MoE models, CP is folded into EP via MoE Parallel Folding — CP ranks are a subset of EP ranks, so the minimum GPU count remains `TP × PP × EP` instead of `TP × PP × EP × CP`. For dense models, CP is an independent axis. CP is kept unchanged during benchmarking.

**Data Parallelism (DP)** — Provides linear speedup by processing more batches in parallel. Gradient AllReduce is overlapped with backward computation by default.


## Validation: Projected vs. Measured Results

To validate the projection tool's accuracy, we compared projected performance against measured results published on the [AMD ROCm Performance Results](https://www.amd.com/en/developer/resources/rocm-hub/dev-ai/performance-results.html#ai-training) page. The workflow is straightforward: benchmark on a single node, then project to 8 nodes and compare against the published multi-node measurements. The projections were obtained by providing the corresponding hardware configuration files via `--hardware-config`.

### MI325X — Dense Models (Llama 3.1)

| Model | Precision | Batch | SeqLen | FSDP | TP | PP | CP | EP | 1-Node (tok/s/GPU) | Measured 8-Node (tok/s/GPU) | Projected 8-Node (tok/s/GPU) | Error |
|-------|-----------|-------|--------|------|----|----|----|----|--------------------|-----------------------------|------------------------------|-------|
| Llama 3.1 8B | FP8 | 2 | 8192 | No | 1 | 1 | 1 | 1 | 16,224 | 16,186 | 17,351 | +7.20% |
| Llama 3.1 70B | FP8 | 4 | 8192 | Yes | 1 | 1 | 1 | 1 | 1,135 | 1,726 | 1,818 | +5.33% |
| Llama 3.1 70B | BF16 | 1 | 8192 | Yes | 1 | 1 | 1 | 1 | 1,135 | 1,174 | 1,066 | -9.20% |

### MI355X — MoE Models (Mixtral 8×22B)

| Model | Precision | Batch | SeqLen | TP | PP | CP | EP | Measured 8-Node (tok/s/GPU) | Projected 8-Node (tok/s/GPU) | Error |
|-------|-----------|-------|--------|----|----|----|----|-----------------------------|-----------------------------|-------|
| Mixtral 8×22B | BF16 | 2 | 8192 | 1 | 4 | 1 | 8 | 3,534 | 3,232 | -8.5% |

### Key Takeaways

- **All projections are within 10% of measured results**, spanning both dense (Llama) and MoE (Mixtral) architectures, FP8 and BF16 precision, and two different GPU generations (MI325X, MI355X).
- The Mixtral result exercises the full projection pipeline: PP reduction (benchmark PP=1, target PP=4), EP All-to-All modeling (with DeepEP), and pipeline schedule simulation — yet still achieves <10% error.


## Assumptions and Limitations

### Assumptions

1. **DP Weak Scaling** — DP scaling assumes weak scaling (constant micro-batch size per GPU); the DP gradient AllReduce overhead is modeled analytically
2. **Communication Model** — Bandwidth efficiency uses a constant factor; all NICs are used in parallel for inter-node traffic
3. **Pipeline Bubbles** — B/W split is 50/50 for zero-bubble scheduling; P2P communication time is small relative to compute
4. **Gradient AllReduce** — By default overlapped with compute; if disabled, added to critical path
5. **MoE All-to-All** — All-to-All time scales with EP size; per-peer latency overhead is constant

### Limitations

1. **Benchmark Accuracy with Reduced Parallelism** — Benchmarking with reduced PP/EP/TP may not capture all behaviors (e.g., GEMM efficiency differences at different TP levels)
2. **Communication Contention** — Model doesn't account for network contention; assumes dedicated bandwidth per collective
3. **Memory Pressure** — Memory impact on performance not fully modeled; activation recomputation overhead not considered in performance
4. **Hardware Heterogeneity** — Assumes homogeneous nodes; GPU frequency variations not modeled


## Tips

- **Start with memory projection**: Run `projection memory` first to verify your model fits in GPU memory before benchmarking performance.
- **Benchmark with what you have**: Use `--benchmark-gpus` to run benchmarks on any number of GPUs (even 1) and project to multi-node. The tool handles parallelism downscaling (PP → EP → TP) and analytical upscaling automatically.
- **No GPU? Use simulation**: With `--profiling-mode simulate`, you can run performance projection entirely on CPU. This is useful for capacity planning without GPU access.
- **Validate simulation accuracy**: Use `--profiling-mode both` to compare simulation against real benchmarks.
- **Understand scaling limits**: DP scaling is limited by `global_batch_size / micro_batch_size`. If you run out of microbatches, adding more nodes won't help.
- **Pipeline scaling**: With PP > 1, layers don't need to divide evenly across stages. The tool distributes remainder layers to the first stages (e.g., 61 layers with PP=4 → [16, 15, 15, 15]). You can also supply explicit per-stage layer counts via `decoder_first_pipeline_num_layers`, `decoder_last_pipeline_num_layers`, or `pipeline_model_parallel_layout`.
- **Recomputation trade-off**: Full recompute dramatically reduces activation memory (e.g., 18 GB → 0.25 GB per MoE layer) at the cost of ~33% more compute.
- **MoE activation dominance**: For MoE models, the MoE MLP activation (scaled by `topk`) typically dominates the per-layer activation budget.


## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED "AS IS" WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
