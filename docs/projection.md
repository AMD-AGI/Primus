# Performance Projection

Primus includes a performance projection tool that benchmarks transformer layers on a single node and projects training iteration times to multi-node configurations.

- **User-facing entry**: `primus-cli … -- projection performance [options]`
- **Implementation entrypoint**: `primus/cli/subcommands/projection.py`
- **Core logic**: `primus/core/projection/performance_projection/projection.py`

## Overview

The performance projection tool:

1. **Benchmarks** transformer layers on a single node to measure forward/backward pass times
2. **Simulates** pipeline parallelism scheduling (including zero-bubble optimization)
3. **Projects** performance to multi-node configurations by modeling:
   - Data Parallelism (DP) scaling
   - Gradient AllReduce communication overhead
   - Expert Parallelism (EP) All-to-All communication overhead
   - Inter-node vs intra-node communication differences

This allows you to estimate training performance on larger clusters without actually running on them.

## Quick Start

Run a basic performance projection for the minimum required nodes:

```bash
export NNODES=1
export HSA_NO_SCRATCH_RECLAIM=1

bash runner/primus-cli direct --script primus/cli/main.py -- \
    projection performance \
    --config examples/megatron/configs/MI300X/deepseek_v2_lite-BF16-pretrain.yaml
```

Project performance to a specific number of nodes:

```bash
export NNODES=1
export HSA_NO_SCRATCH_RECLAIM=1

bash runner/primus-cli direct --script primus/cli/main.py -- \
    projection performance \
    --config examples/megatron/configs/MI300X/deepseek_v2_lite-BF16-pretrain.yaml \
    --target-nodes 4
```

## Command Syntax

```bash
primus-cli [global-options] <mode> [mode-args] -- projection performance [options]
```

### Options

| Option | Type | Description |
|--------|------|-------------|
| `--config` | string | Path to the Primus YAML configuration file (required) |
| `--target-nodes` | int | Target number of nodes for projection. Defaults to minimum required by parallelism config |
| `--hardware-config` | string | Path to YAML file with custom hardware parameters for communication modeling |

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

## How It Works

### 1. Configuration Reduction

If the parallelism configuration requires multiple nodes (e.g., PP=3 needs 3 nodes), the tool automatically reduces the config for single-node benchmarking:

- **Pipeline Parallelism (PP)**: Reduced to fit on 1 node, PP overhead estimated analytically
- **Expert Parallelism (EP)**: Reduced if needed, All-to-All overhead added back

### 2. Layer Benchmarking

The tool benchmarks each transformer layer type:
- Dense attention layers
- MoE (Mixture of Experts) layers
- Measures forward and backward pass times separately

### 3. Pipeline Simulation

For PP > 1, the tool simulates the pipeline schedule to account for:
- Pipeline bubble overhead
- Microbatch interleaving
- Zero-bubble scheduling (if enabled)

### 4. Data Parallel Scaling

The projection models how performance scales with additional nodes:

```
Projected Time = (Base Time / DP_scaling_factor) + Communication Overheads
```

## Scaling Mechanisms

The tool models the following parallelism dimensions and their communication patterns:

### Tensor Parallelism (TP)

**What it does**: Splits individual layer weights across GPUs within a node.

**How it's modeled**: TP communication (AllReduce after each layer) is **already captured** in the single-node benchmark because benchmarking runs with the target TP configuration. No additional modeling needed.

**Communication**: AllReduce within TP group (typically intra-node, fast).

### Pipeline Parallelism (PP)

**What it does**: Distributes layers across pipeline stages. Each stage processes microbatches in sequence.

**How it's modeled**: 
- If PP > 1 but only 1 node available for benchmarking, PP is reduced to 1 for benchmarking
- A **pipeline scheduler simulator** (`simulator.py`) reconstructs the full pipeline schedule
- Simulates the actual 1F1B or zero-bubble schedule with proper send/receive synchronization
- Accounts for pipeline bubble overhead and microbatch interleaving

### Expert Parallelism (EP)

**What it does**: Distributes MoE experts across GPUs. Each GPU holds a subset of experts.

**How it's modeled**:
- If EP spans multiple nodes, the tool estimates **inter-node All-to-All overhead**
- Compares All-to-All time for benchmark EP (intra-node) vs target EP (potentially inter-node)
- Adds the overhead difference to each MoE layer's forward/backward time

**Communication**: All-to-All for token dispatch (before expert computation) and combine (after).

```
All-to-All Message Size = tokens × hidden_size × top_k × 2 (BF16)
```

### Data Parallelism (DP)

**What it does**: Replicates the model across DP groups. Each group processes different data batches.

**How it's modeled**:
- DP provides **linear speedup** by processing more batches in parallel
- Scaling factor = `target_DP / baseline_DP`

**Communication**: Gradient AllReduce across all DP ranks.

```
Gradient AllReduce Size = num_params × 4 (FP32 gradients)
```

**Optimization**: If `overlap_grad_reduce=True` (default), gradient AllReduce is overlapped with backward computation and not on the critical path.

### Context Parallelism (CP)

**What it does**: Splits sequence length across GPUs for long-context training.

**How it's modeled**: CP affects the GPU topology for communication routing. Currently included in minimum GPU requirements calculation.

## Communication Modeling

The tool uses analytical models to estimate collective communication times:

| Collective | Used By | Model |
|------------|---------|-------|
| AllReduce | TP, DP (gradients) | Best of Ring/Hypercube/Bruck/Single-shot |
| All-to-All | EP (MoE dispatch/combine) | Pairwise exchange, accounts for topology |
| P2P Send/Recv | PP (activations) | Point-to-point latency + bandwidth |

Communication times differ significantly based on:
- **Intra-node**: Fast (e.g., NVLink, xGMI)
- **Inter-node**: Slower (e.g., InfiniBand, RoCE) 

### Key Concepts

#### Minimum Nodes Required

The minimum nodes required is determined by:
```
Min Nodes = ceil(TP × PP × EP × CP / GPUs_per_node)
```

#### Scaling Behavior

- **DP scaling**: Linear speedup. Doubling DP halves iteration time (minus communication overhead).
- **PP scaling**: Happens in multiples of pipeline replicas. With PP=3, you need 3, 6, 9... nodes to increase scaling.
- **EP scaling**: Divides the experts on EP nodes.

## Example Output

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

## Tips

- **Start with 1 node**: Always benchmark on 1 node first to establish baseline performance.
- **Understand scaling limits**: DP scaling is limited by global_batch_size / micro_batch_size. If you run out of microbatches, adding more nodes won't help.
- **Check minimum nodes**: If your config requires multiple nodes (e.g., PP=4 with 8 GPUs/node), projection will automatically reduce PP for benchmarking.
- **Pipeline scaling**: With PP > 1, you can only scale in multiples of the pipeline replica size.

## Related Documentation

- [Benchmark Suite](benchmark.md) - For microbenchmarking individual operations
- [Quickstart Guide](quickstart.md) - Getting started with Primus
