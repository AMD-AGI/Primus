# Memory and Performance Projection

Primus projection tools estimate **per-GPU memory** and **training throughput** for large-scale distributed jobs without requiring the full target cluster. Two modes are available: analytical **memory** projection and **performance** projection that combines profiling with simulation.

**Implementation:** `primus/cli/subcommands/projection.py`

| Mode | Command | Role |
|------|---------|------|
| **Memory** | `projection memory` | Estimates per-GPU memory (parameters, optimizer state, activations) using analytical formulas. |
| **Performance** | `projection performance` | Benchmarks on a single node (or sub-node), then projects training time to multi-node configurations. |

**Core logic**

- Memory: `primus/core/projection/memory_projection/`
- Performance: `primus/core/projection/performance_projection/`

Related: [Benchmark suite](./benchmarking.md), [Preflight diagnostics](./preflight.md), [Megatron parameters](../03-configuration-reference/megatron-parameters.md).

---

## Memory projection

### Quick start

```bash
export NNODES=1
export HSA_NO_SCRATCH_RECLAIM=1

bash runner/primus-cli direct --script primus/cli/main.py -- \
  projection memory \
  --config examples/megatron/configs/MI300X/deepseek_v2_lite-BF16-pretrain.yaml
```

Adjust `--config` to your experiment YAML. Memory estimation is analytical; the CLI still expects a normal Primus launch path (including distributed initialization where applicable).

### What it estimates

| Component | Meaning |
|-----------|---------|
| **Parameter memory** | Model weights assigned to this GPU (respecting parallelism). |
| **Optimizer memory** | Optimizer state (for example Adam moments), accounting for sharding across data-parallel groups. |
| **Activation memory** | Activations retained for the backward pass for a given microbatch and sequence length. |

The tool walks a hierarchical profiler structure aligned with the model (embeddings, dense and MoE layers, output head, loss) and aggregates per-component contributions.

### How to interpret results

Console output includes per-component breakdowns and a summary such as parameter count, param+optimizer memory, activation memory for the configured batch size and sequence length, and a projected total. Use these to answer whether a configuration fits in HBM before you allocate large clusters.

---

## Performance projection

### Quick start

Minimum required nodes (derived from parallelism):

```bash
export NNODES=1
export HSA_NO_SCRATCH_RECLAIM=1

bash runner/primus-cli direct --script primus/cli/main.py -- \
  projection performance \
  --config examples/megatron/configs/MI300X/deepseek_v2_lite-BF16-pretrain.yaml
```

### How it works

1. **Profile** layer-level behavior on **one node** (or a subset of GPUs with automatic scaling rules).
2. **Simulate** pipeline scheduling, data parallelism, and communication using analytical models.
3. **Project** iteration time and tokens/s to a **target** node count when you specify one.

### Projecting to a specific node count

```bash
bash runner/primus-cli direct --script primus/cli/main.py -- \
  projection performance \
  --config examples/megatron/configs/MI300X/deepseek_v2_lite-BF16-pretrain.yaml \
  --target-nodes 4
```

If `--target-nodes` is omitted, the tool defaults to the **minimum** number of nodes implied by your parallelism configuration (TP, PP, EP, CP, GPUs per node).

### Parallelism overrides (environment)

You can override parallelism for what-if analysis:

```bash
export PRIMUS_TP=1
export PRIMUS_PP=3
export PRIMUS_EP=8

bash runner/primus-cli direct --script primus/cli/main.py -- \
  projection performance \
  --config examples/megatron/configs/MI300X/deepseek_v2_lite-BF16-pretrain.yaml \
  --target-nodes 6
```

---

## Command reference

### Syntax

```bash
primus-cli [global-options] <mode> [mode-args] -- projection {memory,performance} [options]
```

### Shared options (both modes)

| Option | Description |
|--------|-------------|
| `--config` / `--exp` | Path to the Primus YAML configuration (**required**). |
| `--data_path` | Data directory (default `./data` when included on the parser). |
| `--backend_path` | Optional Megatron/TorchTitan import path appended to `PYTHONPATH`. |
| `--export_config` | Write the merged resolved config to a file. |

### Performance-only options

| Option | Description |
|--------|-------------|
| `--target-nodes` | Target number of nodes for scaling projection. Defaults to the minimum nodes required by TP/PP/EP/CP and GPUs per node. |
| `--benchmark-gpus` | Use fewer than `GPUS_PER_NODE` GPUs for benchmarking; results are scaled analytically back to a full node. |
| `--hardware-config` | YAML file with hardware parameters for communication modeling. |
| `--profiling-mode` | `benchmark` (default, uses GPU), `simulate` (analytical / Origami GEMM + SDPA models, no GPU), or `both` (side-by-side). |
| `--gemm-backend` | GEMM simulation backend when profiling is simulated (`origami`). |
| `--gpu-arch` | Target architecture for simulation (for example `mi300x`, `gfx942`, `mi355x`, `gfx950`); can use `PRIMUS_GPU_ARCH`. |
| `--gpu-clock-mhz` | Override GPU clock in MHz for simulation; can use `PRIMUS_GPU_CLOCK_MHZ`. |

---

## Assumptions and limitations

### Assumptions (performance projection)

1. **Data-parallel scaling** — Compute time scales with ideal weak-scaling assumptions versus data-parallel width.
2. **Communication Model** — Uses simplified bandwidth and latency models (defaults such as efficiency factors may apply).
3. **Pipeline scheduling** — Bubble and overlap behavior is modeled with fixed splits; real frameworks may differ.
4. **Gradients and MoE** — Gradient all-reduce overlap and MoE all-to-all behavior follow the implemented model (for example overlap flags, EP scaling).

### Limitations

1. **Single-node benchmark accuracy** — Reduced PP/EP on the benchmark GPU count may not capture every production behavior.
2. **Contention** — Network contention between jobs is not modeled.
3. **Memory vs speed** — Activation recomputation reduces memory but adds compute; performance projection may not fully reflect that trade-off unless modeled.
4. **Heterogeneity** — Assumes homogeneous nodes; GPU frequency drift across nodes is not modeled.

---

## Tips

1. Run **`projection memory`** first to confirm a configuration is feasible in HBM before spending time on performance projection.
2. Always establish a **single-node** baseline before interpreting multi-node projections.
3. **Data-parallel scaling** is bounded by batching: if you run out of microbatches (`global_batch_size` / `micro_batch_size`), adding nodes may not increase throughput.
4. If the YAML **requires** multiple nodes (for example large PP), the performance path may automatically reduce parallelism for benchmarking and restore it analytically—read the console summary carefully.
5. **No GPU available:** use `--profiling-mode simulate` for CPU-side analytical timing.
6. **Validate models:** use `--profiling-mode both` to compare GPU benchmark timing with simulation on the same config.
7. For **MoE** models, activation memory from MoE layers often dominates; memory projection highlights when recomputation is worth considering.

---

## See also

- [Benchmark suite](./benchmarking.md)
- [Preflight diagnostics](./preflight.md)
- [Post-training workflows](./posttraining.md)
- [Megatron parameters](../03-configuration-reference/megatron-parameters.md)
