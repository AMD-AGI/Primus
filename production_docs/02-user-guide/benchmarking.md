# Benchmark Suite

Primus ships microbenchmarks for GPU compute and distributed communication. They are exposed as the `benchmark` subcommand of the Primus CLI. Use them to sanity-check a node or cluster before long training jobs.

**Implementation:** `primus/cli/subcommands/benchmark.py` (initializes distributed execution, runs the selected suite, then finalizes).

Related documentation: [Preflight diagnostics](./preflight.md) (broader cluster checks), [Memory and performance projection](./projection.md) (training-scale estimates), [Installation](../01-getting-started/installation.md) (environment setup).

---

## Overview and command syntax

```bash
primus-cli [global-options] <mode> [mode-args] -- benchmark <suite> [suite-specific-args]
```

- **`<mode>`** is typically `direct`, `container`, or `slurm` so that `WORLD_SIZE`, `RANK`, `MASTER_ADDR`, and related variables are set consistently.
- **`benchmark`** runs inside the Primus Python CLI; the runner wires up the process environment the same way as training.

The CLI also registers an `attention` suite; the subsections below cover **`gemm`**, **`gemm-dense`**, **`gemm-deepseek`**, **`strided-allgather`**, and **`rccl`**.

---

## Quick start

Single-node GEMM:

```bash
primus-cli direct -- benchmark gemm --M 4096 --N 4096 --K 4096 --dtype bf16 --duration 10
```

Multi-node RCCL on Slurm:

```bash
primus-cli slurm srun -N 4 -- benchmark rccl --op all_reduce --min-bytes 1M --max-bytes 128M
```

---

## Suite reference

### `gemm`

Single-shape general matrix multiply (GEMM) microbenchmark.

| Argument | Description |
|----------|-------------|
| `--M`, `--N`, `--K` | Matrix dimensions (defaults: 4096 / 4096 / 4096). |
| `--trans_a` | Transpose the A matrix. |
| `--trans_b` | Transpose the B matrix. |
| `--dtype` | `bf16`, `fp16`, `fp32`, or `fp8` (`fp8` requires torchao). Default: `bf16`. |
| `--duration` | Run duration in seconds (default: 10). |
| `--output-file` | Destination for results (`.md`, `.csv`, `.tsv`, `.jsonl`, `.jsonl.gz`). Default: `./gemm_report.md`. Use `-` or omit for Markdown on stdout. |

**Example**

```bash
primus-cli direct -- benchmark gemm --M 8192 --N 8192 --K 8192 --dtype bf16 --duration 10 --output-file ./gemm_report.md
```

---

### `gemm-dense`

Dense GEMM workload using Llama-like shape parameters (model-derived GEMMs).

| Argument | Description |
|----------|-------------|
| `--model` | Optional label (for example `Llama3.1_8B`). |
| `--seqlen` | Sequence length (default: 2048). |
| `--hidden-size` | Hidden size (default: 4096). |
| `--intermediate-size` | FFN intermediate size (default: 11008). |
| `--num-attention-heads` | Attention heads (default: 32). |
| `--num-key-value-heads` | KV heads (default: 32). |
| `--head-dim` | Per-head dimension (default: 128). |
| `--vocab-size` | Vocabulary size (default: 32000). |
| `--dtype` | `bf16`, `fp16`, `fp32`, or `fp8` (`fp8` requires torchao). Default: `bf16`. |
| `--mbs` | Microbatch size (default: 1). |
| `--duration` | Seconds per shape (default: 3). |
| `--output-file` | Report path (default: `./gemm-dense_report.md`). |

**Example**

```bash
primus-cli direct -- benchmark gemm-dense --model Llama3.1_8B --seqlen 4096 --dtype bf16
```

---

### `gemm-deepseek`

Dense GEMM workload using DeepSeek-style shapes (MoE / MLA-related dimensions).

| Argument | Description |
|----------|-------------|
| `--model` | Label (for example `Deepseek_V2`, `Deepseek_V3`). |
| `--seqlen` | Sequence length (default: 4096). |
| `--hidden-size` | Hidden size (default: 4096). |
| `--intermediate-size` | Dense FFN intermediate (default: 12288). |
| `--kv-lora-rank` | KV LoRA rank (default: 512). |
| `--moe-intermediate-size` | MoE expert intermediate (default: 1536). |
| `--num-attention-heads` | Attention heads (default: 64). |
| `--num-experts-per-tok` | Experts per token (default: 6). |
| `--n-routed-experts` | Number of routed experts (default: 128). |
| `--n-shared-experts` | Shared experts (default: 2). |
| `--q-lora-rank` | Optional Q LoRA rank. |
| `--qk-nope-head-dim`, `--qk-rope-head-dim`, `--v-head-dim` | Head dimensions for MLA-style attention (defaults: 128 / 64 / 128). |
| `--vocab-size` | Vocabulary size (default: 128256). |
| `--dtype` | `bf16` or `fp16` (default: `bf16`). |
| `--mbs` | Microbatch size (default: 1). |
| `--duration` | Seconds per shape (default: 3). |
| `--output-file` | Report path (default: `./gemm-deepseek_report.md`). |
| `--append` | Append to an existing report instead of overwriting. |

**Example**

```bash
primus-cli direct -- benchmark gemm-deepseek --model Deepseek_V3 --dtype bf16 --append
```

---

### `strided-allgather`

Strided all-gather microbenchmark (useful for multi-rank communication patterns).

| Argument | Description |
|----------|-------------|
| `--sizes-mb` | Comma-separated message sizes in MB per rank (default: `64,128,256`). |
| `--stride` | Rank stride for group formation (default: 8). |
| `--parallel` | Run multiple groups’ all-gathers in parallel. |
| `--iters` | Timed iterations per size (default: 50). |
| `--warmup` | Warmup iterations per size (default: 10). |
| `--dtype` | `fp16`, `bf16`, or `fp32` (default: `bf16`). |
| `--backend` | `nccl`, `gloo`, or `mpi` (default: `nccl`). |

**Example**

```bash
primus-cli slurm srun -N 2 -- benchmark strided-allgather --sizes-mb 64,128 --stride 8 --iters 50
```

---

### `rccl`

RCCL collective benchmark: sweeps message sizes and reports bandwidth and latency statistics.

| Argument | Description |
|----------|-------------|
| `--op` | One or more of: `all_reduce`, `broadcast`, `reduce_scatter`, `all_gather`, `alltoall` (default: `allreduce`). |
| `--sizes` | Explicit size list (for example `1K,2K,4K,8K,1M`). Overrides generated sweep. |
| `--min-bytes` | Minimum message size for generated sweep (default: `1K`). |
| `--max-bytes` | Maximum message size (default: `128M`). |
| `--num-sizes` | Number of points in generated sweep (default: 12). |
| `--scale` | `log2` or `linear` for generated sweeps (default: `log2`). |
| `--dtype` | `bf16`, `fp16`, or `fp32` (default: `bf16`). |
| `--warmup` | Warmup iterations (default: 20). |
| `--iters` | Timed iterations (default: 100). |
| `--repeat` | Repeat each `(op, size)` for stability (default: 1). |
| `--aggregate-repeat` | Emit an extra summary row aggregating repeat runs. |
| `--check` | Enable lightweight correctness checks. |
| `--output-file` | Report path (`.md`, `.csv`, `.tsv`, `.jsonl`, `.jsonl.gz`; default: `./rccl_report.md`). |
| `--append` | Append instead of overwrite. |
| `--per-rank` | Per-rank summary lines. |
| `--per-rank-file` | Path for per-rank stats (if empty, derived from `--output-file` with `_rank` suffix). |
| `--per-iter-trace` | Emit per-iteration trace (can be large). |
| `--trace-file` | Trace output path (if empty, derived from `--output-file`). |
| `--trace-limit` | Max iterations to record per `(op, size)`; `0` means all. |
| `--trace-ops` | Comma-separated ops to include in trace (empty = all). |
| `--trace-sizes` | Comma-separated sizes to include in trace (empty = all). |
| `--cluster` | Label for the report preamble (default: `PRIMUS_CLUSTER` or `amd-aig-poolside`). |

**Example**

```bash
primus-cli slurm srun -N 4 -- benchmark rccl --op all_reduce --min-bytes 1M --max-bytes 128M --dtype bf16
```

---

## Understanding results

- **GEMM suites** emit throughput-oriented metrics suitable for comparing dtypes, shapes, and durations across runs. Keep `duration` long enough to smooth variance on shared clusters.
- **`rccl`** reports collective latency and bandwidth across a size sweep; use it to verify inter-node behavior and to compare against expected NIC bandwidth.
- **Markdown / CSV / TSV / JSONL** output formats support post-processing in notebooks or CI; gzip JSONL is supported for large traces.

---

## Tips

1. **Distributed initialization:** If jobs hang or report uninitialized distributed state, launch through `primus-cli` (`direct` / `container` / `slurm`) rather than calling Python entrypoints manually without the right environment.
2. **Paths:** Prefer **absolute** paths for `--output-file` when using containers or Slurm so the working directory matches your expectations.
3. **Multi-node:** Use your scheduler integration (`primus-cli slurm …`) so rank and address assignment matches your cluster.
4. **Full cluster validation:** Combine targeted `benchmark` runs with [Preflight](./preflight.md) for host, GPU, network, and integrated perf checks.

---

## See also

- [Preflight diagnostics](./preflight.md)
- [Memory and performance projection](./projection.md)
- [Post-training workflows](./posttraining.md)
- [Installation](../01-getting-started/installation.md)
