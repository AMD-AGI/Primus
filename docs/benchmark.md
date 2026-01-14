# Benchmark Suite

Primus includes a small set of performance microbenchmarks under the `benchmark` subcommand.

- **User-facing entry**: `primus-cli … -- benchmark <suite> [suite-args]`
- **Implementation entrypoint**: `primus/cli/subcommands/benchmark.py`

## Quick start

Run a single-node GEMM benchmark:

```bash
primus-cli direct -- benchmark gemm --M 4096 --N 4096 --K 4096 --dtype bf16 --duration 10
```

Run a multi-node communication benchmark on Slurm:

```bash
primus-cli slurm srun -N 4 -- benchmark rccl --op all_reduce --min-bytes 1M --max-bytes 128M
```

## Command syntax

```bash
primus-cli [global-options] <mode> [mode-args] -- benchmark <suite> [suite-specific-args]
```

Notes:
- `benchmark` runs under the **Primus Python CLI** and is typically launched via the `primus-cli` runner (`direct` / `container` / `slurm`) so distributed environment variables are set correctly.
- The benchmark runner initializes distributed by calling `init_distributed()` and finalizes with `finalize_distributed()` (see `primus/cli/subcommands/benchmark.py`).

## Suites

The currently supported suites (as wired in `primus/cli/subcommands/benchmark.py`) are:

### `gemm`
GEMM microbenchmark for a single shape.

Common options (see `primus/tools/benchmark/gemm_bench_args.py`):
- `--M/--N/--K`: GEMM dimensions (default: 4096/4096/4096)
- `--dtype`: `bf16|fp16|fp32` (default: `bf16`)
- `--duration`: seconds (default: 10)
- `--output-file`: write results to a file (default: `./gemm_report.md`)

Example:

```bash
primus-cli direct -- benchmark gemm --M 8192 --N 8192 --K 8192 --dtype bf16 --duration 10
```

### `gemm-dense`
Dense-GEMM benchmark using model-derived shapes (e.g., Llama-like configs).

Common options (see `primus/tools/benchmark/dense_gemm_bench_args.py`):
- `--model`: optional label (e.g., `Llama3.1_8B`)
- `--seqlen`, `--hidden-size`, `--intermediate-size`, `--num-attention-heads`, ...
- `--dtype`: `bf16|fp16|fp32`
- `--mbs`: microbatch size
- `--duration`: seconds per shape
- `--output-file`: default `./gemm-dense_report.md`

Example:

```bash
primus-cli direct -- benchmark gemm-dense --model Llama3.1_8B --seqlen 4096 --dtype bf16
```

### `gemm-deepseek`
DeepSeek-style Dense-GEMM benchmark using DeepSeek-derived shapes.

Common options (see `primus/tools/benchmark/deepseek_dense_gemm_bench_args.py`):
- `--model`: label (e.g., `Deepseek_V3`)
- `--seqlen`, `--hidden-size`, MoE/LoRA-related shape controls
- `--dtype`: `bf16|fp16`
- `--mbs`, `--duration`
- `--output-file`: default `./gemm-deepseek_report.md`
- `--append`: append to existing report

Example:

```bash
primus-cli direct -- benchmark gemm-deepseek --model Deepseek_V3 --dtype bf16 --append
```

### `strided-allgather`
Strided allgather microbenchmark.

Common options (see `primus/tools/benchmark/strided_allgather_bench_args.py`):
- `--sizes-mb`: comma-separated sizes in MB (default: `64,128,256`)
- `--stride`: group stride (default: 8)
- `--parallel`: run multiple groups in parallel
- `--iters`, `--warmup`
- `--dtype`: `fp16|bf16|fp32`
- `--backend`: `nccl|gloo|mpi`

Example:

```bash
primus-cli slurm srun -N 2 -- benchmark strided-allgather --sizes-mb 64,128 --stride 8 --iters 50
```

### `rccl`
RCCL collective benchmark (sweep sizes, emit bandwidth/latency stats).

Common options (see `primus/tools/benchmark/rccl_bench_args.py`):
- `--op`: one or more collectives
- `--sizes`: explicit size list (overrides generated sweep)
- `--min-bytes/--max-bytes/--num-sizes/--scale`: configure generated sweep
- `--dtype`, `--warmup`, `--iters`
- `--repeat` and `--aggregate-repeat` (optional stability/summary)
- `--check`: lightweight correctness checks
- `--output-file`: default `./rccl_report.md` (supports `.md/.csv/.tsv/.jsonl[.gz]`)
- `--append`: append to file instead of overwriting
- `--per-rank` and trace options (`--per-iter-trace`, filters, output paths)

Example:

```bash
primus-cli slurm srun -N 4 -- benchmark rccl --op all_reduce --min-bytes 1M --max-bytes 128M --dtype bf16
```

## Tips

- If you see “distributed not initialized” / hangs, ensure you are launching via `primus-cli` (direct/container/slurm) so `WORLD_SIZE`, `RANK`, `MASTER_ADDR`, etc. are correctly set.
- For file outputs, prefer absolute paths when running under container/Slurm to avoid confusion about working directories.
