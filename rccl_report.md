# RCCL Benchmark Report

- Date: 2025-11-17 09:24:31
- Cluster: amd-aig-poolside
- World Size: 8
- Ops: allreduce
- Message Sizes: 0.0KB, 0.0KB, 0.0KB, 0.0KB, 0.0KB, 0.0KB, 0.1KB, 0.1KB, 0.2KB, 0.5KB, 0.0MB, 0.0MB, 0.0MB, 0.0MB, 0.0MB, 0.0MB, 0.1MB, 0.1MB
- Dtype: bf16
- Warmup: 20
- Iterations: 100
- Repeat: 1
- Correctness Check: off
- Hosts (8): gpu-40, gpu-40, gpu-40, gpu-40, gpu-40, gpu-40 ...

## Command
`primus/cli/main.py benchmark rccl`

- Git Commit: dc9218e

## Key Arguments
- aggregate_repeat: False
- append: False
- check: False
- cluster: amd-aig-poolside
- command: benchmark
- dtype: bf16
- iters: 100
- max_bytes: 128M
- min_bytes: 1K
- num_sizes: 12
- op: ['allreduce']
- output_file: ./rccl_report.md
- per_iter_trace: False
- per_rank: False
- per_rank_file:
- repeat: 1
- scale: log2
- sizes: None
- suite: rccl
- trace_file:
- trace_limit: 0
- trace_ops:
- trace_sizes:
- warmup: 20

## Environment
- HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
- NCCL_DEBUG=
- RCCL_MSCCL_ENABLE=0
- RCCL_MSCCLPP_ENABLE=0
- RCCL_MSCCLPP_THRESHOLD=1073741824

## Metrics
- `p50_ms` / `p95_ms`: 50/95th percentile of critical-path latency (ms)
- `min_ms` / `max_ms`: min/max latency observed on the critical path (ms)
- `eff_gbps`: per-rank effective bandwidth in GB/s, normalized by collective algorithm factor
- `slowest_rank`: rank with the highest p95 latency (format `rX@host`)
- `rank_p95_spread_ms`: max p95 minus median p95 across ranks, indicating imbalance

| host | world | suite | op | bytes | dtype | repeat | p50_ms | p95_ms | min_ms | max_ms | eff_gbps | slowest_rank | rank_p95_spread_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gpu-40 | 8 | rccl | allreduce | 1024 | bfloat16 | 1 | 0.298 | 0.336 | 0.289 | 0.483 | 0.01 | r6@gpu-40 | 0.004 |
| gpu-40 | 8 | rccl | allreduce | 2048 | bfloat16 | 1 | 0.296 | 0.309 | 0.290 | 1.055 | 0.01 | r3@gpu-40 | 0.002 |
| gpu-40 | 8 | rccl | allreduce | 4096 | bfloat16 | 1 | 0.295 | 0.303 | 0.291 | 0.951 | 0.02 | r0@gpu-40 | 0.004 |
| gpu-40 | 8 | rccl | allreduce | 8192 | bfloat16 | 1 | 0.295 | 1.210 | 0.291 | 1.897 | 0.05 | r7@gpu-40 | 0.002 |
| gpu-40 | 8 | rccl | allreduce | 16384 | bfloat16 | 1 | 0.300 | 0.310 | 0.296 | 1.171 | 0.10 | r5@gpu-40 | 0.002 |
| gpu-40 | 8 | rccl | allreduce | 32768 | bfloat16 | 1 | 0.310 | 0.790 | 0.306 | 1.339 | 0.18 | r1@gpu-40 | 0.010 |
| gpu-40 | 8 | rccl | allreduce | 65536 | bfloat16 | 1 | 0.311 | 0.332 | 0.307 | 1.733 | 0.37 | r2@gpu-40 | 0.010 |
| gpu-40 | 8 | rccl | allreduce | 131072 | bfloat16 | 1 | 0.311 | 0.363 | 0.307 | 0.999 | 0.74 | r0@gpu-40 | 0.003 |
| gpu-40 | 8 | rccl | allreduce | 262144 | bfloat16 | 1 | 0.312 | 1.026 | 0.309 | 1.830 | 1.47 | r6@gpu-40 | 0.105 |
| gpu-40 | 8 | rccl | allreduce | 524288 | bfloat16 | 1 | 0.297 | 0.303 | 0.294 | 1.080 | 3.09 | r2@gpu-40 | 0.000 |
| gpu-40 | 8 | rccl | allreduce | 1048576 | bfloat16 | 1 | 0.304 | 0.729 | 0.301 | 1.895 | 6.04 | r1@gpu-40 | 0.011 |
| gpu-40 | 8 | rccl | allreduce | 2097152 | bfloat16 | 1 | 0.311 | 0.315 | 0.308 | 0.343 | 11.78 | r0@gpu-40 | 0.000 |
| gpu-40 | 8 | rccl | allreduce | 4194304 | bfloat16 | 1 | 0.325 | 0.353 | 0.320 | 0.958 | 22.56 | r3@gpu-40 | 0.002 |
| gpu-40 | 8 | rccl | allreduce | 8388608 | bfloat16 | 1 | 0.351 | 0.676 | 0.347 | 1.777 | 41.82 | r5@gpu-40 | 0.013 |
| gpu-40 | 8 | rccl | allreduce | 16777216 | bfloat16 | 1 | 0.403 | 0.411 | 0.396 | 0.559 | 72.89 | r4@gpu-40 | 0.001 |
| gpu-40 | 8 | rccl | allreduce | 33554432 | bfloat16 | 1 | 0.507 | 0.636 | 0.503 | 1.442 | 115.71 | r2@gpu-40 | 0.016 |
| gpu-40 | 8 | rccl | allreduce | 67108864 | bfloat16 | 1 | 0.707 | 0.773 | 0.701 | 1.210 | 166.12 | r6@gpu-40 | 0.010 |
| gpu-40 | 8 | rccl | allreduce | 134217728 | bfloat16 | 1 | 1.104 | 1.155 | 1.091 | 1.199 | 212.82 | r1@gpu-40 | 0.001 |
