# Base GEMM Benchmark Report

- Date: 2025-11-17 08:00:02
- Cluster: amd-aig-poolside
- Benchmark Duration: 10 sec

## GEMM Configuration
- M: 4096
- N: 4096
- K: 4096
- Transpose A: False
- Transpose B: False
- Dtype: bf16

## GEMM Shape
- A: (4096, 4096)
- B: (4096, 4096)
- C: (4096, 4096)

## Metrics
- `avg_time_ms`: average time per matmul (ms)
- `tflops`: total TFLOPS (1e12 ops/sec)
- `bandwidth_gbps`: estimated memory bandwidth usage (GB/s)
- `arith_intensity`: arithmetic intensity (FLOPs per byte)

| host | world | rank | avg_time_ms | tflop | tflops | bandwidth_gbps | arith_intensity |
|---|---|---|---|---|---|---|---|
| gpu-40 | 8 | 0 | 0.251301 | 0.14 | 546.91 | 400.57 | 1365.33 |
| gpu-40 | 8 | 1 | 0.254165 | 0.14 | 540.75 | 396.05 | 1365.33 |
| gpu-40 | 8 | 2 | 0.248105 | 0.14 | 553.95 | 405.73 | 1365.33 |
| gpu-40 | 8 | 3 | 0.247124 | 0.14 | 556.15 | 407.34 | 1365.33 |
| gpu-40 | 8 | 4 | 0.259761 | 0.14 | 529.10 | 387.52 | 1365.33 |
| gpu-40 | 8 | 5 | 0.249374 | 0.14 | 551.14 | 403.66 | 1365.33 |
| gpu-40 | 8 | 6 | 0.235217 | 0.14 | 584.31 | 427.96 | 1365.33 |
| gpu-40 | 8 | 7 | 0.231001 | 0.14 | 594.97 | 435.77 | 1365.33 |
