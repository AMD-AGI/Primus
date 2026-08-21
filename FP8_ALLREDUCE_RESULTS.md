# FLUX four-node FP8 gradient AllReduce

## Goal and method

This experiment tests whether compressing only the cross-node HSDP gradient
AllReduce can improve four-node MLPerf time-to-quality without changing the
intra-node BF16 ReduceScatter. The implementation is opt-in:

```bash
FSDP2_HSDP_FP8_ALL_REDUCE=e4m3
FSDP2_HSDP_FP8_BLOCK_SIZE=1024  # 0 selects one scale per tensor
```

Large BF16 HSDP reductions use a scale shared across the four replica ranks,
E4M3 SUM, and BF16 dequantization. Messages below one million elements remain
BF16. Block mode uses shared power-of-two scales and pads only the final partial
block. The default remains BF16 because no candidate improved time-to-quality.

## Results

All training runs used 4 nodes / 32 MI355X GPUs, HSDP 4x8, MBS 32, GBS 1024,
seed 10007, LR 0.00025, 800 warmup steps, AITER attention, tensorwise FP8 GEMMs,
and no periodic checkpointing.

### Full MLPerf E2E

| Gradient AllReduce | First passing step | Validation loss | TTQ | vs BF16 TTQ |
|---|---:|---:|---:|---:|
| BF16 | 8192 | 0.585865 | 5322.750 s | baseline |
| tensor-scale E4M3 | 8448 | 0.585844 | 5351.290 s | 0.995x |
| block-1024 E4M3 | 8448 | 0.585452 | 5368.598 s | 0.991x |

Tensor-scale E4M3 reduced steady step time from 0.6303 s to 0.6134 s
(1.028x), but required one additional 256-step validation interval. Block-1024
also required that interval and was slightly slower than tensor scaling, so
neither candidate improved E2E TTQ.

### Numerical and short-run gates

A four-rank 64 MiB synthetic benchmark measured:

| Mode | Collective path | Relative L2 | Cosine | Max absolute error |
|---|---:|---:|---:|---:|
| tensor-scale E4M3 | 3.400 ms | 0.050181 | 0.99874210 | 18.0 |
| block-1024 E4M3 | 3.338 ms | 0.050017 | 0.99874890 | 8.0 |
| block-256 E4M3 | 3.373 ms | 0.050017 | 0.99874896 | 8.0 |

The BF16 collective took 5.460 ms. Block scaling reduced worst-case error while
retaining the communication microbenchmark gain.

The 900-step training gates showed that this synthetic improvement did not
translate monotonically to validation quality:

| Mode | Step-768 validation | Mean summary step time | Throughput/GPU |
|---|---:|---:|---:|
| BF16 reference | 0.656036 | 0.6490 s (full-run summary) | 49.9024 |
| tensor-scale E4M3 | 0.655694 | 0.6319 s | 51.2049 |
| block-1024 E4M3 | 0.655819 | 0.6343 s | 51.1058 |
| block-256 E4M3 | 0.656773 | 0.6349 s | 51.0562 |

Block-256 missed the proposed numerical gate and was not promoted to a full
run. A second matched 900-step screen with the packaged v0.4 image measured
0.6405 s for BF16 AllReduce and 0.6321 s for tensor-scale E4M3 (1.013x), with
step-768 validation 0.655773 and 0.654936 respectively.

## Decision

The 1.1x incremental E2E target is not reachable from cross-node gradient
AllReduce compression in this recipe. The measured steady-state gain is only
1.3-2.8%, which is already below 1.1x before accounting for validation cadence.
Block scaling improves maximum synthetic error but does not recover the BF16
convergence step and adds scale-processing cost. Selective BF16 fallback and
error feedback would improve accuracy by reducing or correcting compression,
but cannot raise the performance ceiling. Delayed scaling can remove one small
scale collective, but likewise cannot close the roughly 8 percentage-point gap.

Keep the implementation opt-in for future larger-replica experiments. Do not
enable it in the qualified four-node recipe. Reaching a further 1.1x requires a
compute-path or batch/schedule change rather than this communication-only path.

## Evidence

- Tensor E4M3 full run: `/shared_nfs/zirui/runs/flux_4n_fp8_gradient_ar_e2e_20260820T1544Z`
- Block-1024 full run: `/shared_nfs/zirui/runs/flux_4n_fp8_block1024_e2e_20260821`
- Block-1024 gate: `/shared_nfs/zirui/runs/flux_4n_fp8_block1024_900step_20260821`
- Block-256 gate: `/shared_nfs/zirui/runs/flux_4n_fp8_block256_900step_20260821`
- Packaged-image matched gates: `/shared_nfs/zirui/runs/flux_4n_v04_{bf16,fp8}_ar_900step_20260821`
- Synthetic benchmark: `/shared_nfs/zirui/runs/fp8_block_allreduce_bench_20260821`
