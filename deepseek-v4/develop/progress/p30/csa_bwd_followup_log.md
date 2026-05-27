# CSA BWD Follow-Up Log

This document records the CSA backward optimization experiments requested
during P31 follow-up. The detailed live log is also kept in
`../p31/csa_bwd_optimization_log.md`; this copy exists under `p30/` per the
requested location for optimization process notes.

## Target Shape

Standalone benchmark:

```bash
python deepseek-v4/develop/progress/p31/bench_csa_attention_ep8.py --warmup 1 --iters 5
```

Shape: `B=1, H=64, S=4096, D=512, P=1024, K_topk=512, swa_window=128,
bf16, sink=on`.

## Results

| Experiment | BWD-only mean | Decision |
|---|---:|---|
| Original pool BWD | ~1433 ms | Baseline bottleneck |
| Per-head `dpool` staging | no improvement in EP8 trace | Reverted |
| `num_warps=8` | ~1433 ms | Reverted |
| Sorted top-k ids | ~1411 ms | Benchmark-only; not worth sort overhead |
| Skip `dpool` write-back | ~554 ms | Diagnostic only; not correct for training |
| Per-row sparse `tl.dot` | ~663 ms with `dpool` skipped | Reverted |
| Sparse head-block split, old local branch | ~571 ms | Sparse kernel was fast, local path was not |
| Dense local branch + sparse head-block split | **35.43 ms** | Keep as default |

## Final Design

The CSA backward path is split:

- Local SWA branch reuses the optimized dense `_v4_attention_bwd_kernel`, but
  receives CSA's joint `lse` and `D=(dout*out).sum(-1)`.
- Sparse pool branch runs `_v4_csa_attention_pool_sparse_bwd_kernel`, which
  groups all heads for a query row and uses `tl.dot` for sparse `qk`, `dp`,
  `dq`, and `dpool`.

Profiler attribution at the EP8 CSA shape:

| Kernel | CUDA time |
|---|---:|
| `_v4_attention_bwd_kernel` | 16.48 ms |
| `_v4_csa_attention_pool_sparse_bwd_kernel` | 17.83 ms |

P31 pool/topk fast and release unit tests both passed with the split path.

## rocprof-compute Note

`rocprof-compute` is present in the container, but fails at startup because its
Python dependencies are missing (`plotly`, `dash`, `textual`, and related
packages from the ROCm requirements file). For this one-hour pass, profiling
used `torch.profiler` kernel attribution instead.
