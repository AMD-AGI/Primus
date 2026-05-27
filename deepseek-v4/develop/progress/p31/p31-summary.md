# P31 Summary — CSA In-Kernel Top-K Gather/Scatter

> Plan-5 P31. Main report:
> `../../profile/profile-after-p31-ep8-20260509.{md,html}`.

## Objective

Reduce the cr=4 CSA attention bottleneck that became dominant after P30.
P31 targets the wrapper-side `torch.gather(pool, topk_idxs)` materialisation
and the corresponding gather-autograd scatter by moving sparse top-K pool
loads and `dpool` scatter-add into the Triton CSA kernels.

## What Changed

- Added `_v4_csa_attention_pool_fwd_kernel` and
  `_launch_v4_csa_attention_pool_fwd` in
  `v4_attention_kernels/_triton/v4_csa_attention_fwd.py`.
- Added `_v4_csa_attention_pool_bwd_kernel` and
  `_launch_v4_csa_attention_pool_bwd` in
  `v4_attention_kernels/_triton/v4_csa_attention_bwd.py`.
- Added `V4CSAPoolAttentionFn` and `v4_csa_attention_from_pool` in
  `v4_attention_kernels/v4_csa_attention.py`.
- Routed `DeepseekV4Attention._csa_forward` through
  `v4_csa_attention_from_pool` when `use_v4_triton_csa_attention=True`.
  The eager fallback still builds `gathered` and calls the old reference.
- Updated dispatch tests and startup logging to report
  `v4_csa_attention_from_pool (Triton)`.
- Added P31 pool/topk equivalence tests covering invalid `topk_idxs == -1`,
  duplicate top-K slots, sink on/off, fp32/bf16, FWD, BWD, and `dpool`.

## Verification

| Gate | Result |
|---|---|
| P31 pool/topk fast tests | `8 passed, 8 deselected` |
| P31 pool/topk release tests | `8 passed, 8 deselected` |
| P26 CSA release ratchet | `16 passed, 23 deselected` |
| P26/P27 fast CSA + dispatch ratchet | `43 passed, 16 deselected` |
| Final EP8 smoke | 10/10 iterations, no NaN / Inf, `lm_loss[10]=9.259875E+00` |
| P31 trace/report | `profile-after-p31-ep8-20260509.{md,html}` |

## Performance

| Metric | P30b | P31 | Delta |
|---|---:|---:|---:|
| Steady step time | 4943.4 ms | 4317.0 ms | -12.7 % |
| Steady TFLOP/s/GPU | 138.4 | 158.5 | +14.5 % |
| CSA FWD kernel time | ~153 ms | 123.5 ms | -19 % |
| CSA BWD kernel time | 4.04 s | 3.50 s | -13.5 % |
| cr=4 FWD effective TFLOP/s | 6.71 | 8.32 | +24.0 % |
| cr=4 BWD effective TFLOP/s | 0.64 | 0.73 | +14.8 % |

The final smoke line reports iter 10 at `4312.3/4331.7 ms` and
`158.7/158.0 TFLOP/s/GPU`. The profiler steady window reports
`4317.0 ms` and `158.5 TFLOP/s/GPU`, which is the value recorded in
`develop/perf/proxy_ep8.md`.

## Notes

- A pool-path `BLOCK_K=64` experiment compiled and passed the P31 fast and
  release tests, but proxy smoke regressed to ~155 TFLOP/s/GPU, so it was
  reverted to `BLOCK_K=32`.
- In-kernel gather/scatter is a meaningful partial win, but it does not
  meet the original >=25 % CSA BWD budget by itself. The remaining
  bottleneck is `_v4_csa_attention_pool_bwd_kernel`, especially sparse
  bandwidth and atomics into `dpool`.
- `develop/perf/attention_perf.md` and `develop/perf/proxy_ep8.md` were
  updated per rule R2.5.

## Follow-Up Optimization Probe

After the first P31 close-out, CSA BWD was reopened with a stricter target
of <50 ms. Added `bench_csa_attention_ep8.py` to measure the real EP8 CSA
shape on one GPU without launching full training:

| Case | FWD mean | BWD mean |
|---|---:|---:|
| EP8 real shape, `K_topk=512` | 48.44 ms | 1433.02 ms |
| Dense-local split, `K_topk=512` | 48.33 ms | 35.43 ms |
| EP8 real shape, sorted `K_topk=512` | 48.44 ms | 1410.54 ms |
| Local-only, `K_topk=0` | 0.88 ms | 18.01 ms |
| Sparse reduced, `K_topk=128` | 21.62 ms | 749.56 ms |

Several quick tuning probes were negative: per-head `dpool` staging,
`num_warps=8`, sorted top-k pool ids, and per-row sparse `tl.dot`. The
successful redesign splits CSA BWD: local SWA uses the optimized dense
`_v4_attention_bwd_kernel` with CSA's joint `lse/D`, and sparse pool work
uses a new head-block `_v4_csa_attention_pool_sparse_bwd_kernel`.

Profiler attribution: local dense BWD **16.48 ms** + sparse CSA BWD
**17.83 ms**, meeting the <50 ms backward-kernel target on the standalone
EP8-shape benchmark. Logs: `csa_bwd_optimization_log.md` and
`../p30/csa_bwd_followup_log.md`.
