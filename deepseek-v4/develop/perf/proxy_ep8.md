# Proxy EP8 End-To-End Performance

This table tracks end-to-end training performance for the Plan-5
V4-Flash EP8 proxy.

## Test Shape


| key              | value                                                                                                                                                  |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Host / container | `mi355-gpu-14` / `dev_primus_wenx_693`                                                                                                                 |
| Proxy            | V4-Flash production widths, 8-layer slice                                                                                                              |
| Parallelism      | TP=1, PP=1, EP=8                                                                                                                                       |
| Batch / sequence | MBS=1, GBS=8, `S=4096`                                                                                                                                 |
| Layers           | 8                                                                                                                                                      |
| Compress ratios  | `[0,0,4,128,4,128,4,0]`                                                                                                                                |
| Perf knobs       | `use_v4_triton_attention=True`, `use_v4_triton_csa_attention=True`, `use_turbo_deepep=True`, `use_turbo_grouped_mlp=True`, `use_turbo_attention=False` |


## Results


| Run            | Main delta                                     | Iter time (ms) | TFLOP/s/GPU | vs baseline | Source                                         |
| -------------- | ---------------------------------------------- | -------------- | ----------- | ----------- | ---------------------------------------------- |
| Baseline (P28) | Proxy baseline                                 | 8837.4         | 77.52       | 1.00x       | `profile-baseline-ep8-20260508.md`             |
| P29            | Compiled Sinkhorn                              | ~8630          | 79.1        | 1.02x       | `profile-after-p29-ep8-20260509.md`, P29 smoke |
| P30a           | Dense SWA K-loop pruning (cr=0 only)           | 6437.2         | 106.3       | 1.37x       | trace `1778313739342936814`, P30 smoke         |
| P30b           | Dense + HCA SWA K-loop pruning (cr=0 + cr=128) | 4943.4         | 138.4       | 1.79x       | `profile-after-p30-ep8-20260509.md`, P30 smoke |
| P31            | CSA in-kernel top-K gather/scatter (cr=4)      | 4317.0         | 158.5       | 2.04x       | `profile-after-p31-ep8-20260509.md`, P31 smoke |
| P31b           | CSA dense-local + sparse head-block BWD split  | 964.8          | 709.3       | 9.15x       | trace `1778324637328675032`, P31b smoke        |
| P32            | CSA FWD split (local + sparse + LSE merge)     | 890.5          | 768.4       | 9.92x       | trace `1778476971738245137`, `profile-after-p32-ep8-20260511.md` |


Notes:

- Iter time and TFLOP/s/GPU use the steady / post-warmup numbers
recorded by the proxy smoke or trace report.
- P30b is the current Plan-5 P30 close-out number; P30a is kept because
it explains why the first dense-only pruning left two cr=128 BWD
kernels as 600 ms+ outliers.
- P31 uses the profiler steady window for the table headline. The final
  10-iter smoke after the `BLOCK_K=64` experiment was reverted reports
  `4312.3/4331.7 ms` and `158.7/158.0 TFLOP/s/GPU` on iter 10.
- P31b uses the post-profiler steady iter 10 line from the EP8 proxy run:
  `964.8/1114.5 ms` and `709.3/648.0 TFLOP/s/GPU`. The profiler window
  includes overhead on iter 7, so the table headline uses the clean
  post-profiler instantaneous value. Trace kernels show
  `_v4_csa_attention_pool_sparse_bwd_kernel` at **80.8 ms / 3 launches**
  and `_v4_csa_attention_pool_fwd_kernel` at **123.1 ms / 3 launches**.
- P32 ships the CSA FWD split (local SWA + sparse pool + LSE merge) as
  the new default; V4 attention BWD stays monolithic and CSA BWD stays
  on the gather + atomic dpool path. The bench-only split BWD and
  segmented-reduction CSA BWD paths remain available via
  `PRIMUS_V4_ATTN_BWD_USE_SPLIT=1` / `PRIMUS_V4_CSA_BWD_SEGREDUCE=1`
  (they win the standalone microbench but lose ~190 ms and ~40 ms /
  iter respectively in EP8 because of doubled HBM traffic competing
  with MoE work). P32 iter 10 is the post-profiler steady value:
  `890.5/1037.1 ms` and `768.4/703.8 TFLOP/s/GPU`. Trace kernels show
  `_v4_csa_attention_pool_sparse_fwd_kernel` at **33.6 ms / 3
  launches** (plus ~17 ms of `_v4_attention_fwd_kernel` for the new
  CSA local FWD; previously `_v4_csa_attention_pool_fwd_kernel` was
  **123.1 ms / 3 launches**) and `_v4_csa_attention_pool_sparse_bwd_kernel`
  at **72.5 ms / 3 launches** (vs P31b 80.8 ms).
