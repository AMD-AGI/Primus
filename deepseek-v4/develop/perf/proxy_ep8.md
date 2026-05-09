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


Notes:

- Iter time and TFLOP/s/GPU use the steady / post-warmup numbers
recorded by the proxy smoke or trace report.
- P30b is the current Plan-5 P30 close-out number; P30a is kept because
it explains why the first dense-only pruning left two cr=128 BWD
kernels as 600 ms+ outliers.
