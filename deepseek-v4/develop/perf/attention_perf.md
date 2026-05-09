# Attention Kernel Performance

This table tracks V4 attention-kernel throughput for the Plan-5 EP8
proxy. Update it after attention-related optimisations once the operator
unit tests pass.

## Test Shape And Counting

| key | value |
|---|---|
| Host / container | `mi355-gpu-14` / `dev_primus_wenx_693` |
| Proxy | V4-Flash production widths, 8-layer slice |
| Parallelism | TP=1, PP=1, EP=8 |
| Batch / sequence | MBS=1, GBS=8, `S=4096` |
| Heads / head dim | `H=64`, `head_dim=512`, MQA local KV |
| Sliding window | `attn_sliding_window=128` |
| Compress ratios | `[0,0,4,128,4,128,4,0]` |
| Layer counts per iter | cr=0: 3, cr=4: 3, cr=128: 2 |
| cr=4 top-K | `index_topk=512` |
| cr=128 pool size | `P = S / 128 = 32` |

TFLOP/s is computed from useful attention matmul FLOPs, not from the
old unpruned masked-tile work:

- FWD: `QK + PV = 4 * B * H * head_dim * visible_pairs`.
- BWD: recompute `QK` + `dP` + `dQ` + `dK` + `dV`
  `= 10 * B * H * head_dim * visible_pairs`.
- cr=0 visible pairs: local SWA pairs = `516,160`.
- cr=128 visible pairs: local SWA pairs + pool-visible pairs =
  `516,160 + 63,520 = 579,680`.
- cr=4 visible pairs: local SWA pairs + sparse top-K pairs =
  `516,160 + 4096 * 512 = 2,613,312`.

## Results

| Run | Main delta | cr=0 FWD | cr=0 BWD | cr=4 FWD | cr=4 BWD | cr=128 FWD | cr=128 BWD | Source |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Baseline (P28) | Proxy baseline | 0.54 | 0.26 | 6.59 | 0.64 | 0.59 | 0.29 | `profile-baseline-ep8-20260508.md` |
| P29 | Compiled Sinkhorn | 0.54 | 0.27 | 6.61 | 0.64 | 0.58 | 0.29 | `profile-after-p29-ep8-20260509.md` |
| P30a | Dense SWA K-loop pruning (cr=0 only) | 11.77 | 5.44 | 6.71 | 0.64 | 0.58 | 0.30 | trace `1778313739342936814` |
| P30b | Dense + HCA SWA K-loop pruning (cr=0 + cr=128) | 12.12 | 5.50 | 6.71 | 0.64 | 11.35 | 5.60 | `profile-after-p30-ep8-20260509.md` |
| P31 | CSA in-kernel top-K gather/scatter (cr=4) | 12.12 | 5.50 | 8.32 | 0.73 | 11.35 | 5.60 | `profile-after-p31-ep8-20260509.md` |
| P31b | CSA dense-local + sparse head-block BWD split | 12.12 | 5.50 | 8.32 | 24.17 | 11.35 | 5.60 | `progress/p31/bench_csa_attention_ep8.py` |

All values are effective TFLOP/s for the corresponding kernel family.
For P30b and P31, all five `_v4_attention_bwd_kernel` launches are in
the 30-34 ms range, including the two cr=128 HCA launches. P31's cr=4
values use `_v4_csa_attention_pool_{fwd,bwd}_kernel`; P31b's cr=4 BWD
uses the standalone EP8-shape benchmark after the BWD timer was fixed to
exclude forward execution. The old `gathered` CSA API remains covered by
P26 tests as a fallback/reference.
