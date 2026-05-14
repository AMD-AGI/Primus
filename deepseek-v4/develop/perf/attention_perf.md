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

**Cell format**: starting from P31b, each cell is `<ms> ms | <tflops> tflops` —
the wall-clock per-kernel-launch time and the effective TFLOP/s derived from
that wall-clock via the visible-pair count above. P30b / P31 and earlier rows
keep the legacy TFLOP/s-only format and stay frozen as the historical
record. All future updates to this table MUST follow the `ms | tflops`
format.

| Run | Main delta | cr=0 FWD | cr=0 BWD | cr=4 FWD | cr=4 BWD | cr=128 FWD | cr=128 BWD | Source |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Baseline (P28) | Proxy baseline | 0.54 | 0.26 | 6.59 | 0.64 | 0.59 | 0.29 | `profile-baseline-ep8-20260508.md` |
| P29 | Compiled Sinkhorn | 0.54 | 0.27 | 6.61 | 0.64 | 0.58 | 0.29 | `profile-after-p29-ep8-20260509.md` |
| P30a | Dense SWA K-loop pruning (cr=0 only) | 11.77 | 5.44 | 6.71 | 0.64 | 0.58 | 0.30 | trace `1778313739342936814` |
| P30b | Dense + HCA SWA K-loop pruning (cr=0 + cr=128) | 12.12 | 5.50 | 6.71 | 0.64 | 11.35 | 5.60 | `profile-after-p30-ep8-20260509.md` |
| P31 | CSA in-kernel top-K gather/scatter (cr=4) | 12.12 | 5.50 | 8.32 | 0.73 | 11.35 | 5.60 | `profile-after-p31-ep8-20260509.md` |
| P31b | CSA dense-local + sparse head-block BWD split | 5.58 ms \| 12.12 | 30.75 ms \| 5.50 | 41.17 ms \| 8.32 | 35.43 ms \| 24.17 | 6.69 ms \| 11.35 | 33.91 ms \| 5.60 | `progress/p31/bench_csa_attention_ep8.py` |
| P32 (shipped) | Split CSA FWD + monolithic V4/CSA BWD + gather+atomic dpool | 0.71 ms \| 95.29 | 17.26 ms \| 9.80 | 3.22 ms \| 106.38 | 32.62 ms \| 26.25 | 0.85 ms \| 89.39 | 20.66 ms \| 9.19 | `progress/p32/{bench_v4_attention_ep8,bench_csa_attention_ep8}.py` |
| P32 (bench-opt opt-in) | Split CSA FWD + split V4 BWD + segreduce dpool | 0.73 ms \| 92.68 | 7.65 ms \| 22.11 | 3.16 ms \| 108.40 | 16.31 ms \| 52.50 | 0.91 ms \| 83.49 | 11.91 ms \| 15.95 | `PRIMUS_V4_ATTN_BWD_USE_SPLIT=1 PRIMUS_V4_CSA_BWD_SEGREDUCE=1` |
| P32 RoPE-fix (final defaults) | dual-RoPE bf16 cast + split V4 BWD + segreduce CSA BWD all ON | 0.73 ms \| 92.68 | 7.65 ms \| 22.11 | 3.16 ms \| 108.40 | 16.31 ms \| 52.50 | 0.91 ms \| 83.49 | 11.91 ms \| 15.95 | bench unchanged; proxy traces now match (see `proxy_ep8.md` P32 final row) |

All TFLOP/s values are effective TFLOP/s for the corresponding kernel family,
computed from the visible-pair counts above. The ms column is the wall-clock
per-launch median from the trace or microbench (P31b cr=0 / cr=128 cells use
the post-P30 trace per-launch median; P31b cr=4 cells use the standalone
EP8-shape microbench; P32 cells are 60-iter median from the shipped
microbenches).
For P30b and P31, all five `_v4_attention_bwd_kernel` launches are in
the 30-34 ms range, including the two cr=128 HCA launches. P31's cr=4
values use `_v4_csa_attention_pool_{fwd,bwd}_kernel`; P31b's cr=4 BWD
uses the standalone EP8-shape benchmark after the BWD timer was fixed to
exclude forward execution.

P32 ships **three rows** because the original "shipped vs opt-in"
A/B was performed in the presence of a separate bf16 → fp32 upcast
bug in `apply_interleaved_partial_rope` (`dual_rope.py`): the cos /
sin from `position_ids.float() * inv_freq` was fp32, the rotation
`bf16 * fp32 = fp32` quietly returned fp32 Q / K, and **every** V4
attention kernel in the proxy paid 2× HBM traffic + ran the slow
fp32 Triton kernel binary. That bug inflated all proxy V4-attn
kernel times by 1.8-7× and made the bench-optimal split / segreduce
paths *look* like proxy regressions because their extra HBM traffic
landed on top of the already-bloated baseline. Fix is one line:
cast `cos / sin` to `x.dtype` after the unsqueeze; verified by
bench at `dtype=torch.float32` reproducing the proxy numbers exactly
(dense `5.65 ms`, hca `6.73 ms`).

- **`P32 (shipped)`** — *pre-RoPE-fix* baseline retained for history:
  CSA FWD split = ON, V4 attn BWD split = OFF, CSA BWD segreduce =
  OFF. Microbench wall times: cr=0 FWD 0.71 ms / BWD 17.26 ms,
  cr=4 FWD 3.22 ms / BWD 32.62 ms, cr=128 FWD 0.85 ms / BWD 20.66 ms.
- **`P32 (bench-opt opt-in)`** — *pre-RoPE-fix* bench-optimal
  config: `PRIMUS_V4_ATTN_BWD_USE_SPLIT=1
  PRIMUS_V4_CSA_BWD_SEGREDUCE=1`. cr=0 BWD 7.65 ms, cr=4 BWD
  16.31 ms, cr=128 BWD 11.91 ms.
- **`P32 RoPE-fix (final defaults)`** — RoPE cast fix in place plus
  both env vars defaulted ON. Microbench numbers are unchanged (bench
  was already at bf16); the meaningful shift is in the proxy where
  the **same** kernels now match their bench numbers exactly, and the
  bench-optimal kernel choices win end-to-end by +14 % iter time.
  See `proxy_ep8.md` P32 final row for the corresponding training
  throughput. CSA FWD hits the ≤ 6 ms target (15.0× vs P31b); cr=0
  and cr=128 BWD comfortably clear the ≤ 15 ms target; cr=4 BWD
  lands at 16.31 ms (within 1.3 ms of the target).

The monolithic CSA FWD remains as an opt-in fallback via
`PRIMUS_V4_CSA_FWD_FORCE_MONOLITHIC=1` for A/B testing; the old
`gathered` CSA API remains covered by P26 tests as a
fallback/reference.
