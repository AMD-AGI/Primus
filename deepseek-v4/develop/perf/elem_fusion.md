# Plan-6 elementwise / small-op fusion results

This file tracks the per-fusion delta between the Triton kernel and the
original eager implementation for every plan-6 P34..P39 small-op
fusion.  Each row is pinned against a checked-in microbench script under
`deepseek-v4/develop/progress/p<phase>/bench_<op>.py` and a release-tier
unit-test ratchet under `tests/unit_tests/megatron/extensions/`.

## Common bench shape

V4-Flash EP=8 proxy widths unless otherwise noted:

| key | value |
|---|---|
| `E` (experts / rank) | 32 |
| hidden | 4096 |
| moe ffn | 2048 |
| seq | 4096 |
| dtype | `bfloat16` |
| host / container | `mi355-gpu-8` / `dev_primus_wenx_693` |
| bench flags | `--iters 20 --warmup 5 --n-input-copies 4 --l2-flush-mb 512` |

## P34 — `_stack_grouped_linear_weight`

Replaces `torch.stack(weights, dim=0).transpose(1, 2).contiguous()` with
a single Triton kernel that does a fused `[K, N] -> [N, K]` tile-level
transpose with per-expert int64 pointer dispatch.

Bench: `deepseek-v4/develop/progress/p34/bench_stack_grouped_weight.py`
(`--mode {fc1, fc2}`, raw JSON at `progress/p34/bench/{fc1,fc2}.json`).
Default ON via `PRIMUS_STACK_GROUPED_WEIGHT_TRITON=1`.

| shape | path | FWD median (ms) | BWD median (ms) | FWD GB/s | BWD GB/s |
|---|---|---:|---:|---:|---:|
| fc1 (`E=32, K=4096, N=4096`, 1024 MiB / call) | eager  | 2.821 | 2.329 |  761.3 |  922.0 |
| fc1                                            | triton | 0.470 | 0.599 | 4566.2 | 3582.5 |
| **fc1 speedup**                                |        | **6.00x** | **3.89x** | **+500 %** | **+288 %** |
| fc2 (`E=32, K=4096, N=2048`,  512 MiB / call) | eager  | 1.495 | 1.314 |  718.3 |  817.3 |
| fc2                                            | triton | 0.280 | 0.411 | 3837.2 | 2613.8 |
| **fc2 speedup**                                |        | **5.34x** | **3.20x** | **+434 %** | **+220 %** |

EP=8 proxy A/B: **steady-state iter time 580.65 ms (eager) ->
530.85 ms (Triton), -49.8 ms / -8.6 %** at bit-identical `lm_loss`
(see `progress/p34/p34-summary.md` §4.2).

Unit tests (G37): `tests/unit_tests/megatron/extensions/test_stack_grouped_weight_triton.py`.
**18 fast + 2 release-tier slow** all green; plan-4/5 ratchet
(`pytest -m slow tests/unit_tests/megatron/transformer/deepseek_v4/`)
stayed green (**92 passed, 304 deselected**).

## P35 — `apply_interleaved_partial_rope`

Replaces the 9-op eager chain (`slice / reshape / 4 broadcast muls /
stack / reshape / cat`) in
:func:`primus.backends.megatron.core.transformer.dual_rope.apply_interleaved_partial_rope`
with a single Triton kernel that does one contiguous write with the
rotation baked in (nope prefix copied verbatim, rotary suffix rotated
using interleaved `(2k, 2k+1)` pairing).  BWD applies the transpose
rotation.  Default ON via `PRIMUS_ROPE_TRITON=1`.

Bench: `deepseek-v4/develop/progress/p35/bench_rope_triton.py`
(`--mode {q, k}`, raw JSON at `progress/p35/bench/{q,k}.json`).

| shape | path | FWD median (ms) | BWD median (ms) | FWD GB/s | BWD GB/s |
|---|---|---:|---:|---:|---:|
| Q (B=1, S=4096, H=64, head_dim=512, rd=64; 256 MiB / call) | eager  | 0.437 | 0.524 | 1230.6 | 1024.6 |
| Q                                                          | triton | 0.148 | 0.187 | 3637.9 | 2878.1 |
| **Q speedup**                                              |        | **2.96x** | **2.81x** | **+196 %** | **+181 %** |
| K (B=1, S=4096, H=1,  head_dim=64,  rd=64; 0.5 MiB / call) | eager  | 0.064 | 0.140 |   24.6 |   11.2 |
| K                                                          | triton | 0.027 | 0.093 |   57.2 |   17.0 |
| **K speedup**                                              |        | **2.33x** | **1.51x** | **+133 %** | **+52 %** |

EP=8 proxy A/B: **steady-state iter time 531.7 ms (eager) ->
526.7 ms (Triton), -5.0 ms / -0.94 %** at bit-identical `lm_loss`
(see `progress/p35/p35-summary.md` §4.2).  Smaller than P34 because
RoPE was a moderate (not hot) small-op target.

Unit tests (G38): `tests/unit_tests/megatron/transformer/deepseek_v4/test_p35_rope_triton.py`.
**27 fast + 2 release-tier slow** all green; plan-4/5 ratchet stayed
green (**94 passed, 331 deselected**).

## P36 — `sinkhorn_normalize` fusion

Replaces the plan-5 P29 `torch.compile` Sinkhorn-Knopp normalize body
in :func:`primus.backends.megatron.core.transformer.hyper_connection.sinkhorn_normalize`
with a hand-rolled Triton FWD/BWD kernel pair.  The full 1 + 2*(n_iters
- 1) alternating row/col normalize trajectory runs in registers per
row of the leading axis (V4-Flash `K=4`: 16 fp32 / row stays in VGPRs
for all 39 steps); BWD reads a cached FWD-trajectory buffer and walks
the analytic VJP backward step-by-step.  Default ON via
`PRIMUS_SINKHORN_TRITON=1`.  Routing precedence:
`PRIMUS_SINKHORN_TRITON != "0" > use_compiled > eager` (the plan-5 P29
compiled body stays reachable via `use_compiled=True` AND env=0).

Bench: `deepseek-v4/develop/progress/p36/bench_sinkhorn.py`
(`--mode {k4, k4_small, k8}`, raw JSON at
`progress/p36/bench/{k4,k4_small,k8}.json`).

| shape | path | FWD median (ms) | BWD median (ms) | FWD GB/s | BWD GB/s |
|---|---|---:|---:|---:|---:|
| k4 (V4-Flash B=1, S=4096, K=4; 128 KiB x / call)        | eager    | 0.587 | 1.502 |  18.3 |   7.2 |
| k4                                                       | compiled | 0.270 | 0.623 |  39.8 |  17.2 |
| k4                                                       | triton   | 0.043 | 0.101 | 249.4 | 106.0 |
| **k4 speedup vs eager**                                  |          | **13.62x** | **14.81x** | **+1265 %** | **+1372 %** |
| **k4 speedup vs P29 compiled**                           |          | **6.26x**  | **6.15x**  | **+528 %**  | **+515 %**  |
| k8 (forward-compat B=1, S=4096, K=8; 512 KiB x / call)   | eager    | 0.501 | 1.450 |  85.8 |  29.6 |
| k8                                                       | compiled | 0.288 | 0.623 | 149.4 |  69.0 |
| k8                                                       | triton   | 0.072 | 0.146 | 600.6 | 294.7 |
| **k8 speedup vs eager**                                  |          | **7.00x**  | **9.94x**  | **+599 %**  | **+894 %**  |
| **k8 speedup vs P29 compiled**                           |          | **4.02x**  | **4.27x**  | **+302 %**  | **+327 %**  |

EP=8 proxy A/B: **iter-10 instantaneous 526.2 ms (compiled fallback)
-> 515.0 ms (Triton), -11.2 ms / -2.1 %** at bf16-bit-identical
`lm_loss` (9.258826 Triton vs 9.258817 compiled fallback; diff
9e-6, well below the 1e-3 bf16 floor).  Matches microbench-predicted
16 calls × 0.75 ms = **12.0 ms / iter** within profiler noise.  See
`progress/p36/p36-summary.md` §4.2.

Unit tests (G39): `tests/unit_tests/megatron/transformer/deepseek_v4/test_p36_sinkhorn_triton.py`.
**26 fast + 1 release-tier slow** all green; plan-4/5 ratchet stayed
green (**95 passed, 357 deselected** in 73.27 s; the +1 vs P35 is the
G39 release-tier V4-Flash test).

## P37 — HyperConnection `compute_weights` tail fusion

Replaces the 7-9 ATen elementwise ops in
:meth:`primus.backends.megatron.core.transformer.hyper_connection.HyperMixer.compute_weights`
between the `_packed_logits` GEMM and the
`sinkhorn_normalize` call (3 slices + 3 fused-multiply-adds + 2
sigmoid + 1 softmax + 2 eps adds) with a single Triton FWD kernel +
single Triton BWD kernel.  Saves `(sigmoid(pre_logit),
sigmoid(post_logit), softmax(comb_logit))` as fp32 state for
backward; BWD walks the analytic VJP per-element and uses host-side
`torch.sum` to reduce `d_base` / `d_scale` partials (avoids
cross-block atomic adds).  Default ON via `PRIMUS_HC_TRITON=1`.
The matmul inside `_packed_logits` and the `collapse` / `expand`
matmul-adjacent glue stay eager.

Bench: `deepseek-v4/develop/progress/p37/bench_hc_glue.py`
(`--mode {v4, small}`, raw JSON at `progress/p37/bench/{v4,small}.json`).

| shape | path | FWD median (ms) | BWD median (ms) | FWD GB/s | BWD GB/s |
|---|---|---:|---:|---:|---:|
| v4 (V4-Flash B=1, S=4096, K=4; ~0.94 MiB FWD / call) | eager  | 0.102 | 0.405 |   9.6 |   3.4 |
| v4                                                    | triton | 0.044 | 0.276 |  22.5 |   5.0 |
| **v4 speedup**                                        |        | **2.34x** | **1.47x** | **+135 %** | **+47 %** |
| small (B=2, S=64, K=4)                                | eager  | 0.097 | 0.405 |   0.3 |   0.1 |
| small                                                 | triton | 0.041 | 0.268 |   0.7 |   0.2 |
| **small speedup**                                     |        | **2.36x** | **1.51x** | **+136 %** | **+51 %** |

EP=8 proxy A/B (iters 4-10 mean, excl. profile-end iter 7):
**519.4 ms (eager tail) -> 516.1 ms (Triton), -3.3 ms / -0.64 %**
at bf16-bit-identical `lm_loss` (9.258826 both A and B).  Matches
microbench prediction of ~3 ms / iter (16 calls × 0.19 ms FWD+BWD
delta).  Iter-10-instantaneous: 514.9 -> 512.1 ms.  See
`progress/p37/p37-summary.md` §4.2.

Unit tests (G40): `tests/unit_tests/megatron/transformer/deepseek_v4/test_p37_hc_glue_triton.py`.
**21 fast + 1 release-tier slow** all green.

## P38 — Indexer small-op fusion

Pending.

## P39 — V4 hash router post-logits fusion

Pending.
