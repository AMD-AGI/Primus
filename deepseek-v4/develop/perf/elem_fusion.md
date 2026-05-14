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

Pending.

## P37 — HyperConnection elementwise fusion

Pending.

## P38 — Indexer small-op fusion

Pending.

## P39 — V4 hash router post-logits fusion

Pending.
