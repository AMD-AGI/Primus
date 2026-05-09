# Plan-5 P30 — V4 Triton dense/HCA attention SWA pruning

> Phase summary written at P30 close. Deeper perf details live in the
> post-P30 report at `../../profile/profile-after-p30-ep8-20260509.{md,html}`.

---

## 1. Objective

P30 started from the post-P29 trace: `_v4_attention_bwd_kernel` was
**3.18 s / 36.8 %** of step time and `_v4_attention_fwd_kernel` was
**641 ms / 7.4 %**. The phase goal was to reduce dense / HCA
`v4_attention` kernel time without changing the V4 attention math or
the existing dtype contract.

The chosen optimisation was **SWA K-loop range pruning** for dense
`compress_ratio == 0` and HCA `compress_ratio == 128` layers. It was
lower risk than autotune or HCA LSE-merge because it only skips key
tiles that are provably outside the sliding window for every query row
in a Triton program.

Trace review after the dense-only cut showed exactly the issue the user
called out: five `_v4_attention_bwd_kernel` launches existed, three were
~30 ms but the two HCA (`compress_ratio == 128`) launches were still
600 ms+. The final P30 patch adds HCA split-mask mode so those two
launches are pruned too.

---

## 2. What changed

| component | path | change |
|---|---|---|
| FWD kernel | `primus/backends/megatron/core/transformer/v4_attention_kernels/_triton/v4_attention_fwd.py` | Adds `n_loop_start` for `SWA_WINDOW > 0`. In HCA split-mask mode, runs a pruned local-SWA loop plus a short pool-suffix loop under one online softmax. |
| BWD kernel | `primus/backends/megatron/core/transformer/v4_attention_kernels/_triton/v4_attention_bwd.py` | Applies the same visible K-tile range to the recompute loop, including HCA local prefix + pool suffix split. |
| API | `primus/backends/megatron/core/transformer/v4_attention_kernels/v4_attention.py` | Adds optional `hca_local_seqlen: int = 0` to activate HCA split-mask mode. Default remains the existing generic behavior. |
| wrapper dispatch | `primus/backends/megatron/core/transformer/deepseek_v4_attention.py` | Dense V4 Triton attention calls `v4_attention(..., additive_mask=None, swa_window=attn_sliding_window)`. HCA V4 Triton attention passes pool-only `extra_mask` plus `hca_local_seqlen=S`. Eager paths keep original full masks. |
| tests | `tests/unit_tests/megatron/transformer/deepseek_v4/test_v4_p25_v4_attention_{fwd,bwd}.py` | HCA candidate now exercises split-mask mode while the eager reference still uses the full `cat([local_mask, pool_mask])` additive mask. |

HCA still computes one joint softmax. The optimisation changes only how
the K-loop reaches the same visible keys: local SWA prefix is computed
in-kernel, and the compressed-pool suffix uses the pool-only mask.

---

## 3. Gates

| gate | status | numbers |
|---|---|---|
| **G34 fast tier** — dense/HCA FWD+BWD parity | **GREEN** | `pytest -q tests/unit_tests/megatron/transformer/deepseek_v4/test_v4_p25_v4_attention_fwd.py tests/unit_tests/megatron/transformer/deepseek_v4/test_v4_p25_v4_attention_bwd.py -m "not slow"` → **67 passed in 14.53s**. |
| **G34 release tier** — production-shape parity | **GREEN** | Same files with `-m slow` → **64 passed in 31.50s**. Covers `head_dim=512`, sink, dense SWA, HCA split-mask, FWD and BWD tolerances. |
| **G34a smoke** — EP=8 10-iter proxy | **GREEN** | `progress/p30/run_smoke_v4_attention_swa_prune_ep8.sh`; no NaN / Inf / banned warnings; `lm_loss[10] = 9.258615E+00`; steady throughput **~138.4 TFLOP/s/GPU**. |
| **G34a trace + report** | **GREEN** | Trace at `output/amd/tas-mi355x-20260509/p30_profile_swa_prune_pp1_ep8_seq4096/.../*.pt.trace.json`; report at `develop/profile/profile-after-p30-ep8-20260509.{md,html}`. |

---

## 4. Performance delta

| metric | post-P29 | post-P30 | delta |
|---|---:|---:|---:|
| Steady trace step time | 8.63 s | 4.94 s | **−42.8 %** |
| Smoke steady TFLOP/s/GPU | 79.1 | ~138.4 | **+75 %** |
| `_v4_attention_bwd_kernel` | 3.18 s | 160 ms | **−95.0 %** |
| `_v4_attention_fwd_kernel` | 641 ms | 30 ms | **−95.3 %** |
| Dense/HCA `v4_attention` total | 3.82 s | 192 ms | **−95.0 %** |
| `_v4_csa_attention_bwd_kernel` | 4.03 s | 4.04 s | ~0 % |
| GPU active | 99.5 % | 99.0 % | unchanged |
| Multi-stream overlap factor | 1.00× | 1.00× | unchanged |

P30 clears the Y budget by a wide margin. All five
`_v4_attention_bwd_kernel` launches are now in the **30-34 ms** range:
`31.1, 34.1, 33.7, 30.3, 30.9 ms`.

---

## 5. Hand-off to P31

The residual dominant bottleneck is now CSA attention BWD:
`_v4_csa_attention_bwd_kernel` is **4.04 s / 81.9 %** of the post-P30
step. P31 should focus on the CSA plan already seeded in plan-5:
in-kernel `topk_idxs` gather / BWD scatter-add, K-tile prefetching, and
`K_topk=512` autotune.

Deferred P30 follow-ups:

| follow-up | reason deferred |
|---|---|
| Dense/HCA per-shape autotune | SWA pruning already cleared the P30 Y budget without autotune warmup risk. |
| Persistent FWD kernel | FWD is no longer the dominant dense/HCA cost after pruning. |
| HCA LSE-merge | Still valuable for HCA, but it needs a new correctness surface; P30 kept the shipped change narrow. |

---

## 6. Artefacts shipped under `progress/p30/`

| file | purpose |
|---|---|
| `.gitignore` | excludes logs, traces, tfevents, tgz artefacts, and debug output |
| `run_smoke_v4_attention_swa_prune_ep8.sh` | G34a 10-iter EP=8 smoke |
| `run_baseline_trace_ep8_p30.sh` | G34a torch.profiler trace capture |
| `p30-summary.md` | this file |
