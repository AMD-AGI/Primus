# Plan-5 P29 — Sinkhorn fp32 reduce kill (RESCOPED)

> Phase summary written at P29 close. One-page record of objective,
> what changed, what's green, what was de-scoped, and how the next
> phase consumes the result. The deeper write-ups live in
> `refinement.md` (forensic root cause + fix decision) and the
> post-P29 trace report at
> `../../profile/profile-after-p29-ep8-20260509.{md,html}`.

---

## 1. Objective

P29 was rescoped at P28 close (`afd7ea59`) from the original
"small-op kernel-launch fusion" mandate to **root-cause and eliminate
the dominant `aten::sum` fp32 reduce kernel** that the P28 baseline
trace pinned at 7.61 s / 87.3 % of step time.

The original seeded targets (`v4_fused_q_proj`, `v4_fused_kv_proj`,
`v4_fused_o_proj`, `v4_fused_compressor` + `v4_fused_indexer`,
`v4_fused_moe_router`) were **de-scoped** because the P28 trace
showed the GPU at 99.7 % active and the CPU-bound floor at 0.3 %
(≪ 10 % rule). Kernel-launch tail is not the bottleneck at V4-Flash
production widths.

---

## 2. Forensic root cause (P28 trace dive)

Three iterative passes (`_forensics{,2,3}.py`) over the P28 chrome-
trace JSON produced the attribution table below:

| metric | value |
| --- | --- |
| matching `reduce_kernel<512, 1, ...>` launches in steady iter | **624 / 717 = 87 % by count** |
| Σ kernel duration | **7607.9 ms / 99.95 %** of all matching reduce-kernel time |
| direct launcher cpu_op | `aten::sum` (100 %) |
| deepest Python source line | `primus/backends/megatron/core/transformer/hyper_connection.py:47 sinkhorn_normalize` (304 / 316 = 96 % of matched) |
| input shape | `(1, 4096, 4, 4) → keepdim=True dim=-1` fp32 |
| algebra | 8 layers × 39 reductions / call × 2 (FWD + AOT-autograd BWD) = 624 launches / iter ✓ |

**Why so slow:** the reduction is `[16384, 4] → [16384, 1]` — only
4 elements per output, ~256 KiB total. Memory-bound floor on
MI355X: ~51 µs. Observed: **12.19 ms** — **~240× over floor**. HIP's
default `reduce_kernel<512, 1, ...>` is sized for huge reductions;
for 4 elements per output it allocates 32 blocks × 512 threads but
only 64 threads / block do useful work (12.5 % occupancy) plus
~5 µs / launch × 624 launches = 3.1 ms pure launch overhead.

Wrong kernel for a tiny inner reduction. Dispatcher cannot be fixed;
must avoid issuing the 624 launches.

---

## 3. The fix — `torch.compile`-fused Sinkhorn behind a flag

| component | path | what it does |
| --- | --- | --- |
| config flag | `primus/backends/megatron/core/models/deepseek_v4/deepseek_v4_transformer_config.py:117` | `use_v4_compiled_sinkhorn: bool = False` |
| compiled path | `primus/backends/megatron/core/transformer/hyper_connection.py:60-115` | `_compiled_sinkhorn_cache` keyed on `(n_iters, eps, in_dtype)` (shape NOT in key); `_build_compiled_sinkhorn` wraps the algorithm in `@torch.compile(fullgraph=True, dynamic=True)` |
| dispatch | `primus/backends/megatron/core/transformer/hyper_connection.py:136..191` | `sinkhorn_normalize(..., use_compiled: bool = False)` selects compiled vs eager |
| plumbing | `primus/backends/megatron/core/models/deepseek_v4/deepseek_v4_block.py:521,623,631,640,648` | `DeepseekV4HybridLayer` reads `config.use_v4_compiled_sinkhorn` → `HyperMixer.__init__` → `HyperMixer.compute_weights` |
| YAML | `primus/configs/models/megatron/deepseek_v4_base.yaml:66`, V4-Flash YAML | env-overridable, default `False` |
| run script | `run_deepseek_v4.sh:92,153`, `run_deepseek_v4_flash_proxy.sh:110` | proxy default `True`; baseline default `False` |

### Critical design note — why `dynamic=True` not `dynamic=False`

`dynamic=False` would force one Dynamo specialisation per shape AND
closures from the same factory share a `code` object — that collides
with Dynamo's `cache_size_limit=8`, and `fullgraph=True` then raises
`FailOnRecompileLimitHit` even though our `(n_iters, eps, dtype)`
cache is fine. `dynamic=True` ships ONE shape-generic Inductor
kernel; production sees only `[1, 4096, 4, 4]` so no specialisation
cost. This came up live during G32 release-tier and is documented at
`hyper_connection.py:60-86`.

---

## 4. Gates

| gate | status | numbers |
| --- | --- | --- |
| **G32** — FWD + BWD parity (compiled vs eager) | **GREEN** | 10 / 10 tests pass; fast tier `(B=2, S=64, K=4)` atol=1e-5; release tier `(B=1, S=4096, K=4)` atol=1e-5, marked `pytest.mark.slow`; cache hit on second call asserted; `HyperMixer` flag-propagation test included. Test file: `tests/unit_tests/megatron/transformer/deepseek_v4/test_v4_p29_compiled_sinkhorn.py`. |
| **G33a** — 10-iter EP=8 smoke (compiled flag on) | **GREEN** | `progress/p29/run_smoke_compiled_sinkhorn_ep8.sh`; no NaN / Inf / banned warnings; `lm_loss[10] = 9.258` vs P28 `9.258` (bit-for-bit at fixed seed); steady throughput **79.1 vs 77.5 TFLOP/s/GPU = +2.0 %**. |
| **G33b** — post-P29 trace + report (budget X1) | **GREEN, X1 met by ~1000×** | `progress/p29/run_baseline_trace_ep8_p29.sh`; report at `develop/profile/profile-after-p29-ep8-20260509.{md,html}`; on the critical `(1,4096,4,4)` shape: **624 → 16 launches** and **7607.9 ms → 0.2 ms** (−97.4 % launches, **−99.997 % kernel time**); across all variants of the slow reduce template: 7607.9 ms → 6.91 ms. |

---

## 5. Numerics — what hit and what did not

| metric | P28 baseline | post-P29 | delta vs baseline | budget | status |
|---|---:|---:|---:| --- | --- |
| `aten::sum` fp32 reduce — Σ kernel time (critical shape) | 7607.9 ms | 0.2 ms | **−99.997 %** | ≥ 50 % drop (**X1**) | **CLEAR** |
| `aten::sum` fp32 reduce — launches (critical shape) | 624 | 16 | −97.4 % | — | — |
| Steady iter wall time | 8.83 s | 8.63 s | −2.3 % | ≥ 35 % drop | **MISSED** |
| Steady TFLOP/s/GPU | 77.5 | 79.1 | +2.0 % | ≥ 60 % gain | **MISSED** |
| HBM peak | ~195 GiB | ~195 GiB | 0 % | no regression | **CLEAR** |
| `lm_loss` after iter 10 (fixed seed) | 9.258 | 9.258 | bit-for-bit | within 5e-2 | **CLEAR** |
| Multi-stream overlap factor | 1.87× | 1.00× | −0.87× | — | (informational; explains the wall-time miss) |

### Why the wall-time gain is small (and why this is fine)

The **multi-stream overlap factor dropping from 1.87× to 1.00×** is
the explanation, not a regression. The P28 trace was already running
the giant fp32 reduce on a separate HIP compute stream **in parallel
with V4 Triton attention BWD on stream-0**. Stream-0 was the wall-
time critical path. Killing the reduce removed a parallel hitchhiker;
it did not free critical-path bandwidth. The GPU remains 99.5 %
active; Σ kernel dur ≈ wall-clock GPU active = 8.58 s.

**The new top-3 wall-time bottlenecks are all V4 Triton attention BWD:**

| rank | kernel | total | % step |
|---:|---|---:|---:|
| 1 | `_v4_csa_attention_bwd_kernel` | 4.03 s | **46.8 %** |
| 2 | `_v4_attention_bwd_kernel` | 3.18 s | **36.8 %** |
| 3 | `_v4_attention_fwd_kernel` | 641 ms | 7.4 % |

Combined V4 attention (FWD + BWD across cr ∈ {0, 4, 128}) =
**8.0 s of 8.63 s = 92.6 % of step**. This is exactly the plan-5
P30 / P31 mandate, unchanged.

---

## 6. De-scope decisions recorded at P29 close

| task | decision | rationale |
| --- | --- | --- |
| Hand-Triton fall-back kernel (`v4_sinkhorn.py`) | **NOT NEEDED — out of tree** | `torch.compile` path over-shoots X1 by ~1000×. Hand-Triton stays as a plan-5 follow-up if a future change pulls Sinkhorn out of the small-input regime where Inductor + `triton.autotune` already win. |
| Global default flip of `use_v4_compiled_sinkhorn` | **DEFERRED outside current Plan-5 scope** | Wall-time delta is +2 % only; cold-compile cost is paid once per `(n_iters, eps, dtype)` per process — a footgun for short-iter unit-test harnesses. Proxy default flipped to `True` so P30 / P31 measure against the post-P29 plan-5 baseline; baseline `run_deepseek_v4.sh` stays `False` so future work can A/B against this knob without env plumbing. |
| Original seeded P29 candidates (a..e fused projections) | **DE-SCOPED at plan-5 P28** (re-confirmed here) | CPU-bound floor 0.3 % at production widths; small-op-launch tail < 10 % rule. Stays as plan-5 follow-ups. |

---

## 7. Hand-off to P30 / P31

The P29 trace **redirects but does not change** the plan-5 P30 / P31
mandate:

* **P30** — V4 Triton dense / HCA attention (`_v4_attention_fwd_kernel`
  + `_v4_attention_bwd_kernel`) covers `cr ∈ {0, 128}`. Now 36.8 %
  step (BWD) + 7.4 % step (FWD) = 44.2 % of wall time. **Y budget
  retargeted** at the BWD: ≥ 25 % drop on `_v4_attention_bwd_kernel`
  via per-shape autotune (BLOCK_M / BLOCK_N at `head_dim=512`),
  persistent-kernel sweep, and the HCA LSE-merge variant that
  plan-4 P25 deferred.
* **P31** — V4 Triton CSA attention (`_v4_csa_attention_fwd_kernel`
  + `_v4_csa_attention_bwd_kernel`) covers `cr == 4`. Now 46.8 %
  step (BWD) + 1.8 % step (FWD) = 48.6 % of wall time. **Z budget
  retargeted** at the BWD: ≥ 25 % drop on
  `_v4_csa_attention_bwd_kernel` via in-kernel `topk_idxs` gather +
  K-tile prefetch + autotune `BLOCK_K` for `K_topk=512`.

Combined plan-5 final target (unchanged from P28): **≥ 110
TFLOP/s/GPU steady at Sq=4096 EP=8 single-node**; **≥ 39 % over the
post-P29 79.1 TFLOP/s/GPU baseline**.

---

## 8. Artefacts shipped under `progress/p29/`

| file | purpose |
| --- | --- |
| `refinement.md` | full forensic write-up + chosen-fix design + perf-budget contract |
| `_forensics.py` / `_forensics2.py` / `_forensics3.py` | trace-dive helpers (gitignored output: `forensics_output.txt`, `forensics2_output.txt`, `forensics3_output.txt`) |
| `forensics3_after_p29.txt` | shape-attribution snapshot of the post-P29 trace (kept for the report) |
| `run_smoke_compiled_sinkhorn_ep8.sh` | G33a smoke script |
| `run_baseline_trace_ep8_p29.sh` | G33b chrome-trace capture script |
| `.gitignore` | excludes `*.log`, `*.json`, `forensics{,2,3}_output.txt`, `*.tgz`, `*.tfevents*`, `trace_*.txt` |
| **`p29-summary.md`** | this file |

External artefacts (kept, not in `progress/p29/`):

| file | purpose |
| --- | --- |
| `develop/profile/profile-after-p29-ep8-20260509.md` + `.html` | post-P29 baseline report (rendered via `develop/profile/_tools/render_baseline_report.py`) |
| `tests/unit_tests/megatron/transformer/deepseek_v4/test_v4_p29_compiled_sinkhorn.py` | G32 numerical equivalence test |

---

## 9. Known follow-ups carried forward

| item | reason | revisit at |
| --- | --- | --- |
| Hand-Triton fused-Sinkhorn kernel `v4_sinkhorn.py` | Out-of-tree; not needed at current shape | Only if Sinkhorn input shape changes (e.g. multi-node EP, different `hc_mult`) and the compiled path stops winning |
| Global default flip of `use_v4_compiled_sinkhorn` | +2 % wall-time delta at single-node EP=8 + cold-compile footgun for unit tests | Future follow-up outside current Plan-5 scope |
| Pre-existing failure in `test_v4_mtp.py::test_helper_pulls_norm_and_linear_from_v4_provider` | Confirmed unrelated to P29 (stash + re-run reproduces with NO P29 changes); known bad MagicMock chain in MTP fixture | Plan-5 follow-up (or sooner if MTP work re-opens) |

---

## 10. Commit chain

| commit | scope |
| --- | --- |
| **1ea7e7a8** | Plan-5 P29 (RESCOPED) — `torch.compile`-fused Sinkhorn behind `use_v4_compiled_sinkhorn` flag; G32 + G33a + G33b green; post-P29 baseline pinned for P30 / P31 |

Status row pinning will land in a follow-up `docs(deepseek-v4)[P29]:
pin status.md P29 cells to the P29 SHA (1ea7e7a8)` once the feature
commit is on origin.
