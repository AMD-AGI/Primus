# 03 — Plan-5 Test Strategy

> Plan-5 reuses the test conventions from `../plan-4/03-test-strategy.md`.
> Plan-5 is **measurement-driven**: every optimisation phase has a
> baseline cost it is buying down (set in P28's report), and every
> phase ships its own perf-budget gate. Correctness gates (G23 /
> G24 / G25 / G26 / G27 / G28 / G29 / G30) ratchet — every plan-5
> phase MUST keep them green. Plan-5 adds gates G31..G34b.

## Gate matrix

| Gate | Phase | Type | What it checks | Where it lives |
| --- | --- | --- | --- | --- |
| **G31** | P28 | smoke | TP=1 PP=1 EP=8 10-iter smoke under `run_deepseek_v4_flash_proxy.sh` (V4-Flash widths, 8 layers, all four perf knobs on, calibrated `seq_length`). Loss curve stable; no NaN / Inf; no banned warnings (plan-3 / plan-4 ratchet: no `submodule init failed` / `c10d::allreduce` / `fallback to nn.Linear` / `unsupported dispatcher module` / `using local Compressor\|Indexer` / `fallback to alltoall` / `v4_attention SMEM exceeded` / `v4_attention NaN observed` / `v4_csa_attention NaN observed`). | `deepseek-v4/develop/progress/p28/run_baseline_trace_ep8.sh` (drives the proxy with `PROFILE=True --profile_step_start 6 --profile_step_end 7`) |
| **G31a** | P28 | report | Baseline analysis report (md + html) under `deepseek-v4/develop/profile/profile-baseline-ep8-<YYYYMMDD>.{md,html}` covering run config, per-iter wall time (cold / warm / steady), GPU vs CPU active / idle %, top-N kernels, kernel launch count + interval, module-level CPU time attribution, comm time, ranked bottleneck list, and **per-phase improvement budgets** that pin the X / Y / Z numbers in `01-roadmap.md`. | `deepseek-v4/develop/profile/profile-baseline-ep8-<YYYYMMDD>.{md,html}` (the report itself is the artefact) |
| **G32** | P29 (RESCOPED) | runtime (GPU) | Forward + backward equivalence: compiled `sinkhorn_normalize` (`torch.compile(fullgraph=True, dynamic=True)`) matches the eager loop at fast tier (`B=2 S=64 K=4`, fp32 + autograd-fp32) within `atol=1e-5, rtol=1e-5`. Release tier at V4-Flash production input shape (`B=1 S=4096 K=4`) marked `pytest.mark.slow`, same tolerance. Parametrised over `n_iters ∈ {5, 20}` so the cache key path is exercised. Cold-compile time recorded as a `print()` line in the test for the P29 status row. | `tests/unit_tests/megatron/transformer/deepseek_v4/test_v4_p29_compiled_sinkhorn.py` |
| **G33a** | P29 (RESCOPED) | smoke | TP=1 PP=1 EP=8 10-iter smoke under the proxy with `USE_V4_COMPILED_SINKHORN=True`. Plan-4 ratchet (G23..G30) green; plan-3 / plan-4 banned-warning grep set returns 0; `lm_loss` after 10 iters within 5e-2 of the P28 baseline at the same fixed seed. | `deepseek-v4/develop/progress/p29/run_smoke_compiled_sinkhorn_ep8.sh` |
| **G33b** | P29 (RESCOPED) | perf + report | Capture chrome-trace iter 6 → 7 with `USE_V4_COMPILED_SINKHORN=True`; render `develop/profile/profile-after-p29-ep8-<YYYYMMDD>.{md,html}` reusing the P28 report tooling (`develop/profile/_tools/render_baseline_report.py`). **Perf gate (X1 from P28 report)**: `aten::sum` fp32 reduce kernel time drops by ≥ 50 % vs the P28 baseline (7.61 s → ≤ 3.81 s); steady iter wall time drops by ≥ 35 %; steady TFLOP/s/GPU ≥ 1.6 × P28 baseline. If the gate fails, escalate to the hand-Triton fall-back kernel before closing P29 (status logged in `progress/p29/post_compile_results.md`). | `deepseek-v4/develop/progress/p29/run_baseline_trace_ep8_p29.sh` + `develop/profile/profile-after-p29-ep8-<YYYYMMDD>.{md,html}` |
| **G34** | P30 | runtime (GPU) | SWA K-loop pruning equivalence: dense `v4_attention` with kernel-native `swa_window` matches the existing additive-mask semantics, and HCA split-mask mode (`pool_mask + hca_local_seqlen`) matches the original full `cat([local_mask, pool_mask])` additive-mask semantics. Fast tier and release tier cover FWD + BWD, sink, bf16 tolerances, and `head_dim=512` V4-Flash / V4-Pro shapes. | `tests/unit_tests/megatron/transformer/deepseek_v4/test_v4_p25_v4_attention_fwd.py` + `test_v4_p25_v4_attention_bwd.py` (`-m "not slow"` and `-m slow`) |
| **G34a** | P30 | smoke + perf | TP=1 PP=1 EP=8 10-iter proxy smoke with `USE_V4_COMPILED_SINKHORN=True` and V4 Triton dense/CSA attention on. No NaN / Inf / banned warnings; fixed-seed `lm_loss` remains aligned with post-P29; capture a chrome trace and render `profile-after-p30-ep8-<YYYYMMDD>.{md,html}`. Perf gate is measured against post-P29: all five `_v4_attention_bwd_kernel` launches, including the two cr=128 HCA launches, drop materially and steady TFLOP/s/GPU improves. | `deepseek-v4/develop/progress/p30/run_smoke_v4_attention_swa_prune_ep8.sh` + `run_baseline_trace_ep8_p30.sh` + `deepseek-v4/develop/profile/profile-after-p30-ep8-<YYYYMMDD>.{md,html}` |
| **G34b** | P31 | runtime + smoke + perf | CSA in-kernel top-K gather/scatter equivalence: `v4_csa_attention_from_pool(q, k_local, v_local, pool, topk_idxs, ...)` matches eager CSA with wrapper-side `torch.gather(pool, topk_idxs)` and autograd scatter to `dpool`. Fast and release tiers cover FWD + BWD, invalid `topk_idxs == -1`, duplicate top-K slots, sink on/off, fp32/bf16, and `head_dim=512`. Then rerun the EP8 proxy smoke/trace and render `profile-after-p31-ep8-<YYYYMMDD>.{md,html}`; perf gate is improvement vs P30b with no NaN / Inf / banned warnings. Follow-up CSA BWD tuning uses the standalone EP8-shape microbenchmark before any full-training trace. | `tests/unit_tests/megatron/transformer/deepseek_v4/test_v4_p31_v4_csa_in_kernel_gather.py` + `deepseek-v4/develop/progress/p31/bench_csa_attention_ep8.py` + `deepseek-v4/develop/progress/p31/run_smoke_csa_in_kernel_gather_ep8.sh` + `run_baseline_trace_ep8_p31.sh` |

## Plan-4 ratchet (every plan-5 commit MUST keep these green)

Plan-5 inherits the plan-4 correctness ratchet:

- **G23 / G24** (`v4_attention` FWD + BWD equivalence at fast tier).
- **G25** (`v4_attention` determinism with `attn_dropout=0.0`).
- **G26 / G27** (`v4_csa_attention` FWD + BWD equivalence at fast tier).
- **G28** (release-tier shape gate at production V4-Flash + V4-Pro
  dims, `pytest.mark.slow`).
- **G29** (dispatch precedence + startup log line).
- **G30** (TP=1 PP=1 EP=8 smoke at `seq_length=128`).

Each plan-5 phase opens with a "ratchet check" — `pytest -q
tests/unit_tests/megatron/transformer/deepseek_v4/` (with
`--run-slow` for G28) — and the phase row in `progress/status.md`
records the pass count. Any drop in the green count blocks the phase
commit.

## Banned-warning ratchet

Plan-5 inherits the plan-3 + plan-4 ratchet (no `submodule init failed`
/ `c10d::allreduce` / `fallback to nn.Linear` / `unsupported dispatcher
module` / `using local Compressor|Indexer` / `fallback to alltoall` /
`v4_attention SMEM exceeded` / `v4_attention NaN observed` /
`v4_csa_attention NaN observed`) and adds:

1. `"sinkhorn_normalize compile error"` / `"torch._inductor"` ERROR-
   level traceback — torch.compile of `sinkhorn_normalize` MUST compile
   cleanly on the proxy shape; latent compile errors that only surface
   on a different shape block G32.
2. `"DeepEP contract violation"` — any log line indicating
   `moe_router_dtype != fp32` or `moe_shared_expert_overlap=True` when
   `use_turbo_deepep=True` fails the smoke.

## Perf-budget contract

Plan-5 uses **steady-iter TFLOP/s/GPU** as the headline metric (matches
plan-3 P20's V4-aware FLOPS formula and plan-4 P27 G30's "+37 % vs
P22 eager" delta). Secondary metrics:

- **Step time (ms / iter)** — for plotting per-phase deltas.
- **`aten::sum` fp32 reduce kernel total (ms / iter)** — P29 budget X1:
  ≤ 50 % of P28 baseline (7.61 s → ≤ 3.81 s).
- **GPU stream-0 active % / multi-stream overlap factor** — P28
  baseline already shows the GPU is 99.7 % active with overlap factor
  1.87×; the post-P29 trace should preserve or improve both.
- **CPU active % / idle %** — P28 baseline CPU-bound floor is 0.3 %;
  no plan-5 phase regresses this.
- **Top-N kernel time** — per-phase improvement target tied to a
  specific kernel name in the P28 report's bottleneck list.
- **HBM peak (MiB)** — should not regress; P28 calibration confirmed
  ~68 % HBM at `Sq=4096`, no plan-5 phase pushes it higher.
- **Kernel launch count / iter** — P29 should drop the launch count
  by ~600 (the 624 `aten::sum` launches go to ~1 launch per `sinkhorn_
  normalize` call).

The P28 report sets the per-phase X / Y / Z budgets; the per-phase
G33 / G34 gates assert ≥ budget and **document the actual delta** in
the phase status row. Plan-5 does not commit to a specific budget at
plan-time — that would require pre-knowledge of the trace; it
commits to a measurement-driven budget that the report owns in
writing.

## GPU-toy harness

G32 + G34 require GPU. The harness:

- Reuses plan-4's `compare_fwd_bwd` (`tests/unit_tests/megatron/
  transformer/deepseek_v4/v4_attention_test_utils.py`) and
  `v4_attention_shapes.py` shape fixtures.
- Reuses plan-4's `tests/unit_tests/megatron/transformer/deepseek_v4/
  conftest.py` `torch.cuda.empty_cache()` autouse fixture so
  release-tier tests do not OOM each other.
- Reuses plan-4's `pytest.mark.slow` opt-in for release-tier shapes
  (default CI runs fast tier only; `pytest --run-slow` runs the full
  matrix).
- Skips on machines without CUDA / Triton (`pytest.importorskip(
  "triton")`).

## Reporting hand-off

Plan-5 closes after P31 with the latest EP8 proxy trace/report and
the P31 entry in `../progress/status.md`. The plan-5 close-out note
records:

- Commit SHAs for P28 / P29 / P30 / P31.
- The baseline → final TFLOP/s/GPU delta + per-phase share.
- Per-phase gate totals (G31 / G32 / G33a / G33b / G34 / G34a / G34b) +
  plan-4 ratchet totals.
- Follow-ups surfaced (FP8 / FP4 / mxfp4, multi-node EP, long-
  context, convergence, HF state-dict adapter).
