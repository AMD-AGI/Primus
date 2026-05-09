# 03 — Plan-5 Test Strategy

> Plan-5 reuses the test conventions from `../plan-4/03-test-strategy.md`.
> Plan-5 is **measurement-driven**: every optimisation phase has a
> baseline cost it is buying down (set in P28's report), and every
> phase ships its own perf-budget gate. Correctness gates (G23 /
> G24 / G25 / G26 / G27 / G28 / G29 / G30) ratchet — every plan-5
> phase MUST keep them green. Plan-5 adds five new gates (G31..G35).

## Gate matrix

| Gate | Phase | Type | What it checks | Where it lives |
|---|---|---|---|---|
| **G31** | P28 | smoke | TP=1 PP=1 EP=8 10-iter smoke under `run_deepseek_v4_flash_proxy.sh` (V4-Flash widths, 8 layers, all four perf knobs on, calibrated `seq_length`). Loss curve stable; no NaN / Inf; no banned warnings (plan-3 / plan-4 ratchet: no `submodule init failed` / `c10d::allreduce` / `fallback to nn.Linear` / `unsupported dispatcher module` / `using local Compressor\|Indexer` / `fallback to alltoall` / `v4_attention SMEM exceeded` / `v4_attention NaN observed` / `v4_csa_attention NaN observed`). | `deepseek-v4/develop/progress/p28/run_baseline_trace_ep8.sh` (drives the proxy with `PROFILE=True --profile_step_start 6 --profile_step_end 7`) |
| **G31a** | P28 | report | Baseline analysis report (md + html) under `deepseek-v4/develop/profile/profile-baseline-ep8-<YYYYMMDD>.{md,html}` covering run config, per-iter wall time (cold / warm / steady), GPU vs CPU active / idle %, top-N kernels, kernel launch count + interval, module-level CPU time attribution, comm time, ranked bottleneck list, and **per-phase improvement budgets** that pin the X / Y / Z / W numbers in `01-roadmap.md`. | `deepseek-v4/develop/profile/profile-baseline-ep8-<YYYYMMDD>.{md,html}` (the report itself is the artefact) |
| **G32.a** | P29 candidate (a) | runtime (GPU) | Forward + backward equivalence: `v4_fused_q_proj(hidden, q_down_w, q_up_w, q_norm_w, q_pe_freqs, eps)` matches the eager small-op chain at every (variant ∈ {flash, pro}) × (S ∈ {small, medium}) × (dtype ∈ {fp32, bf16}) shape, within the dtype tolerance budget (`fp32 atol=1e-4`, `bf16 fwd atol=2e-2`, `bf16 bwd atol=5e-2`). Release-tier extension at V4-Flash + V4-Pro production widths, marked `pytest.mark.slow`. | `tests/unit_tests/megatron/transformer/deepseek_v4/test_v4_p29_fused_q_proj.py` |
| **G32.b** | P29 candidate (b) | runtime (GPU) | Forward + backward equivalence for `v4_fused_kv_proj` (single-latent KV projection + LN + position-axis split). Same parameter grid + tolerance budget as G32.a. | `test_v4_p29_fused_kv_proj.py` |
| **G32.c** | P29 candidate (c) | runtime (GPU) | Forward + backward equivalence for `v4_fused_o_proj` (grouped low-rank output projection). Same parameter grid + tolerance budget. | `test_v4_p29_fused_o_proj.py` |
| **G32.d** | P29 candidate (d) | runtime (GPU) | Forward + backward equivalence for `v4_fused_compressor` and `v4_fused_indexer`. Same parameter grid; restricted to `compress_ratio ∈ {4, 128}` shapes (compressor) and `compress_ratio == 4` shapes (indexer). | `test_v4_p29_fused_compressor.py`, `test_v4_p29_fused_indexer.py` |
| **G32.e** | P29 candidate (e) | runtime (GPU) | Forward + backward equivalence for `v4_fused_moe_router` (gate-softmax + topk + permute-mask). Parametrised over `(use_turbo_deepep on / off)` so any DeepEP-contract violation is caught locally. Asserts `moe_router_dtype=fp32` is preserved (DeepEP contract). | `test_v4_p29_fused_moe_router.py` |
| **G33** | P29 + P30 + P31 | smoke + perf | TP=1 PP=1 EP=8 10-iter smoke under the proxy with the chosen optimisation switches on. Same correctness contract as G31 (no NaN / no banned warnings). **Perf gate**: steady-iter TFLOP/s/GPU ≥ baseline + per-phase budget (X for P29, Y for P30, Z for P31; budgets pinned in P28's report). | `deepseek-v4/develop/progress/p29..p31/run_perf_smoke_*.sh` |
| **G34** | P30 | runtime (GPU) | HCA LSE-merge variant equivalence: `v4_attention(use_v4_attention_lse_merge=True)` matches the single-kernel-with-additive-bias variant at every (variant ∈ {flash, pro}) × (compress_ratio == 128) × (S ∈ {small, medium}) × (dtype ∈ {fp32, bf16}) shape, within the bf16 tolerance budget. Release-tier extension marked `pytest.mark.slow`. | `tests/unit_tests/megatron/transformer/deepseek_v4/test_v4_p30_lse_merge.py` |
| **G35** | P32 | smoke + perf | Final TP=1 PP=1 EP=8 10-iter smoke under the proxy with all P29 / P30 / P31 / P32 optimisations on. Same correctness contract as G31. **Perf gate**: steady-iter TFLOP/s/GPU ≥ baseline + W (cumulative budget pinned in P28's report). HBM peak ≤ baseline + reserve (in-kernel CSA gather should free ≈ 64 GiB / microbatch at V4-Flash). | `deepseek-v4/develop/progress/p32/run_final_trace_ep8.sh` + `deepseek-v4/develop/profile/profile-final-ep8-<YYYYMMDD>.{md,html}` |

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

1. `"v4_fused_* compile error"` — torch.compile / Triton compilation
   errors fail the smoke loud. Fusion code MUST compile on the proxy
   shape; latent compile errors that only surface on a different shape
   block the corresponding G32.
2. `"DeepEP contract violation"` — any log line indicating
   `moe_router_dtype != fp32` or `moe_shared_expert_overlap=True` when
   `use_turbo_deepep=True` fails the smoke.

## Perf-budget contract

Plan-5 uses **steady-iter TFLOP/s/GPU** as the headline metric (matches
plan-3 P20's V4-aware FLOPS formula and plan-4 P27 G30's "+37 % vs
P22 eager" delta). Secondary metrics:

- **Step time (ms / iter)** — for plotting per-phase deltas.
- **GPU stream-0 active %** — should rise from baseline after
  P29 (small-op tail closed → fewer launches → less idle).
- **CPU active % / idle %** — should fall from baseline after P29.
- **Top-N kernel time** — per-phase improvement target tied to a
  specific kernel name in the P28 report's bottleneck list.
- **HBM peak (MiB)** — should drop materially after P31 (in-kernel
  CSA gather drops the wrapper-side `[B, H, Sq, K, D]` materialisation).
- **Kernel launch count / iter** — should fall from baseline after
  P29 (every fused chain drops 5–10 launches per layer).

The P28 report sets the per-phase X / Y / Z / W budgets; the per-phase
G33 / G35 gates assert ≥ budget and **document the actual delta** in
the phase status row. Plan-5 does not commit to a specific budget at
plan-time — that would require pre-knowledge of the trace; it
commits to a measurement-driven budget that the report owns in
writing.

## GPU-toy harness

G32.{a..e} + G34 require GPU. The harness:

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

Plan-5 closes with the P32 entry in `../progress/status.md` plus the
final trace + report under `develop/profile/profile-final-*`. The
plan-5 hand-off note (in `plan-5/02-phase-details.md`) records:

- Commit SHAs for P28 / P29 / P30 / P31 / P32.
- The baseline → final TFLOP/s/GPU delta + per-phase share.
- Per-phase gate totals (G31 / G32.{a..e} / G33 / G34 / G35) +
  plan-4 ratchet totals.
- Follow-ups surfaced (FP8 / FP4 / mxfp4, multi-node EP, long-
  context, convergence, HF state-dict adapter).
