# 03 — Plan-6 Test Strategy

> Plan-6 reuses the test conventions from
> `../plan-5/03-test-strategy.md`. Plan-6 is **measurement-driven**:
> every fusion phase has a baseline cost it is buying down (set in
> the plan-5 P32 final trace or the per-phase microbenchmark), and
> every phase ships its own perf-budget gate + a default-on env knob
> that flips to default-off if the EP8 proxy A/B regresses. The
> plan-5 P32 RoPE bf16 cast bug is the load-bearing precedent: the
> proxy A/B is the source of truth, not the microbench.

Correctness gates (plan-4 G23 / G24 / G26 / G27 / G29 plus plan-5
G32 / G34 / G34b / G35) ratchet — every plan-6 phase MUST keep them
green. Plan-6 adds G36..G42.

## Gate matrix

| Gate | Phase | Type | What it checks | Where it lives |
| --- | --- | --- | --- | --- |
| **G36** | P33 | runtime (CPU) | SWA visible-pair pruning: `_visible_pairs(swa_window=128, S_eff=4096, compress_ratio ∈ {0, 4, 128}, index_topk=512)` returns the analytic visible-pair count derived in `develop/perf/attention_perf.md`. Independent reference path: a small Python loop over `(t, s)` causal pairs inside the test. Parametrise `hc_mult ∈ {1, 4}`. | `tests/unit_tests/backends/megatron/test_deepseek_v4_flops_patches.py::TestG36SWAVisiblePairs` |
| **G36a** | P33 | runtime (CPU) | HyperConnection `fn.weight` matmul count: `compute_v4_flops` on the V4-Flash 8-layer proxy config (`B=8, S=4096, L=8, K=4, D=4096, mtp=0`) returns a `_V4FlopsBreakdown.hc` that equals a hand-computed reference (int64 byte-equal). | `test_deepseek_v4_flops_patches.py::TestG36aHCMatmul` |
| **G37** | P34 | runtime (GPU) | `_stack_grouped_weight_fwd_kernel` FWD bit-equal vs `torch.stack(weights).transpose(1, 2).contiguous()` at fast tier (`E=4, K=8, N=8, fp32`) and release tier (`E=32, K=4096, N=2048, bf16`, `pytest.mark.slow`). BWD `torch.autograd.gradcheck` at fast tier fp32 small shape. BWD bit-equal vs eager at release tier (`atol=0` — pure layout transform, no float math). | `tests/unit_tests/megatron/extensions/test_p34_stack_grouped_weight_triton.py` |
| **G38** | P35 | runtime (GPU) | `apply_interleaved_partial_rope` Triton FWD parity vs eager body within bf16 `atol=1e-3 rtol=1e-3` (fp32 `atol=1e-6 rtol=1e-6`) at fast tier (`B=2 S=64 H=4 head_dim=8 rotary_dim=8`) and release tier (`B=1 S=4096 H=64 head_dim=512 rotary_dim=64`, `pytest.mark.slow`). BWD `gradcheck` at fast tier fp32; BWD bf16 parity at release tier within the plan-4 P25 BWD tolerance budget. Parametrise `rotary_dim ∈ {0, 16, 64}` so the no-op and non-V4 branches are exercised. | `tests/unit_tests/megatron/transformer/deepseek_v4/test_p35_rope_triton.py` |
| **G39** | P36 | runtime (GPU) | Triton `sinkhorn_normalize` FWD parity (a) vs eager loop and (b) vs plan-5 P29 `torch.compile` path within `atol=1e-5 rtol=1e-5` fp32 / `atol=1e-3 rtol=1e-3` bf16. BWD `gradcheck` at fast tier fp32; BWD bf16 parity at release tier. Parametrise `n_iters ∈ {5, 20}`. **Doubly-stochastic property check**: `row_sum.allclose(1, atol=eps*K)` and `col_sum.allclose(1, atol=eps*K)`. | `tests/unit_tests/megatron/transformer/deepseek_v4/test_p36_sinkhorn_triton.py` |
| **G40** | P37 | runtime (GPU) | (a) Each of the three Triton sub-kernels (`_hc_post_linear_glue`, `_hc_collapse`, `_hc_expand_outer`) FWD parity vs the eager body extracted from `HyperMixer.compute_weights` / `collapse` / `expand`, within bf16 `atol=1e-3 rtol=1e-3`. (b) End-to-end `HyperMixer.compute_weights(x)` + `HyperHead.forward(x)` FWD parity vs the existing plan-2 P14 unit-test reference, within the same tolerance. (c) BWD `gradcheck` at fast tier fp32. (d) **Dtype-contract parametrise**: `(in_dtype, compute_dtype, out_dtype) ∈ {(bf16, fp32, bf16), (fp32, fp32, fp32)}`. | `tests/unit_tests/megatron/transformer/deepseek_v4/test_p37_hyper_connection_glue_triton.py` |
| **G41** | P38 | runtime (GPU) | Triton `Indexer` scoring FWD `scores` parity vs eager `Indexer.forward` (extracted pre-`topk`) within bf16 `atol=5e-3 rtol=5e-3`. Post-`topk` `topk_idxs` bit-equal vs the eager full chain (load-bearing — sparse top-K indices must be exact). BWD `gradcheck` at fast tier fp32. Release-tier shape (`B=1, S=4096, P=1024, H=8, Hd=128`) marked `pytest.mark.slow`. | `tests/unit_tests/megatron/transformer/deepseek_v4/test_p38_indexer_triton.py` |
| **G42** | P39 | runtime (GPU) | V4 router post-logits FWD `probs` / `routing_map` bit-equal vs eager across 3 score functions (`softmax`, `sigmoid`, `sqrtsoftplus`) × {with / without expert bias} × {hash / topk} = 12 cases at fast (`N=128, E=32, K=4`) and release tiers (`N=4096, E=256, K=6`, `pytest.mark.slow`). BWD `gradcheck` at fast tier fp32 small shape. **Sparse output comparison** — `(probs.nonzero(), routing_map.nonzero())` bit-equal between Triton and eager. | `tests/unit_tests/megatron/transformer/deepseek_v4/test_p39_router_post_triton.py` |
| **G37a / G38a / G39a / G40a / G41a / G42a** | P34..P39 | smoke + perf | TP=1 PP=1 EP=8 10-iter proxy smoke under `run_deepseek_v4_flash_proxy.sh` with the corresponding `PRIMUS_<NAME>_TRITON=1` flag (every prior flag also `1`). Plan-4 ratchet (G23..G30) plus plan-5 G32 / G34 / G34b / G35 stay green. Plan-3 / plan-4 banned-warning grep returns 0. `lm_loss` after 10 iters within `5e-2` of the post-P32-final baseline at fixed seed (algorithm is unchanged for every fusion). | `progress/p3X/run_smoke_p3X_ep8.sh` |
| **G37b / G38b / G39b / G40b / G41b / G42b** | P34..P39 | perf + report | Capture chrome-trace iter 6 → 7 with the flag on; render `develop/profile/profile-after-p3X-ep8-<YYYYMMDD>.{md,html}` reusing `develop/profile/_tools/render_baseline_report.py` (plan-5 P28 tooling). Phase-specific perf gate: (P34) `hipMemcpyWithStream` ≤ 50 ms; (P35) `CatArrayBatchedCopy_contig` ≈ 0; (P36) `Torch-Compiled Region` + `CompiledFunctionBackward` ≈ 0; (P37) `elementwise_manual_unroll<128, 8>` ≤ 30 ms; (P38) Indexer score chain not visible above the trace-noise floor; (P39) router post-logits chain not visible. Plus: end-to-end iter time drops by the per-phase budget set in `01-roadmap.md`. **If proxy A/B regresses, the env flips to `"0"` and the regression is documented in `progress/p3X/p3X-summary.md`** (plan-5 P32 split-BWD / segreduce precedent). | `progress/p3X/run_baseline_trace_ep8_p3X.sh` + `develop/profile/profile-after-p3X-ep8-<YYYYMMDD>.{md,html}` |
| **G40 (close-out)** | P40 | docs | `develop/perf/elem_fusion.md` exists with one row per shipped fusion (P34..P39). `develop/perf/proxy_ep8.md` `P40 final` row pinned. Every `[x]` row in Phase 33..40 of `progress/status.md` has a commit SHA. Every `progress/p3X/p3X-summary.md` follows R2.1. | `develop/perf/elem_fusion.md` + `develop/perf/proxy_ep8.md` + `progress/status.md` + `progress/p3X/p3X-summary.md` |

## Plan-4 + Plan-5 ratchet (every plan-6 commit MUST keep these green)

Plan-6 inherits both ratchets:

- **Plan-4 G23 / G24** (`v4_attention` FWD + BWD equivalence, fast).
- **Plan-4 G25** (`v4_attention` determinism, `attn_dropout=0.0`).
- **Plan-4 G26 / G27** (`v4_csa_attention` FWD + BWD equivalence, fast).
- **Plan-4 G28** (release-tier shape gate, `pytest.mark.slow`).
- **Plan-4 G29** (dispatch precedence + startup log line).
- **Plan-4 G30** (TP=1 PP=1 EP=8 smoke at `seq_length=128`).
- **Plan-5 G32** (`torch.compile` Sinkhorn FWD + BWD parity).
- **Plan-5 G33a / G33b** (post-P29 proxy smoke + trace).
- **Plan-5 G34** (SWA prune dense / HCA equivalence).
- **Plan-5 G34a** (post-P30 proxy smoke + trace + report).
- **Plan-5 G34b** (P31 CSA in-kernel gather/scatter equivalence + smoke).
- **Plan-5 G35** (P32 operator-microbench-driven kernel speed-ups).

Each plan-6 phase opens with a "ratchet check" — `pytest -q
tests/unit_tests/megatron/transformer/deepseek_v4/` (with
`--run-slow` for release-tier gates) — and the phase row in
`progress/status.md` records the pass count. Any drop in the green
count blocks the phase commit.

## Banned-warning ratchet

Plan-6 inherits the plan-3 + plan-4 + plan-5 ratchet (no
`submodule init failed` / `c10d::allreduce` / `fallback to nn.Linear`
/ `unsupported dispatcher module` / `using local Compressor|Indexer`
/ `fallback to alltoall` / `v4_attention SMEM exceeded` /
`v4_attention NaN observed` / `v4_csa_attention NaN observed` /
`sinkhorn_normalize compile error` / `torch._inductor` ERROR /
`DeepEP contract violation`) and adds:

1. `"PRIMUS_<NAME>_TRITON kernel asserted"` — every plan-6 Triton
   kernel guards its preconditions (contiguity / dtype / rank);
   any assertion fired in production proxy means a bad kwarg made
   it through the wrapper. The G37a..G42a smokes grep for it.
2. `"V4FlopsBreakdown.hc not present"` — plan-3 P20 log-line grep
   already exists for `attn_qkv_o` / `attn_scores` / etc.; P33
   extends it with the new `hc` row. Smokes assert it is printed
   exactly once at rank 0.

## Perf-budget contract

Plan-6 uses **steady-iter wall-clock (ms / iter)** + **corrected
TFLOP/s/GPU** as the headline metrics (P33 provides the corrected
denominator that the remaining phases measure against). Secondary
metrics per phase:

- **Per-kernel trace bucket** — each phase has a specific GPU-time
  bucket it is buying down (P34: `hipMemcpyWithStream`; P35:
  `CatArrayBatchedCopy_contig`; P36: `Torch-Compiled Region` +
  `CompiledFunctionBackward`; P37: `elementwise_manual_unroll<128,
  8>`; P38: Indexer scoring chain; P39: router post-logits chain).
- **Microbench wall-clock** — each phase ships a `<ms> ms |
  <throughput>` cell in `develop/perf/elem_fusion.md`.
- **GPU stream-0 active %** — the post-plan-5 baseline is 99.7 %;
  no plan-6 phase regresses this.
- **CPU active % / idle %** — the post-plan-5 baseline CPU-bound
  floor is 0.3 %; no plan-6 phase regresses this.
- **HBM peak (MiB)** — should not regress; the P34 Triton path
  allocates the same output buffer as the eager stack (so HBM
  steady state is identical; transient working set drops).
- **Kernel launch count / iter** — every phase reduces the count
  by at least the number of ops it fuses.

Per-phase target deltas (cumulative from plan-5 P32 final `603 ms`):

| Phase | Target iter (ms) | Cumulative delta vs P32 final |
| --- | ---: | ---: |
| P33 (denominator only) | 603 | 0 |
| P34 (stack-grouped-weight) | ≤ 450 | -150 |
| P35 (RoPE) | ≤ 420 | -180 |
| P36 (Sinkhorn) | ≤ 405 | -195 |
| P37 (HC elemwise) | ≤ 385 | -215 |
| P38 (Indexer) | ≤ 370 | -230 |
| P39 (Router post-logits) | ≤ 360 | -240 |
| P40 (close-out) | ≤ 360 | -240 |

Targets are **best-effort**. The phase row records the actual
delta in writing (R2.4 status-pin convention). If a phase
regresses end-to-end iter time, the phase ships with its env
default flipped to `0` and the regression is documented in
`progress/p3X/p3X-summary.md` (plan-5 P32 split-BWD / segreduce
default-OFF then default-ON precedent — once the root cause is
fixed, the default flips back).

## GPU-toy harness

G37..G42 require GPU. The harness:

- Reuses plan-4's `compare_fwd_bwd` (`tests/unit_tests/megatron/
  transformer/deepseek_v4/v4_attention_test_utils.py`) and
  `v4_attention_shapes.py` shape fixtures where applicable
  (P35 RoPE Triton).
- Reuses plan-4's `conftest.py` `torch.cuda.empty_cache()` autouse
  fixture so release-tier tests do not OOM each other.
- Reuses plan-4's `pytest.mark.slow` opt-in for release-tier shapes
  (default CI runs fast tier only; `pytest --run-slow` runs the
  full matrix).
- Skips on machines without CUDA / Triton (`pytest.importorskip(
  "triton")`).
- P34 uses a new harness module
  `tests/unit_tests/megatron/extensions/conftest.py` (mirrors the
  V4 attention conftest) since it lives under `extensions/` not
  `transformer/deepseek_v4/`.

## Reporting hand-off

Plan-6 closes after P40 with the latest EP8 proxy trace/report,
the per-phase microbenchmark numbers (one row per fusion in
`develop/perf/elem_fusion.md`), and the P33..P40 entries in
`progress/status.md`. The plan-6 close-out note records:

- Commit SHAs for P33 / P34 / P35 / P36 / P37 / P38 / P39 / P40.
- The plan-5-P32-final → plan-6-P40 iter-time + TFLOP/s/GPU delta
  + per-phase share.
- Per-kernel microbenchmark deltas (one per fusion).
- Per-phase gate totals (G36..G42 + a..b suffixes) + plan-4 /
  plan-5 ratchet totals.
- Follow-ups surfaced (e.g. structural refactor of grouped-MLP to
  `[E, K, N]` single `nn.Parameter` if P34 Triton fuse does not
  fully close the gap; HC `expand` matmul + outer-product joint
  fuse if downstream microbench shows the boundary kernel is the
  next bottleneck).
