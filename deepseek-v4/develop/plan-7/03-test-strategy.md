# 03 — Plan-7 Test Strategy

> Plan-7 reuses the test conventions from
> `../plan-6/03-test-strategy.md`.  Plan-7 is **measurement-driven**:
> every fusion phase has a baseline cost it is buying down (set in
> the post-P44 trace or the per-phase microbenchmark), and every
> phase ships its own perf-budget gate + a default-on env knob that
> flips to default-off if the EP8 proxy A/B regresses.

Correctness gates (plan-4 G23 / G24 / G26 / G27 / G29 plus plan-5
G32 / G34 / G34b / G35 plus plan-6 G36..G46) ratchet — every
plan-7 phase MUST keep them green.  Plan-7 adds G47..G50.

## Gate matrix

| Gate | Phase | Type | What it checks | Where it lives |
| --- | --- | --- | --- | --- |
| **G47** | P45 | runtime (GPU) | Custom Triton fused Adam FWD bit-equal vs upstream Apex / TE Adam at fast tier (`[8 params × 4096 elements]`, fp32, fp32 master).  BF16 master + remainder: ULP-difference ≤ 1 vs upstream at fast tier; bit-equal upper 16 bits.  10-step micro-rollup: max abs diff in param ≤ 1e-3.  Release-tier slow: full V4-Flash param list, 100 steps, loss-curve diff ≤ 1e-3 vs upstream at fixed seed. | `tests/unit_tests/megatron/extensions/test_p45_fused_adam_triton.py` |
| **G47a / G47b** | P45 | smoke + perf | EP=8 10-iter proxy smoke with `PRIMUS_FUSED_ADAM_TRITON=1`; plan-4/5/6 ratchet stays green; `lm_loss` after 10 iters within `5e-2` of post-P44 baseline.  Chrome-trace iter 6 → 7; Adam ε-add + multi-tensor-master buckets drop by ≥ 150 ms; iter time drops ≥ 100 ms vs P44 final. | `progress/p45/run_smoke_p45_ep8.sh` + `develop/profile/profile-after-p45-ep8-<YYYYMMDD>.{md,html}` |
| **G48** | P46 | runtime (GPU) | Fused grad-scale Triton kernel FWD bit-equal vs upstream `multi_tensor_scale` at fast tier fp32 + bf16. | `tests/unit_tests/megatron/extensions/test_p46_fused_grad_scale.py` |
| **G48a / G48b** | P46 | smoke + perf | EP=8 smoke + chrome trace with `PRIMUS_FUSED_GRAD_SCALE=1`; trace `multi_tensor<scale>` bucket drops to ≈ 0; iter time drops ≥ 3 ms vs P45 final. | `progress/p46/run_baseline_trace_ep8_p46.sh` + `develop/profile/profile-after-p46-ep8-<YYYYMMDD>.{md,html}` |
| **G49** | P47 | runtime (GPU) | Fused grad-norm clip Triton pipeline FWD bit-equal vs upstream `clip_grad_norm_fp32` at fast tier fp32 + bf16.  L2 norm reduction uses the same order as the upstream multi-tensor functor (left-to-right, no block-shuffle). | `tests/unit_tests/megatron/extensions/test_p47_fused_grad_norm_clip.py` |
| **G49a / G49b** | P47 | smoke + perf | EP=8 smoke + chrome trace with `PRIMUS_FUSED_GRAD_NORM_CLIP=1`; trace `reduce<l2norm_bf16>` + `multi_tensor<l2norm>` buckets drop to ≈ 0; iter time drops ≥ 6 ms vs P46 final. | `progress/p47/run_baseline_trace_ep8_p47.sh` + `develop/profile/profile-after-p47-ep8-<YYYYMMDD>.{md,html}` |
| **G50 (close-out)** | P48 | docs | `develop/perf/proxy_ep8.md` `P48 final` row pinned.  Every `[x]` row in Phase 45..48 of `progress/status.md` has a commit SHA.  Every `progress/p4X/p4X-summary.md` follows R2.1. | `develop/perf/proxy_ep8.md` + `develop/perf/elem_fusion.md` + `progress/status.md` + `progress/p4X/p4X-summary.md` |

## Plan-4 / 5 / 6 ratchet (every plan-7 commit MUST keep these green)

Plan-7 inherits all three ratchets:

- **Plan-4 G23 / G24 / G25 / G26 / G27 / G28 / G29 / G30** (V4 attention + CSA + smoke).
- **Plan-5 G32 / G33a / G33b / G34 / G34a / G34b / G35** (Sinkhorn + SWA + CSA + BWD-split + RoPE-fix).
- **Plan-6 G36 / G36a / G37 / G38 / G39 / G40 / G41 / G42 / G43 / G44 / G45 / G46** (TFLOP/s correction + plan-6 elemwise fusion stack).

Each plan-7 phase opens with a "ratchet check" — `pytest -q
tests/unit_tests/megatron/transformer/deepseek_v4/` (with
`--run-slow` for release-tier gates) plus
`tests/unit_tests/megatron/extensions/` — and the phase row in
`progress/status.md` records the pass count.  Any drop in the
green count blocks the phase commit.

## Banned-warning ratchet

Plan-7 inherits the plan-3 + plan-4 + plan-5 + plan-6 ratchet plus
adds:

- `Adam fused fallback to upstream`  — the patch should NEVER
  print this (means the probe failed); if it does, the phase row
  in `status.md` documents why.

## Perf-budget contract

| Phase | Iter-time drop budget vs prev | TFLOP/s/GPU climb |
| --- | ---: | ---: |
| **P45** | ≥ 100 ms | +28 % |
| **P46** | ≥ 3 ms | +1 % |
| **P47** | ≥ 6 ms | +2 % |
| **P48 (close-out)** | n/a — proxy bake-off pin | n/a |

Any phase whose proxy A/B falls short of the budget ships with
its env default flipped to `"0"` and the regression goes in the
phase's "failed / negative probes" section in
`progress/p4X/p4X-summary.md`.
