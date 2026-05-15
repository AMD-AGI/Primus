# 03 — Plan-8 Test Strategy

> Plan-8 reuses the test conventions from
> `../plan-4/03-test-strategy.md` (the canonical attention-kernel
> precedent) plus `../plan-7/03-test-strategy.md` (microbench-
> driven descope methodology).  Plan-8 is **microbench- +
> proxy-A/B-driven**: every kernel ships behind an env knob that
> defaults OFF until both gates fire.  The plan-5 P32 RoPE bug +
> plan-7 P45 budget-mis-calibration are the load-bearing
> precedents — the proxy A/B is the source of truth, not the
> microbench.

Correctness gates (plan-4 G23 / G24 / G26 / G27 / G29 plus plan-5
G32 / G34 / G34b / G35 plus plan-6 G36..G46 plus plan-7 G47..G49)
ratchet — every plan-8 phase MUST keep them green.  Plan-8 adds
G50..G56.

## Gate matrix

| Gate | Phase | Type | What it checks | Where it lives |
| --- | --- | --- | --- | --- |
| **G49** | P49 | runtime (Python) | (a) `_tilelang.is_tilelang_path_enabled()` returns False when `PRIMUS_V4_TILELANG_ATTN` is unset / `"0"`; True when `"1"`. (b) Importing `primus.backends.megatron.core.transformer.v4_attention_kernels._tilelang` does NOT run any tilelang JIT (lazy). (c) Setting `PRIMUS_V4_TILELANG_ATTN=1` with no P50..P55 kernels landed yet routes through Triton with a single rank-0 warning (banned-warning ratchet exempt). (d) Plan-4..plan-7 gate ratchet is bit-for-bit unchanged at default-off. | `tests/unit_tests/megatron/transformer/deepseek_v4/test_p49_tilelang_dispatch.py` |
| **G50** | P50 | runtime (GPU) | FWD `out` + `lse` parity vs the plan-4 G23 eager reference (`reference.py::eager_v4_attention`) within bf16 `atol=2e-3 rtol=2e-3` at fast tier (`B=2, H=4, Sq=Sk=64, head_dim=64`) and release tier (`B=1, H=64, Sq=Sk=4096, head_dim=512, bf16, pytest.mark.slow`).  Parametrise `(MQA / MHA) × (sink / no_sink) × (SWA / full) × (additive_mask / no_mask)` = 16 combinations.  Determinism check at `attn_dropout=0.0` (G50d). | `tests/unit_tests/megatron/transformer/deepseek_v4/test_p50_v4_attention_fwd_tilelang.py` |
| **G50a / G50b** | P50 | smoke + perf | EP=8 10-iter proxy smoke with `PRIMUS_V4_TILELANG_ATTN=1` (cr=0 only); plan-4..plan-7 ratchet stays green; lm_loss within `5e-2` of P48 baseline.  Microbench `progress/p50/bench_v4_attention_fwd_tilelang.py` reports FWD median ms + TFLOP/s vs the plan-4 P25 Triton kernel; target ≥ 1.2× speedup at V4-Flash widths.  **If microbench loses, P50 still ships behind the env knob default-off and the regression is documented in `p50-summary.md`** (microbench wins are required for default-on, not for landing). | `progress/p50/run_smoke_p50_dense_ep8.sh` + `progress/p50/bench/{v4_flash, smoke}.json` |
| **G51** | P51 | runtime (GPU) | BWD `dQ` / `dK` / `dV` / `dSink` parity vs the plan-4 G24 eager reference within bf16 `atol=5e-3 rtol=5e-3` at fast tier + release-tier slow.  `torch.autograd.gradcheck` fast tier fp32 small shape.  Parametrise (MQA / MHA / sink / no_sink / SWA / full / additive_mask / no_mask) = 16 combinations. | `tests/unit_tests/megatron/transformer/deepseek_v4/test_p51_v4_attention_bwd_tilelang.py` |
| **G51a / G51b** | P51 | smoke + perf | EP=8 10-iter proxy smoke (cr=0 fully on tilelang now — FWD + BWD); microbench reports BWD median ms + TFLOP/s vs the plan-5 P32 final Triton split BWD; target ≥ 1.2× speedup at V4-Flash widths. | `progress/p51/run_smoke_p51_dense_ep8.sh` + `progress/p51/bench/{v4_flash, smoke}.json` |
| **G52** | P52 | runtime (GPU) | FWD `out` + `lse` parity vs the plan-4 G23 eager reference HCA path within bf16 `atol=2e-3 rtol=2e-3`.  Parametrise `hca_local_seqlen ∈ {0, 4096}` × `(sink / no_sink)` × `(SWA window=128, full)` plus the pool-only additive mask path.  Fast + release-tier slow. | `tests/unit_tests/megatron/transformer/deepseek_v4/test_p52_v4_hca_fwd_tilelang.py` |
| **G52a / G52b** | P52 | smoke + perf | EP=8 10-iter proxy smoke with cr=0 + cr=128 on tilelang; CSA still on Triton.  Microbench ≥ 1.15× FWD speedup vs the plan-5 P32 final Triton HCA split-mask FWD. | `progress/p52/run_smoke_p52_hca_ep8.sh` + `progress/p52/bench/{v4_flash, smoke}.json` |
| **G53** | P53 | runtime (GPU) | BWD parity vs the plan-4 G24 eager reference HCA path within bf16 `atol=5e-3 rtol=5e-3`.  `gradcheck` fast tier fp32.  Parametrise as G52. | `tests/unit_tests/megatron/transformer/deepseek_v4/test_p53_v4_hca_bwd_tilelang.py` |
| **G53a / G53b** | P53 | smoke + perf | EP=8 10-iter proxy smoke (cr=0 + cr=128 fully on tilelang).  Microbench ≥ 1.15× BWD speedup vs the plan-5 P32 final Triton HCA split-mask BWD. | `progress/p53/run_smoke_p53_hca_ep8.sh` + `progress/p53/bench/{v4_flash, smoke}.json` |
| **G54** | P54 | runtime (GPU) | FWD `out` + `lse` parity vs the plan-4 G26 eager reference CSA path within bf16 `atol=2e-3 rtol=2e-3`.  Parametrise `K_topk ∈ {0, 64, 512}` × `(sink / no_sink)` × `(SWA window=128, full)` × `(pre_gathered / from_pool)` = 24 combinations.  Fast + release-tier slow. | `tests/unit_tests/megatron/transformer/deepseek_v4/test_p54_v4_csa_fwd_tilelang.py` |
| **G54a / G54b** | P54 | smoke + perf | EP=8 10-iter proxy smoke (cr=0 + cr=128 + cr=4 all on tilelang).  Microbench ≥ 1.2× FWD speedup vs the plan-5 P32 final Triton CSA sparse FWD. | `progress/p54/run_smoke_p54_csa_ep8.sh` + `progress/p54/bench/{v4_flash, smoke}.json` |
| **G55** | P55 | runtime (GPU) | BWD parity vs the plan-4 G27 eager reference CSA path within bf16 `atol=5e-3 rtol=5e-3`.  `gradcheck` fast tier fp32.  Parametrise as G54.  Bonus: parametrise `PRIMUS_V4_TILELANG_CSA_BWD_SEGREDUCE ∈ {0, 1}` — both BWD pool variants must pass parity. | `tests/unit_tests/megatron/transformer/deepseek_v4/test_p55_v4_csa_bwd_tilelang.py` |
| **G55a / G55b** | P55 | smoke + perf | EP=8 10-iter proxy smoke (all three families on tilelang, FWD + BWD).  Microbench ≥ 1.2× BWD speedup vs the plan-5 P32 final Triton CSA gather + atomic BWD. | `progress/p55/run_smoke_p55_csa_ep8.sh` + `progress/p55/bench/{v4_flash, smoke}.json` |
| **G56** | P56 | docs | `develop/perf/attention_perf.md` has one row per plan-8 kernel × FWD / BWD.  `develop/perf/proxy_ep8.md` `P56 final` row pinned to the 15-iter clean bake-off.  Every `[x]` row in Phase 49..56 of `progress/status.md` has a commit SHA + date.  Every `p4X-summary.md` follows R2.1.  `PRIMUS_V4_TILELANG_ATTN` default decided by the bake-off (`"1"` if ≥ 30 ms / iter saved vs P48, else `"0"`). | `develop/perf/attention_perf.md` + `develop/perf/proxy_ep8.md` + `progress/status.md` + `progress/p4X/p4X-summary.md` |

## Plan-4 / 5 / 6 / 7 ratchet (every plan-8 commit MUST keep these green)

Plan-8 inherits all four ratchets:

- **Plan-4 G23 / G24 / G25 / G26 / G27 / G28 / G29 / G30** (V4
  attention + CSA + smoke).
- **Plan-5 G32 / G33a / G33b / G34 / G34a / G34b / G35** (Sinkhorn +
  SWA + CSA + BWD-split + RoPE-fix).
- **Plan-6 G36 / G36a / G37 / G38 / G39 / G40 / G41 / G42 / G43 /
  G44 / G45 / G46** (TFLOP/s correction + plan-6 elemwise fusion).
- **Plan-7 G47 / G48 / G49** (Adam prototype + grad-scale +
  grad-norm parity).

Each plan-8 phase opens with a "ratchet check" — `pytest -q
tests/unit_tests/megatron/transformer/deepseek_v4/` (with
`--run-slow` for release-tier gates) plus
`tests/unit_tests/megatron/extensions/` — and the phase row in
`progress/status.md` records the pass count.  Any drop in the
green count blocks the phase commit.

## Banned-warning ratchet

Plan-8 inherits the plan-3 / 4 / 5 / 6 / 7 ratchet plus adds:

- `tilelang JIT cache miss — falling back to interpreter` — the
  tilelang import should NEVER print this in production; cache
  cold-start runs `progress/p49/build_tilelang_kernels.sh` first.
- `v4 attention fallback to triton — tilelang kernel raised` —
  the dispatcher should NEVER print this when
  `PRIMUS_V4_TILELANG_ATTN=1` and the kernel parity test has
  passed.  If it does, the env flag is flipped back to `"0"`
  immediately.

## Perf-budget contract

| Phase | Microbench speedup vs Triton | Proxy iter-time drop vs prev phase | TFLOP/s/GPU climb |
| --- | ---: | ---: | ---: |
| **P50** | ≥ 1.2× FWD       | ≥ 0 ms (cr=0 only)         | ≥ 0 %  |
| **P51** | ≥ 1.2× BWD       | ≥ 5 ms  (cr=0 fully)       | ≥ +1 % |
| **P52** | ≥ 1.15× FWD      | ≥ 3 ms  (cr=128 added)     | ≥ +1 % |
| **P53** | ≥ 1.15× BWD      | ≥ 3 ms  (cr=128 fully)     | ≥ +1 % |
| **P54** | ≥ 1.2× FWD       | ≥ 5 ms  (cr=4 added)       | ≥ +1 % |
| **P55** | ≥ 1.2× BWD       | ≥ 10 ms (cr=4 fully)       | ≥ +2 % |
| **P56**  | n/a — bake-off pin | cumulative ≥ 30 ms (preferred) | n/a |

Any phase whose proxy A/B falls short of the budget ships with
its env default flipped to `"0"` and the regression goes in the
phase's "failed / negative probes" section in
`progress/p4X/p4X-summary.md`.

Bonus: any phase whose **microbench wins** but **proxy A/B
regresses** ships behind the env knob default-off + a forensic
attribution analysis (R9.3) in the phase summary — same precedent
as plan-7 P45 ("`torch._foreach_add_` is already at HBM peak,
budget mis-calibrated").
