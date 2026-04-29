# 00 — DeepSeek-V4 Phase 8+ Roadmap (v2)

> This roadmap starts after the first bring-up line (Phase 1-7).
> It targets architecture convergence (`ModuleSpec`) and runtime performance
> convergence (`TESpecProvider` reuse) without changing `third_party/`.

## Objectives

| Objective | What success looks like |
|---|---|
| Restore spec-driven architecture | V4 decoder is built from V4 `ModuleSpec` chain, not from post-init decoder swap |
| Reuse TE runtime path where applicable | Norm/linear/MoE expert path can select TE-backed modules by provider |
| Align with Megatron MoE parallel patterns | Dispatcher/EP integration replaces temporary routed-output all-reduce pattern |
| Raise release confidence | Functional, numerical, distributed, and performance gates are explicitly tracked |

## Phase Overview

| # | Phase | Type | Key Deliverables | Exit Criteria |
|---|---|---|---|---|
| **8** | ModuleSpec main-path refactor | architecture | V4 layer/block/mtp spec topology; model init no longer depends on decoder swap for runtime | `transformer_layer_spec` directly builds and runs V4 decoder path |
| **9** | TE provider reuse integration | performance architecture | V4 provider selection matrix (`TE`/`Local`), TE-backed RMSNorm/parallel linear/grouped GEMM adoption plan | Runtime module map clearly shows TE-reused vs custom components with fallback |
| **10** | MoE + distributed convergence | distributed integration | Hash/learned router dispatch integration with Megatron dispatcher + EP semantics | 1x8 and PP/EP smoke run with no autograd warning regressions in MoE path |
| **11** | Validation + release gates | quality / release | Regression matrix, convergence checks, throughput comparison, release checklist | All mandatory gates pass and blockers are documented |

## Dependency Graph

```mermaid
flowchart TD
phase8Spec[Phase8_ModuleSpecRefactor]
phase9TE[Phase9_TEProviderReuse]
phase10MoE[Phase10_MoEDistributedConverge]
phase11QA[Phase11_ValidationRelease]

phase8Spec --> phase9TE
phase9TE --> phase10MoE
phase10MoE --> phase11QA
```

## Milestones

| Milestone | Scope | Phases |
|---|---|---|
| **R0: Replan locked** | `plan-1` documents + `status.md` Phase 8+ tracking section in place | bootstrap |
| **R1: Spec architecture aligned** | V4 runtime decoder path is spec-driven | P8 |
| **R2: TE baseline aligned** | TE-backed modules integrated with controlled fallback | P9 |
| **R3: Distributed MoE aligned** | EP/dispatcher path converged and stable in smoke | P10 |
| **R4: Release ready** | Regression + convergence + perf gates pass | P11 |

## Constraints

- Primus extension rule remains unchanged: all implementation under `primus/`, no
  direct modifications in `third_party/`.
- Existing `develop/progress/status.md` history is append-only.
- Any temporary compatibility path added during P8-P10 must include a retirement
  condition and owner in documentation.

## Top Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Spec refactor intersects PP/VP behavior | Runtime mismatch and hard-to-debug stage failures | Define shape and ownership contract first; land with targeted integration checks |
| TE integration changes numerics/perf unexpectedly | Regression in convergence or stability | Keep per-module fallback toggles and perform A/B validation by test matrix |
| MoE dispatcher migration breaks hash-router assumptions | Incorrect routing or silent accuracy drift | Freeze deterministic routing test vectors and compare before/after outputs |
| Multiple backends (Megatron-LM vs Bridge compatibility) | Signature/runtime drift | Document supported runtime baseline and add import-path sanity checks |
