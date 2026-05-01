# Plan-1 As-Built Notes + Plan-2 Pointer

> **Status:** historical / advisory.
> **Active plan of record:** [`deepseek-v4/develop/plan-2/`](../plan-2/README.md) (architecture-faithful rewrite).
> **Active progress tracker:** [`deepseek-v4/develop/progress/status.md`](../progress/status.md) — Phase 12+ section.
> **Author:** code review snapshot, `dev/wenx/deepseek-v4` `e194e039..HEAD`, 2026-05-01.

This post closes out the plan-0 / plan-1 work program and points the next phase
of work at plan-2. It is written as an **as-built note**, not a project plan:
it describes what landed on the branch, what we expected to land that did not,
and which architectural decisions were rolled back by the plan-2 review.

If you only have one minute, jump to:

- [TL;DR](#tldr)
- [What plan-1 actually shipped](#what-plan-1-actually-shipped)
- [Where plan-1 fell short of "real DeepSeek-V4"](#where-plan-1-fell-short-of-real-deepseek-v4)
- [Pointer to plan-2](#pointer-to-plan-2)

---

## TL;DR

- Plan-0 (P0–P7) shipped a runnable single-node smoke: configs, trainer
  dispatch, layer specs, HC + Hybrid Attention scaffolding, MoE / activation /
  RoPE / MTP, and a `1×8 PP=2 EP=4` BF16 smoke that reaches
  `iteration 3` cleanly.
- Plan-1 (P8–P10) delivered the **plumbing** — a spec-driven runtime
  (`DeepseekV4Model` on `LanguageModule`), a single V4 provider class
  (`DeepSeekV4SpecProvider(PrimusTurboSpecProvider)`), and a Megatron-dispatcher
  MoE bridge with EP all-reduce fallback gated behind a debug toggle.
- Plan-1 P11 (validation + release gates) **did not start**: the architecture
  review at the end of plan-1 concluded that the current modules are not yet a
  faithful DeepSeek-V4, so a regression / convergence campaign on the current
  modules would only certify the wrong model.
- Plan-2 is an **architecture-faithful rewrite** of the V4 modules on top of
  Megatron's standard `spec + config + provider + submodule + build_module`
  pattern — see [plan-2/README.md](../plan-2/README.md).

---

## What plan-1 actually shipped

Cross-reference this section against
[`progress/status.md` Phase 8 (v2) – Phase 11 (v2)](../progress/status.md).

### P8 (v2) — ModuleSpec main-path refactor — **DONE**

Landed pieces:

- A full V4 layer / decoder / MTP `ModuleSpec` topology rooted at
  `get_deepseek_v4_runtime_decoder_spec` in
  `primus/backends/megatron/core/models/deepseek_v4/deepseek_v4_layer_specs.py`.
- `DeepseekV4Model` now subclasses `LanguageModule` and builds the decoder
  directly from the runtime spec via `build_module`. The legacy GPT-placeholder
  super-init swap is retired as the **default** path.
- Builder (`deepseek_v4_builders.py`) resolves and forwards the V4 runtime
  decoder spec only — it no longer constructs a GPT layer spec just to make
  `super().__init__` happy.
- Runtime instantiate / forward validated in container `dev_primus_wenx_691`
  with the expected decoder + attention topology.

### P9 (v2) — TE / Turbo provider reuse — **DONE**

Landed pieces:

- `DeepSeekV4SpecProvider(PrimusTurboSpecProvider)` introduced in
  `core/extensions/transformer_engine_spec_provider.py`. It is the **single V4
  provider entry** for norm + linear + grouped-MLP module decisions.
- `deepseek_v4_layer_specs.py` resolves one provider instance, threads its mode
  through block / layer params, and routes norm + MoE spec construction through
  the provider.
- Dense-MLP projections route through `provider.linear()` in `duplicated` mode
  (TE / Turbo) with a local fallback when provider modules are unavailable.
  Attention projections use `DeepseekV4AttentionSubmodules + build_module`.
- V4 MoE (`v4_moe.py`) has provider grouped-MLP instantiation and a grouped
  forward dispatch (expert bucketing → grouped forward → scatter-add) with a
  local-expert fallback.
- A/B validation report landed at
  [`plan-1/03-phase9-provider-ab-report.md`](../plan-1/03-phase9-provider-ab-report.md):
  local forward passes, TE module-map build + CUDA forward both pass, and the
  TE / Turbo host-input path has an explicit CUDA guard in the decoder forward.

### P10 (v2) — MoE + distributed path convergence — **PARTIAL**

Landed pieces:

- `DeepseekV4MoESubmodules` defined and wired through the FFN spec in
  `deepseek_v4_layer_specs.py`. Router / dispatcher / experts / shared-experts
  are instantiated through `build_module`.
- Dispatcher bridge: V4 MoE forward now follows the Megatron flow
  `dispatch_preprocess → token_dispatch → dispatch_postprocess → expert_compute
  → combine`, with runtime dispatcher selection and a local fallback.
- Routed-output `all_reduce` fallback retired from the **active** EP path. It
  remains gated behind the explicit `v4_enable_ep_allreduce_fallback` debug
  toggle until plan-2 P21 cleanup.
- Clamped-SwiGLU backend compatibility checks landed: a grouped backend that
  cannot declare `supports_clamped_swiglu` (or override via config) is
  downgraded to local experts with a warning.

Did **not** land in plan-1:

- A formal **PP token-id propagation contract** for hash-routed layers
  (explicit stage ownership, transport rules, fail-fast assertions). The
  in-tree solution is still the `decoder._v4_token_ids` attribute stash, which
  plan-2 P14 / P15 replaces with a forward-kwarg.
- A distributed smoke run with **deterministic routing snapshots** comparing
  before / after dispatcher migration. The 1×8 PP=2 EP=4 functional smoke
  passes (reaches `iteration 3`), but a reproducible routing-vector diff is
  missing.

### P11 (v2) — Validation + release gates — **NOT STARTED**

Reason: the plan-2 architecture review (recorded in
[`plan-2/00-review-findings.md`](../plan-2/00-review-findings.md)) concluded
that the current modules diverge from real DeepSeek-V4 in five
**architecture-faithfulness** dimensions and ten **Megatron-reuse / spec**
dimensions. Running the regression matrix and convergence campaign on the
current modules would have certified the wrong model. P11 is **paused** and
its scope is rolled into plan-2 P19 (distributed re-validation) and plan-2 P20
(numerical / convergence / perf gates).

---

## Where plan-1 fell short of "real DeepSeek-V4"

Plan-1 succeeded at the **plumbing** dimension — spec topology, provider
class, dispatcher bridge — but did not converge on the **module-level
correctness** dimension. The five-line summary of the gap:

| # | Symptom on the branch | Real V4 | Severity |
|---|---|---|---|
| **A1** | Attention has separate `linear_k_proj` / `linear_v_proj`. | Single `wkv` projection; `K = V = kv` (single-latent MQA). | **CRIT** — wastes parameters and blocks checkpoint loading. |
| **A2** | No per-head `q_norm` / `kv_norm` after the low-rank Q / KV projections. | Per-head RMSNorm on the head dimension after `wq_b` and `wkv_b`. | **CRIT** — numerical drift vs reference. |
| **A3** | `HashRouter` has no learnable gate weight; outputs uniform `1/topk` weights. | Hash gate has a learnable `gate_linear` that produces real scores; the hash table only decides expert ids. | **CRIT** — model cannot match the trained checkpoint. |
| **A4** | `clamped_swiglu` clamps the **post-multiplication** output. | Clamping is on the **pre-multiplication** legs: `silu(gate)` clamped to `(-α, +α)` and `up` clamped to `(-α, +α)`, **then** multiplied. | **CRIT** — different gradient, different numerics. |
| **A5** | No state-dict adapter. | Required to load `DeepSeek-V4-Flash` / `V4-Pro` HF safetensors. | **CRIT** — official checkpoints cannot be loaded today. |

And — equally important — the Megatron-LM reuse contract was not followed:

| # | Symptom | What it should be | Severity |
|---|---|---|---|
| **B1** | `DeepseekV4Attention` is a plain `nn.Module`. | Subclass of `MLASelfAttention` (V4 == MLA + extras). | **HIGH** |
| **B2** | `DeepseekV4TransformerBlock` does not subclass `TransformerBlock`. | It should — to pick up upstream PP / recompute / SP behavior. | **HIGH** |
| **B3** | `DeepseekV4HybridLayer` does not subclass `TransformerLayer`. | It should — HC residual is an override, not a re-implementation. | **HIGH** |
| **B4** | `DeepseekV4MoE` does not subclass `MoELayer`. | It should — load-balance / z-loss / dispatcher lifecycle for free. | **HIGH** |
| **B5** | MTP bypasses `MultiTokenPredictionBlock`. | Should reuse the upstream PP-aware MTP block. | **HIGH** |

A more complete list (28 findings, severity-ranked CRIT / HIGH / MED / LOW)
lives in [`plan-2/00-review-findings.md`](../plan-2/00-review-findings.md).

### Distributed correctness footnote

Three distributed-correctness items were also flagged by the review and
deferred to plan-2:

- **C1** Hyper-Connections × Pipeline Parallelism: `HyperHead` is currently
  applied at every PP stage, which destroys the K-stream context across the
  PP boundary. Plan-2 P15 fixes this by stream-packing `[S, B, K, D]`
  → `[S*K, B, D]` for PP send / recv and only applying `HyperHead` on the
  `post_process` stage.
- **C2** `decoder._v4_token_ids` attribute stash leaks state across PP and
  microbatches. Plan-2 P14 / P15 thread `token_ids` as a forward kwarg.
- **C3** Position IDs are faked internally; caller-supplied `position_ids`
  are ignored. Plan-2 P15 honors them.

---

## Pointer to plan-2

The active plan of record is now plan-2. It is **not** an iteration on
plan-1 — it rewrites the modules where plan-1 diverged from real V4 or from
Megatron's `spec + config + provider + submodule + build_module` pattern.

Documents:

- [`plan-2/README.md`](../plan-2/README.md) — top-level index
- [`plan-2/00-review-findings.md`](../plan-2/00-review-findings.md) — full
  severity-ranked review
- [`plan-2/01-roadmap.md`](../plan-2/01-roadmap.md) — phases P12 → P21,
  dependency graph, milestones, top risks
- [`plan-2/02-target-architecture.md`](../plan-2/02-target-architecture.md) —
  module-by-module rewrite map
- [`plan-2/03-phase-details.md`](../plan-2/03-phase-details.md) — granular
  tasks / exit criteria / risks
- [`plan-2/04-test-strategy.md`](../plan-2/04-test-strategy.md) — L0 → L3
  test pyramid + release gates G1 → G13

Schedule:

- **Block A (landed):** Apr 28 → May 01 — plan-0 P0–P7, plan-1 P8–P10,
  plan-2 P12 (lockdown + review).
- **Holiday:** May 02 → May 05 — no engineering activity.
- **Block B (planned):** May 06 → May 09 — plan-2 P13 → P21 over four
  intensive working days. See
  [`progress/timeline.html`](../progress/timeline.html) for the
  day-by-day Gantt and
  [`progress/deepseek_v4_roadmap.pptx`](../progress/deepseek_v4_roadmap.pptx)
  slide 8 ("开发计划 · development schedule") for the slide format.

Guiding principles for plan-2:

- **Reuse > reinvent.** Subclass `MLASelfAttention`, `TransformerLayer`,
  `TransformerBlock`, `MoELayer`, `TopKRouter`,
  `MultiTokenPredictionBlock`, and `(Yarn)RotaryEmbedding`.
- **Spec > monolith.** Express V4 extensions (Compressor / Indexer /
  DualRoPE / HyperMixer / AttentionSink) as `ModuleSpec` submodules so the
  provider system can swap implementations.
- **Parity > perf.** Architecture parity with the released
  V4-Flash / V4-Pro checkpoints **before** any FP8 / Turbo / convergence
  campaign.
- **Tests gate phases.** Every phase from P13 onward contributes tests at
  the right level (L0 unit / L1 module integration / L2 distributed smoke
  / L3 release gate). Nothing ships without G1 → G13 green.

---

## Cross-references

- Architecture deep dive: [`01-deepseek-v4-architecture-deep-dive.md`](01-deepseek-v4-architecture-deep-dive.md)
- Plan-0 (paused): [`develop/plan-0/`](../plan-0/)
- Plan-1 (paused): [`develop/plan-1/`](../plan-1/)
- Plan-2 (active): [`develop/plan-2/`](../plan-2/)
- Progress tracker: [`develop/progress/status.md`](../progress/status.md)
- HTML timeline: [`develop/progress/timeline.html`](../progress/timeline.html)
- PPT roadmap: [`develop/progress/deepseek_v4_roadmap.pptx`](../progress/deepseek_v4_roadmap.pptx)
