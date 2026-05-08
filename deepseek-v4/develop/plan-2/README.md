# Plan-2 — DeepSeek-V4 in Primus: Architecture-Faithful Rewrite

> Plan-2 supersedes the bring-up oriented plan-0 / plan-1.
>
> The first two plans got us **module shells, a runnable distributed smoke,
> and a provider-driven spec wiring**, but the resulting model is **not a
> faithful DeepSeek-V4** — the attention, MoE gate, activation, and
> checkpoint surface all diverge from the reference. Plan-2 rewrites the
> core modules on top of Megatron's standard
> `spec + config + provider + submodule + build_module` pattern, with
> aggressive reuse of `MLASelfAttention`, `MoELayer`, `TransformerLayer`,
> `TransformerBlock`, `MultiTokenPredictionBlock` and
> `(Yarn)RotaryEmbedding`.
>
> Goal: **architecture parity with the released V4-Flash / V4-Pro
> checkpoints** + **Megatron-native distributed correctness** before any
> further perf / FP8 / convergence campaign.

## Documents

- [`00-review-findings.md`](./00-review-findings.md): code review notes from
  the current branch. Issues are tagged `A*` / `B*` / `C*` / `D*` / `E*` /
  `F*` (architecture, megatron-reuse, distributed, spec hygiene, code
  quality, test) and severity-graded.
- [`01-roadmap.md`](./01-roadmap.md): phase overview + dependency graph +
  milestones for plan-2.
- [`02-target-architecture.md`](./02-target-architecture.md): target module
  layout — class-by-class mapping from V4 reference → Megatron-native modules
  + V4-specific extension points.
- [`03-phase-details.md`](./03-phase-details.md): phase-by-phase tasks, exit
  criteria, and risks.
- [`04-test-strategy.md`](./04-test-strategy.md): regression matrix +
  numerical-alignment plan + checkpoint-load gate.

## Highlights

- **Replace the V4 attention with `MLASelfAttention`** + a thin V4 subclass
  that adds `q_norm`, `kv_norm`, grouped low-rank O-projection, learnable
  attention sink, and the optional `Compressor`/`Indexer` branches as
  spec submodules.
- **Replace the V4 MoE with `MoELayer`** + V4-specific `Router`
  (`HashRouter` with a *learnable* `gate.weight` + `tid2eid` lookup,
  matching HF / NeMo) and a clamped-SwiGLU activation function selected via
  `DeepSeekV4SpecProvider`.
- **Replace the standalone block** with a `DeepseekV4TransformerBlock` that
  subclasses `TransformerBlock`, and a `DeepseekV4HybridLayer` that
  subclasses `TransformerLayer` and overrides residual mixing with the HC
  `HyperMixer`.
- **Wire MTP through Megatron's `MultiTokenPredictionBlock`** and retire
  the standalone `DeepseekV4MTPBlock`.
- **Fix HC × PP**: lift hidden shape to `[S, B, K, D]` across PP boundaries
  (or use serialize/deserialize helpers); apply `HyperHead` only on the
  last PP stage.
- **Fix TP**: switch projection specs from `parallel_mode="duplicated"` to
  `column_parallel` / `row_parallel` so QKV/O are sharded.
- **Pre-training first**: Plan-2 ships from-scratch pre-training. The
  HF state-dict adapter + V4-Flash checkpoint load (token-0 logits
  ≤1e-2 vs HF) is **deferred to P22+** — see
  [`03-phase-details.md`](./03-phase-details.md#p22--hf-state-dict-adapter--v4-flash-checkpoint-load-deferred-follow-up)
  and `02-target-architecture.md` §7. Activate that follow-up when an
  SFT / evaluation campaign needs the HF weights.
- **Numerical alignment within Primus is still a gate**: per-module
  forward agreement with inline HF references (G2 / G3 / G4 / G5) on
  CPU 4L toys is mandatory; the full V4-Flash safetensors numerical
  gate (G8 / G9) moves with the deferred adapter to P22+.
