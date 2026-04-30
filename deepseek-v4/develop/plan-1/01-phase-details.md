# 01 — Phase 8-11 Detailed Tasks (v2)

> Format: Tasks -> Exit Criteria -> Risks/Notes.
> Progress must be tracked in `deepseek-v4/develop/progress/status.md` (append-only).

---

## Phase 8 (v2) — ModuleSpec Main-Path Refactor

### Tasks

1. **Define V4 runtime spec topology**
   - Build V4-native layer/block spec mapping in
     `primus/backends/megatron/core/models/deepseek_v4/deepseek_v4_layer_specs.py`.
   - Separate:
     - custom V4 modules that stay self-defined (e.g., new attention variants),
     - backend-provider-driven modules (norm/linear/mlp paths).
2. **Stop relying on decoder post-swap as the primary runtime**
   - Refactor `primus/backends/megatron/core/models/deepseek_v4/deepseek_v4_model.py`
     so runtime decoder construction is spec-driven.
   - Keep a temporary guarded compatibility path only if needed.
3. **Builder contract alignment**
   - Ensure `deepseek_v4_builders.py` resolves layer spec and mtp spec in a way
     that is consistent with Megatron `build_module` expectations.
4. **PP/VP/MTP compatibility contract**
   - Document and validate local-layer slicing and stage ownership behavior
     under spec-driven path.
5. **Cleanup and deprecation plan**
   - Mark placeholders and legacy swap path with explicit retirement conditions.

### Exit Criteria

- V4 decoder runtime path is constructed via `transformer_layer_spec` and runs
  without requiring `self.decoder` replacement as the default behavior.
- Layer/block/mtp spec ownership and module mapping are documented and testable.
- No regression in baseline bring-up flow (same minimal smoke path still runs).

### Risks / Notes

- Refactor touches model construction and distributed layer ownership together;
  keep commits small and verify each step with focused smoke checks.
- If temporary compatibility switches are introduced, default behavior and
  migration deadline must be documented.

---

## Phase 9 (v2) — TE Provider Reuse Integration

### Tasks

1. **Create V4 provider class (single entry point)**
   - In `primus/backends/megatron/core/extensions/transformer_engine_spec_provider.py`,
     define `DeepSeekV4SpecProvider(PrimusTurboSpecProvider)`.
   - Keep this class as the only V4-facing provider entry for norm/linear/MoE
     expert spec decisions.
2. **Define provider API contract for V4 spec build**
   - Enumerate methods used by V4 spec construction (norm, linear/projection,
     grouped-MLP/expert path, attention-side projection helpers).
   - Document which methods are inherited unchanged from `PrimusTurboSpecProvider`
     and which are V4-overridden.
3. **Wire V4 specs to provider**
   - Update `deepseek_v4_layer_specs.py` to resolve one provider instance and
     construct layer/block submodules through `DeepSeekV4SpecProvider`.
   - Keep V4-specific attention kernels (`Dense`/`CSA`/`HCA`) and routers
     custom; provider controls surrounding reusable modules.
4. **Migrate norm + projection paths through provider**
   - Replace eligible `_RMSNorm` and projection module selection points with
     provider-driven specs.
   - Preserve dtype, tensor-shape, TP/PP contracts, and deterministic fallback.
5. **Providerize MoE expert compute selection**
   - Route V4 MoE expert module choice through provider grouped-GEMM decision
     path where semantics are compatible.
   - Preserve clamped-SwiGLU behavior and existing hash/learned routing logic.
6. **Runtime mode observability**
   - Add explicit runtime logging/reporting for active provider mode
     (`TE`/`Turbo`/`Local`) used by V4 key module categories.
7. **A/B validation package**
   - Produce repeatable baseline (`Local`) vs provider-enabled (`TE`/`Turbo`)
     runs with module-map snapshot and behavior/perf comparison.

### Exit Criteria

- `DeepSeekV4SpecProvider` exists and is the single V4 provider entry used by
  V4 spec construction.
- V4 runtime `ModuleSpec` generation (`deepseek_v4_layer_specs.py`) is wired to
  provider methods for norm/projection/MoE expert module selection.
- Runtime mode switching (`TE`/`Turbo`/`Local`) is executable on the same model
  config with stable forward/backward behavior.
- V4 MoE expert path has provider-driven grouped-GEMM integration point with
  documented fallback behavior.
- A/B report (baseline vs provider-enabled) is published with module map,
  behavior verdict, and known gaps.

### Risks / Notes

- Provider inheritance can hide behavior drift if inherited methods change
  upstream; pin and verify method-level assumptions for V4-critical paths.
- TE/Turbo availability differs by hardware/backend and TP mode; keep local
  fallback path as safe default until A/B gates pass.
- Primus-Turbo grouped-GEMM / dispatcher dependencies may impose extra runtime
  constraints; do not couple router semantics to provider mode.
- Provider API drift across Megatron variants (Megatron-LM vs Bridge) must be
  guarded with import checks and smoke coverage.

---

## Phase 10 (v2) — MoE and Distributed Path Convergence

### Tasks

1. **Router + dispatcher integration**
   - Move from local per-expert scatter logic to Megatron-compatible dispatcher
     flow while preserving hash-router and learned-router semantics.
2. **EP path cleanup**
   - Replace temporary routed-output all-reduce fallback with dispatcher/EP-safe
     autograd path.
3. **Token id propagation**
   - Define and implement stable token-id propagation rules across PP stages for
     hash-routed layers.
4. **Distributed shape/ownership protocol**
   - Lock tensor shape and ownership expectations across TP/PP/EP boundaries.
5. **Failure observability**
   - Add focused checks/logging around routing density, dropped tokens, and
     dispatcher consistency under distributed runs.

### Exit Criteria

- Single-node and distributed smoke runs complete with stable MoE routing path.
- No known autograd warnings remain on active MoE distributed path.
- Hash and learned routing semantics remain deterministic under fixed seeds.

### Risks / Notes

- Dispatcher migration is high-risk; include deterministic golden-vector checks
  to catch silent route drift.
- PP token-id handling is easy to break; prioritize stage-wise tests early.

---

## Phase 11 (v2) — Validation, Regression, and Release Gates

### Tasks

1. **Regression gate execution**
   - Run unit + integration + distributed smoke matrix defined in
     `02-test-strategy.md`.
2. **Numerical alignment**
   - Compare selected checkpoints/steps against reference expectations with
     fixed seeds and documented tolerances.
3. **Convergence checks**
   - Run short convergence campaigns and compare loss trends with baseline.
4. **Performance checks**
   - Compare TE-on vs TE-off throughput and memory behavior on target setup.
5. **Release checklist**
   - Produce final go/no-go checklist and unresolved-risk register.

### Exit Criteria

- Mandatory tests and quality gates pass with no blocking regressions.
- Remaining risk items are explicitly documented and accepted.
- Release checklist is complete and reproducible.

### Risks / Notes

- Throughput and convergence must both be evaluated; performance-only wins are
  insufficient for release.
- If one metric regresses, issue triage and rollback criteria must be explicit.
