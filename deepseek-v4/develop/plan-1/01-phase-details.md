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

1. **Provider strategy**
   - Introduce an explicit provider-selection layer for V4 path:
     `TESpecProvider` vs local provider fallback.
2. **Norm path alignment**
   - Replace standalone `_RMSNorm` usage where feasible with provider-driven norm
     (TE-backed when enabled, local fallback when disabled/unavailable).
3. **Linear path alignment**
   - Migrate eligible `nn.Linear` paths in V4 blocks to provider-selected
     parallel linear modules, preserving tensor-shape contracts.
4. **MoE expert path alignment**
   - Plan and integrate grouped-GEMM capable path through provider strategy
     where semantics match V4 clamped SwiGLU requirements.
5. **Custom attention boundary**
   - Keep new V4 attention kernels/modules custom where needed, while ensuring
     surrounding projection/norm pieces can still benefit from provider reuse.

### Exit Criteria

- A module-level matrix exists and is implemented:
  - `TE reused`,
  - `custom kept`,
  - `fallback local`.
- Runtime can switch TE/local path with stable behavior.
- Performance-critical modules (norm/linear/MoE expert path) have TE-backed
  integration points and fallback documentation.

### Risks / Notes

- TE integration can alter numerical behavior; keep A/B checks mandatory.
- Provider API drift across Megatron variants (Megatron-LM vs Bridge) must be
  guarded with compatibility tests/import checks.

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
