# 02 — Phase 8-11 Test Strategy (v2)

> This strategy defines mandatory checks for architecture refactor, TE reuse,
> distributed MoE convergence, and release readiness.

## Test Principles

- **Same input, same expectation**: fixed seed and deterministic fixtures for
  architectural/numerical comparisons.
- **Progressive confidence**: unit -> integration -> distributed smoke ->
  convergence/performance.
- **A/B validation for risky changes**: especially around TE integration and MoE
  dispatcher migration.
- **Append-only reporting**: every test campaign writes results and conclusion
  to the progress records.

## Test Matrix

| Level | Focus | Phase | Scope | Pass Criteria |
|---|---|---|---|---|
| Unit | Spec construction | P8 | V4 layer/block/mtp spec objects and build path | Spec build succeeds and module ownership matches design doc |
| Unit | Provider parity | P9 | `DeepSeekV4SpecProvider`-driven norm/linear/MoE expert path under TE/local/turbo modes | Provider-selected paths instantiate and run forward/backward under each mode |
| Unit | Router determinism | P10 | Hash and learned router deterministic checks | Fixed seed gives stable route decisions and top-k probabilities |
| Integration | Decoder runtime | P8-P9 | V4 decoder forward/backward on toy model | No crash, no NaN/Inf, expected tensor shapes |
| Integration | MoE distributed path | P10 | EP route + dispatcher end-to-end toy run | No autograd warning regressions on active path |
| Distributed smoke | Single node | P10 | 1x8 (or target local setup) short run | Training reaches target iterations without hang |
| Distributed smoke | Multi-stage setup | P10 | PP/EP combined short run | Stage ownership and token propagation are valid |
| Regression | Full gate run | P11 | Unit + integration + smoke aggregate | All mandatory gates green |
| Numerical | Alignment check | P11 | Selected logits/loss snapshots | Differences within documented tolerance |
| Performance | Throughput A/B | P11 | TE on vs TE off baseline | TE path does not regress mandatory baseline |

## Mandatory Gates

| Gate | Description | Blocking |
|---|---|---|
| **G1** | Spec-driven runtime decoder path is validated | Yes |
| **G2** | TE/local dual path instantiation + runtime checks pass | Yes |
| **G3** | MoE dispatcher/EP path passes no-warning distributed smoke | Yes |
| **G4** | Numerical alignment within tolerance | Yes |
| **G5** | Convergence trend does not regress baseline expectation | Yes |
| **G6** | Throughput/memory check meets minimum release target | Yes |

## Suggested Campaign Templates

### Template A — Architecture refactor check (P8)

- Inputs: tiny V4 config, fixed seed.
- Steps:
  1. Build model through V4 spec path.
  2. Run forward + backward for short iterations.
  3. Confirm no decoder swap dependency in effective runtime path.
- Artifacts:
  - model construction log,
  - shape trace summary,
  - failure notes (if any).

### Template B — TE reuse A/B (P9)

- Inputs: same config/seed, provider modes:
  - `Local` (fallback baseline),
  - `TE` (or `Turbo`) enabled path.
- Steps:
  1. Run identical short training windows under each provider mode.
  2. Compare loss trajectory, runtime errors, and key tensor statistics.
  3. Assert active provider class is `DeepSeekV4SpecProvider`.
  4. Capture active module map (norm/linear/MoE expert path) from runtime logs.
- Artifacts:
  - A/B result table,
  - provider-mode module map snapshot,
  - fallback usage map,
  - regression verdict.

### Template C — Distributed MoE convergence check (P10)

- Inputs: PP/EP test config with hash + learned routing coverage.
- Steps:
  1. Execute single-node distributed smoke.
  2. Execute multi-stage PP/EP smoke.
  3. Validate token-id propagation and routing consistency.
- Artifacts:
  - rank-wise health summary,
  - routing diagnostics summary,
  - blocker list.

### Template D — Release gate run (P11)

- Inputs: selected release config matrix.
- Steps:
  1. Run full matrix in dependency order (unit -> integration -> distributed).
  2. Run numerical alignment checks.
  3. Run convergence and throughput checks.
- Artifacts:
  - gate scoreboard (G1-G6),
  - open risks and owner,
  - go/no-go decision record.

## Triage Rules

- Any mandatory gate failure is a release blocker.
- If TE path fails but local path passes, classify as **performance-path blocker**
  and keep fallback on by default until fixed.
- If numerical alignment and performance goals conflict, numerical correctness
  takes priority for release.
- Any temporary workaround must include:
  - exact scope,
  - rollback condition,
  - owner and target phase.
