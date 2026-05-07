# Worker: Correctness-Lite

**Status**: Stub  
**Extends**: `_envelope.md`

Goal: spot-check a Plan's loss curve / grad norm against the active Reference (T1, see §S2). Produce `{pass | drift, delta_pct}`.

## Skill scope

- `skills/workflow/correctness.md`

## Tools

- `observe.compare_loss(run_id, reference_curve)`

## Output

`summary.headline`: "lite-correctness PASS, max delta_pct=0.42% (tol=3σ window)" or "DRIFT at tokens=2.3e7, delta=2.1% > tol".

`status=failed` with `failure.kind=NUMERICAL` if drift detected → Orchestrator goes to ABORT (per §12.2).
