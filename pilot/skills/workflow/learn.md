# Learn (Knowledge Precipitation)

**Status**: v1
**Stage**: post-`REPORT`, before session close
**Tool boundary**: `pilot.tools.learn analyze` (read-only over session state)
                 + `pilot.tools.knowledge write` (writes drafts only)
**Authoritative schemas**: `schemas/knowledge_draft.schema.json`

This file is the **single source of truth** for how a finished tuning
session is mined for cross-session findings, and how those findings flow
into the human review pipeline. The contract is intentionally narrow:
the LLM never edits `skills/` or `pilot/tools/` directly — it writes
*drafts* to `state/knowledge_drafts/`, which a human curator promotes
via git PR.

---

## 1. Purpose

Close the last link in the autonomy chain: every session must improve
the next. Concretely, after `REPORT`, `learn analyze` reads the session
artifacts and surfaces four classes of finding that the engine could
have used during the run but didn't (or shouldn't have, but did):

| Finding class | What it captures | Resulting draft kind |
|---|---|---|
| `catalog_gaps` | axes the session moved (or tried) but `_axis_translator` returns `None` for | `final_best_case` (champion-promoting) / `env_recipe` (env-only) / `failure_pattern` (tried but lost) |
| `constraint_gaps` | failure messages matching a known mutex shape (`axis_taxonomy.md §2.14`) that `constraint.check` didn't pre-empt | `failure_pattern` |
| `calibration_drifts` | engine-predicted `expected_gain_band_pct` vs measured gain differs by more than 2× | `model_calibration_drift` |
| `anti_pattern_signals` | env / trainer values that consistently regressed by ≥5% across rounds | `failure_pattern` (with DANGER note) |

Each finding maps to one of the four existing `KnowledgeDraft` `kind`s,
so this stage **does not change** `knowledge_draft.schema.json`.

---

## 2. Inputs

| Input | Source | Required |
|---|---|---|
| `tuning_state.yaml` | `state/<session>/tuning_state.yaml` | yes |
| `r*_exec_results.json` | `state/<session>/r*_exec_results.json` | optional (used for failure-pattern fingerprints) |
| `r*_snapshots.json` | `state/<session>/r*_snapshots.json` | optional (used for measured gains) |
| `run_history` | inline in `tuning_state.yaml` OR sidecar | optional but strongly recommended for catalog/anti-pattern detection |
| `DiagnosisReport` YAMLs | engine emission per round | optional (powers `calibration_drifts`) |

`learn analyze` is **best-effort**: it gracefully degrades when the
optional inputs are absent. On the canonical path (orchestrator stores
`run_history` in `tuning_state.yaml`), all four finding classes fire;
on the fallback path (only `r*_exec_results.json` available), only
`constraint_gaps` and headline-derived signals are produced.

---

## 3. Output (`LearnAnalysis`)

```yaml
schema_version: "1.0"
session_id:    <id>
generated_at:  <ISO-8601 UTC>
inputs:
  tuning_state_ref:           <path>
  trial_count:                <int>
  snapshot_count:             <int>
  stage_history_len:          <int>
  run_history_len:            <int>
  diagnosis_reports_provided: <bool>
findings:
  catalog_gaps:        [CatalogGap, ...]
  constraint_gaps:     [ConstraintGap, ...]
  calibration_drifts:  [CalibrationDrift, ...]
  anti_pattern_signals:[AntiPatternSignal, ...]
suggested_drafts:      [<draft payload>, ...]   # one per finding
```

Persisted to `state/learn/<session_id>/learn_analysis.yaml` when
`--write-analysis` is passed.

---

## 4. Mutex fingerprint table

`learn analyze` ships with a fixed table of regex fingerprints that
match the failure messages of every `axis_taxonomy.md §2.14` rule. The
table currently covers:

| `rule_id` | Match shape (regex, case-insensitive) |
|---|---|
| `REQ-PP-DEFER-EMB` | `defer.*embedding.*wgrad.*pipeline.*not used` |
| `REQ-PP-OVRLP-P2P` | `interleaved pipeline parallelism` |
| `MUTEX-CG-IMPL` | `--enable-cuda-graph.*--cuda-graph-impl` (either order) |
| `MUTEX-DEEPEP-ROUTER` | `DeepEP.*float32.*probs` |
| `KNOWN-BLOCKER-CG-ENUM` | `requires string as left operand, not CudaGraphScope` |
| `KNOWN-BLOCKER-CG-HIP` | `HIP error: invalid argument` |
| `MUTEX-PROFILE-HIPBLASLT` | `PRIMUS_HIPBLASLT_TUNING.*profile` |

Adding a new fingerprint is a one-row append in
`pilot/tools/learn.py::_MUTEX_FINGERPRINTS` plus a row in
`axis_taxonomy.md §2.14`. The two MUST be kept in lockstep.

---

## 5. Anti-pattern detection (§S4.2)

`learn analyze` does NOT itself enforce the §S4.2 rejection rules
(empty / over-broad binding, contradiction with anti-patterns,
over-200-char headline). Those live in `pilot.tools.knowledge.write`,
which `learn emit_drafts` calls last. So:

```
learn.analyze   →  finding (in-memory)
learn.emit_drafts → knowledge.write  →  draft.yaml under state/knowledge_drafts/
                       │
                       └── auto-rejects on §S4.2 violation; sets accepted=false
```

A rejected draft still lands on disk (with `accepted: false` and a
populated `reasons` array) so the curator can see why.

---

## 6. Promotion path (out of scope here)

```
state/knowledge_drafts/<draft_id>.yaml
        │
        └── curator review (manual)
                │
                └── git PR into:
                       skills/workflow/axis_taxonomy.md  (catalog_gaps)
                       pilot/tools/_axis_translator.py   (catalog_gaps)
                       pilot/tools/constraint.py         (constraint_gaps)
                       skills/optimization/<bottleneck>/ (failure_pattern, env_recipe)
```

This file does NOT specify the curator workflow — that's a human
process. The Pilot side guarantees only that drafts land in a
predictable shape.

---

## 7. CLI summary

```
python -m pilot.tools.learn analyze \
    --session <session_dir> \
    [--diagnosis-glob 'state/<session>/diagnosis/r*.yaml'] \
    [--write-analysis] \
    [--emit-drafts] \
    [--drafts-root state/knowledge_drafts]
```

Two-step flow (recommended for first invocation):

```bash
python -m pilot.tools.learn analyze --session <dir> --write-analysis
# review state/learn/<session>/learn_analysis.yaml
python -m pilot.tools.learn analyze --session <dir> --emit-drafts
```

Single-step flow (CI):

```bash
python -m pilot.tools.learn analyze --session <dir> --write-analysis --emit-drafts
```

---

## 8. Cross-references

- Catalog patches → `axis_taxonomy.md §2`, `_axis_translator.py`.
- Constraint patches → `axis_taxonomy.md §2.14`, `constraint.py::check`.
- Calibration patches → `replan.md §3` priority formula, `diagnose.md §3`
  `expected_gain_band_pct` emission.
- Anti-pattern patches → `axis_taxonomy.md §2.12` DANGER row pattern.
- Draft schema → `knowledge_draft.schema.json`.
- §S4 governance (drafts vs entries) → `skills/knowledge/SKILL.md` (TODO).
