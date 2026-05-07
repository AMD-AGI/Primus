# state/ — runtime artifacts

This directory holds Pilot's working memory at runtime. **Everything here is gitignored.**

## Layout (created on demand by `tools/state.py`)

```
state/
├── cluster_profile.yaml          # reused across jobs (by version + age)
├── calibration_state.yaml        # §S1 calibrated Execution Model params
├── tuning_state.yaml             # entry of stage-exit checkpoints
├── plan_graph.yaml               # search-space tree
├── candidate_pool.yaml           # latest Re-Plan output
├── blacklist.yaml                # §S3.5 node blacklist
│
├── references/                   # §S2 reference curves
│   └── <model_family>__<dataset>__<seed>.yaml
│
├── checkpoints/                  # per-round full snapshots (replayable)
│   ├── r0/
│   ├── r1/
│   └── ...
│
├── handoff/                      # §13 strategy C: Orchestrator self-handoff landing point
│   └── <session_id>__<at>.yaml
│
├── knowledge_drafts/             # §S4 LEARN drafts (curator promotes via PR)
│   └── <session_id>/
│       └── <draft_id>.yaml
│
└── archive/                      # historical artifacts (drafts that didn't merge, old runs)
    └── ...
```

## Conventions

- **One YAML / JSON file per logical artifact.** Do not concatenate.
- **Append-only for `checkpoints/r<N>/`** — never overwrite a written round; if a round needs amending, write `r<N>_v2/`.
- **Per-session isolation**: long-running sessions create subdirectories under `checkpoints/` and `knowledge_drafts/` keyed by `session_id`.
- **External archival**: production deployments should rsync this directory to durable storage indexed by `session_id`.

## Regression / CI fixtures

CI fixture state lives separately at `tests/fixtures/state/`, not in this directory. Never run a real session against the test fixture.
