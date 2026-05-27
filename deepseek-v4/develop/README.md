# DeepSeek-V4 in Primus — Development Workspace

`deepseek-v4/develop/` is the development knowledge base for **integrating
DeepSeek-V4 training support into Primus**. All plans, technical analysis,
and per-phase records live here.

## Directory Layout

```
deepseek-v4/develop/
├── README.md                     ← this file (quick index)
├── rules/                        ← project-wide working rules + habits
│   └── rule.md                   ← review-before-commit, status-pin pattern, etc.
├── techblog/                     ← Step 1 deliverable: deep-dive write-up + PNG diagrams
│   ├── README.md
│   ├── 01-deepseek-v4-architecture-deep-dive.md
│   ├── render_diagrams.py
│   └── diagrams/{architecture,csa,hca,mhc}.png
│
├── plan-<n>/                     ← per-plan documents (plan-0..plan-5 today)
│   ├── README.md                 (optional)
│   ├── 01-roadmap.md             phase overview, dependencies, milestones
│   ├── 02-phase-details.md       per-phase task lists, design notes
│   └── 03-test-strategy.md       gate matrix, ratchet, perf-budget contract
│
├── notes/                        ← ad-hoc notes / investigations during development
│   └── README.md
│
├── profile/                      ← perf-trace analysis reports
│   ├── _tools/                   ← report renderers
│   └── profile-{baseline,after-p<id>}-ep<N>-<YYYYMMDD>.{md,html}
│
└── progress/                     ← per-phase progress tracker
    ├── status.md                 ← real-time per-task status
    └── p<id>/                    ← per-phase scratch + summary
        ├── p<id>-summary.md      ← one-page phase close (R2.1)
        ├── run_smoke_*.sh
        ├── run_*_trace_*.sh
        └── ...
```

## Current Status

- [x] **Step 1**: architecture investigation + tech blog (see `techblog/`)
- [x] **Step 2**: development plan (see `plan-<n>/`)
- [-] **Step 3**: actual code development (in progress — currently
      plan-5 P29 closed, P30 next)

For the task-level breakdown see [`progress/status.md`](progress/status.md).

## Suggested Reading Order

| Audience | Path |
|---|---|
| **First-time reader / want the working rules first** | [`rules/rule.md`](rules/rule.md) |
| **Reviewer / want a 1-min overview** | `plan-<n>/README.md` → `plan-<n>/01-roadmap.md` (latest plan) |
| **Reviewer / want to know where each module lives** | `plan-1/01-code-layout.md` |
| **Developer / about to pick up a phase** | `plan-<n>/02-phase-details.md`, jump to that phase |
| **Want to understand the architecture itself** | `techblog/01-deepseek-v4-architecture-deep-dive.md` |
| **Want to track progress** | `progress/status.md` and the latest `progress/p<id>/p<id>-summary.md` |

## Naming Conventions

- **`primus/...`** — landing path for all changes; `third_party/` is read-only.
- **`deepseek-v4/develop/`** — **design docs / notes / progress only, no production code**.
- When citing the V4 reference implementation, use a relative path, for example
  `deepseek-v4/deepseek-ai/DeepSeek-V4-Flash/inference/model.py:Compressor`.
