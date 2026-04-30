# DeepSeek-V4 in Primus — Development Workspace

`deepseek-v4/develop/` is the development knowledge base for **integrating
DeepSeek-V4 training support into Primus**. All plans, technical analysis,
and per-phase records live here.

## Directory Layout

```
deepseek-v4/develop/
├── README.md                     ← this file (quick index)
├── techblog/                     ← Step 1 deliverable: deep-dive write-up + PNG diagrams
│   ├── README.md
│   ├── 01-deepseek-v4-architecture-deep-dive.md
│   ├── render_diagrams.py
│   └── diagrams/{architecture,csa,hca,mhc}.png
│
├── plan/                         ← Step 2 deliverable: development plan (this step)
│   ├── README.md                 ← navigation / overview of the plan documents
│   ├── 00-roadmap.md             ← overall roadmap: phase breakdown, milestones, dependencies
│   ├── 01-code-layout.md         ← full file landing list (where every new module goes in primus)
│   ├── 02-phase-details.md       ← per-phase task list, deliverables, exit criteria, risks
│   └── 03-testing-strategy.md    ← test strategy: numerical alignment, convergence, parallelism, perf
│
├── notes/                        ← ad-hoc notes / investigations during development
│   └── README.md
│
└── progress/                     ← per-phase progress tracker
    └── status.md                 ← real-time per-task status (manually or scripted)
```

## Current Status

- [x] **Step 1**: architecture investigation + tech blog (see `techblog/`)
- [x] **Step 2**: development plan (see `plan/`)
- [ ] **Step 3**: actual code development

For the task-level breakdown see [`progress/status.md`](progress/status.md).

## Suggested Reading Order

| Audience | Path |
|---|---|
| **Reviewer / want a 1-min overview** | `plan/README.md` → `plan/00-roadmap.md` |
| **Reviewer / want to know where each module lives** | `plan/01-code-layout.md` |
| **Developer / about to pick up a phase** | `plan/02-phase-details.md`, jump to that phase |
| **Want to understand the architecture itself** | `techblog/01-deepseek-v4-architecture-deep-dive.md` |
| **Want to track progress** | `progress/status.md` |

## Naming Conventions

- **`primus/...`** — landing path for all changes; `third_party/` is read-only.
- **`deepseek-v4/develop/`** — **design docs / notes / progress only, no production code**.
- When citing the V4 reference implementation, use a relative path, for example
  `deepseek-v4/deepseek-ai/DeepSeek-V4-Flash/inference/model.py:Compressor`.
