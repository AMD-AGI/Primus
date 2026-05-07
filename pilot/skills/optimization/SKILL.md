# Optimization Strategies

**Status**: Stub
**Read by**: Re-Plan Worker
**Domain**: per-bottleneck optimization knowledge

Per-bottleneck strategy subtrees:

| Bottleneck | Subdir | Primary moves |
|------------|--------|---------------|
| COMM       | `comm/`     | bucket / overlap / topology |
| PIPELINE   | `pipeline/` | vpp / microbatch / balance |
| MEMORY     | `memory/`   | recompute / offload / fragmentation |
| COMPUTE    | `compute/`  | mbs / parallel / kernel hints |
| MoE        | `moe/`      | routing / dispatch / load balance |

Each subdir has:
- `SKILL.md` — strategy overview for that bottleneck
- specific move files
- `env.md` (if applicable) — points to `skills/env/*.md` catalog

> env knowledge organization principle: `skills/env/*` is the **only catalog**; `optimization/*/env.md` only references it. See §4.2.
