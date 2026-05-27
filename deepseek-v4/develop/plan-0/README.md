# Primus DeepSeek-V4 Integration Plan — Overview

This directory is the **development plan** document set for adding DeepSeek-V4
training support to Primus. After any interruption, simply pick up at the next
phase listed in [`00-roadmap.md`](00-roadmap.md).

## Document Index

| File | Description |
|---|---|
| [`00-roadmap.md`](00-roadmap.md) | **Overall roadmap.** Goals, milestones, dependencies for the 8 phases. Read this first. |
| [`01-code-layout.md`](01-code-layout.md) | **Full code landing list.** Every new / modified file with its phase and reference. |
| [`02-phase-details.md`](02-phase-details.md) | **Per-phase detailed plan.** Tasks, technical notes, exit criteria, risks. |
| [`03-testing-strategy.md`](03-testing-strategy.md) | **Test strategy.** Forward numerical alignment / training convergence / parallelism / perf. |

## One-Line Summary

> Add a new `model_type=deepseek_v4` to Primus following the existing mamba /
> MLA extension pattern: stand up the yaml + builder + LayerSpec + skeleton
> model_type first (P1–P3), then implement the 5 V4-specific modules
> (HC / Hybrid Attention / Hash MoE / MTP / Muon — P4–P5), and finally do
> correctness validation, perf tuning and FP4/FP8 (P6–P8).

## High-Level Phase Roadmap

```
                ┌────── Required for MVP (runs on a single MI355X) ──────┐
Phase 1   →   Phase 2   →   Phase 3   →   Phase 4   →   Phase 5
config &      register      LayerSpec    HC + Hybrid    MoE Hash
yaml          model_type    & dispatch   Attention      + sqrtsoftplus
                                                          + MTP
                                                          + clamped SwiGLU

                                                         ↓
                                            ┌── Muon optimizer ──┐
                                                         ↓
                ┌─── Correctness / Perf / Quantization (function first) ──┐
Phase 6   →   Phase 7   →   Phase 8
TP/PP/EP      numerical    FP4 / FP8
runs          alignment    quantization
              + convergence (deferable)
```

See [`00-roadmap.md`](00-roadmap.md) for details.

## What We **Are NOT** Doing in v1

To keep the MVP scope tight, the following items are **deferred by default**.
They are listed explicitly in the **out-of-scope** section of `02-phase-details.md`:

1. **Anticipatory Routing** — an optional optimization in the V4 paper, not
   required for convergence; deferred.
2. **A custom sparse-attn HIP kernel** — v1 uses dense + mask (matching the
   `inference/model.py` reference) for correctness; perf tuning lives in P8.
3. **FP4 training** — we land BF16 / FP8 first.
4. **Per-version configs for V4-Pro / V4-Pro-Thinking / V4-40B / V4-V40B** —
   v1 only validates V4-Flash; the other variants get yaml at the end of
   P1 but are not run.

## Already Delivered (Step 1)

- `../techblog/01-deepseek-v4-architecture-deep-dive.md` — V4 deep dive vs V3 / V3.2 (with 4 architecture diagrams).
- `../techblog/diagrams/{architecture,csa,hca,mhc}.png` — diagrams rendered directly via Pillow.

## How to Start Working

1. Open [`00-roadmap.md`](00-roadmap.md), decide which phase to work on.
2. In [`02-phase-details.md`](02-phase-details.md) jump to the matching phase
   section and walk down the task list.
3. After each task is done, tick it in [`../progress/status.md`](../progress/status.md)
   and fill in the commit hash.
4. Any investigation notes go into [`../notes/`](../notes/), filename
   `YYYY-MM-DD-<topic>.md`.
