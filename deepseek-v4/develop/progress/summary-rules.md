`deepseek-v4/develop` is the canonical path that holds the development
progress and supporting information for DeepSeek-V4. Use it to produce
an end-to-end summary of the current state of DeepSeek-V4 development.
Match the **visual style** of the screenshot I shared (style only — do
not reuse any of the data from it). Emit the summary as an HTML file
at `progress/develop-summary-{date}.html`.

**Page 1 — Agentic development methodology used on DeepSeek-V4.**
Cover the standing process: write rules first and follow them; plan
before executing; split work across multiple plans, each broken down
into phases; per-cycle workflow `plan → commit → develop → review →
(iterate on review feedback) → commit → update PR description`. Keep
a per-phase progress record under `progress/`. In optimization
phases, every phase change is paired with a profile + bottleneck
analysis, and the relevant operator perf table plus the EP8 proxy
end-to-end perf table under `perf/` are updated. Inspect the actual
layout under `develop/` and describe the development logic in your
own words to complete this page.

**Page 2 — Development progress (every plan and phase).** Show
how many phases have closed, which core models / modules were
built, etc. Group the phases by category (architecture, perf,
docs, …) and summarize what each group delivered. Use the
status table and per-phase summaries in `progress/` as the source
of truth.

**Page 3 — Optimization recap (perf-only phases).** Cover which
optimizations shipped, the EP8 proxy iter-time / TFLOP/s curves,
and the per-operator perf curves. Source the data from
`perf/proxy_ep8.md`, `perf/attention_perf.md`, and
`perf/elem_fusion.md`.

After rendering, review the final HTML to make sure no page has
truncated text or missing content.
