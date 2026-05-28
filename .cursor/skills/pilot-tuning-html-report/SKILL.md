---
name: pilot-tuning-html-report
description: Generate a self-contained, presentation-quality HTML report from a Primus Pilot tuning session markdown report. Use when the user attaches or references a Pilot session report (typically under `output/pilot/<session-id>.md`) and asks to generate an HTML version / HTML report / dashboard / visualization / shareable report — for example "@output/pilot/X.md generate an html report", "make an html version of this tuning report", "给这个调优报告生成 html", "把这个 session 报告做成 html", "html version of the pilot run". The skill emits a single self-contained `.html` file (no CDN, no external CSS/JS) with inline SVG charts (waterfall + per-round trajectories), KPI cards, decision trace table, lessons, and reproducible launch command — suitable for email / Slack / browser without a server.
---

# Pilot Tuning Session — HTML Report Generator

Use this when the user wants to turn a Pilot session report markdown into a presentation-quality HTML page. The generated HTML is the artefact people actually look at in meetings; the markdown stays the source of truth.

## Output contract

- **Input**: one Pilot session markdown (typically `output/pilot/<session-id>.md`).
- **Output**: one self-contained HTML file at the sibling path `output/pilot/<session-id>.html` (same basename, `.html` extension), overwriting if it exists.
- **No external deps**: no `<link>` to external CSS, no `<script src=...>` to CDN, no fonts loaded from Google Fonts. CSS inlined in `<style>`, charts as inline `<svg>`. Must open correctly when double-clicked from a file system with no network.

**Canonical reference example** (the visual / structural baseline to follow):
- Markdown source: `output/pilot/deepseek_v2_lite-mi300x-1node-20260515.md`
- HTML output: `output/pilot/deepseek_v2_lite-mi300x-1node-20260515.html`

Always read the reference HTML before generating a new one. It encodes the exact CSS, layout, and SVG patterns that this skill assumes — re-deriving them by hand will drift.

## When NOT to use

- Generic markdown → HTML conversion (use `pandoc` or similar — this skill is Pilot-session-shaped).
- Multi-session aggregation / dashboard. That's `tools/backend_gap_report/site/`, which is a separate, multi-report aggregator.
- A live dashboard with filters / search. This skill emits a static report for a single session.

## Expected markdown sections (what the skill extracts)

A Pilot session report — produced at the end of `pilot/skills/tuning-loop/` — has this section structure. The skill extracts the fields listed under each section.

| Markdown section | Fields to extract |
|---|---|
| **Header** (`## Tuning Report — <model> ... on <cluster>`) | model name, framework (BF16/FP8/...), cluster shape (e.g. `MI300X 1×8`) |
| **Session metadata bullets** | `Session ID`, `YAML` path, `Cluster` description, `Rounds` total + breakdown, `GPU·h spent`, `Wallclock`, `Baseline tps`, `Final tps`, `Net gain`, `Constraints met` (incl. mem cap value) |
| **Final plan** (`### Final plan — champion <id>`) | parallelism (tp/pp/dp/ep/vpp/cp), runtime (mbs/gbs/recompute/seq_len), comm flags, env diff, primus-defaults applied, Phase-B changes, dropped features, sidebars |
| **Decision trace — Phase A** | per-round row: id, variable changed, tps, Δ vs parent, mem, status, decision |
| **Decision trace — Phase B** | per-round row: id, variable, parent, tps, Δ vs parent, Δ vs strict champion, mem, status/decision |
| **Per-feature contribution** (code block) | ordered list of `(feature_name, cumulative_tps, Δ%)` increments from baseline to final |
| **Final bottleneck profile** (code block) | bottleneck class (COMPUTE_BOUND / COMM_BOUND / ...), `comm_ratio`, `bubble_ratio`, `overlap_ratio`, `moe_dispatch_ratio` (if MoE), `mem_peak`, `gpu_util_avg`, `step_time`, `MFU` |
| **Lessons** | numbered list of carry-forward lessons (typically 5–10) |
| **Things to retry** (table) | rows of `(item, why interesting, est. cost)` |
| **Artifacts on disk** (bullets) | per-round run dir paths + cluster baseline path |
| **Copy-pasteable override block** (bash code block) | the launch command |

If any section is missing in the source markdown, render the corresponding HTML block with a muted `not available in source report` note rather than fabricating data.

## HTML output structure (top → bottom)

This is the section order the reference HTML uses. Follow it exactly so reports stay comparable.

1. **`<header class="hero">`** — dark gradient banner with eyebrow ("Primus Pilot · Tuning Session Report"), model+cluster title, and a `<ul class="hero__meta">` of (Session, Date, Driver, YAML).
2. **`<div class="kpis">`** — 4 KPI cards floating over the hero seam (negative top margin). The four:
   - **Net throughput gain** (`+X.XX %`, `kpi__value--good` when positive)
   - **Memory peak** (`XXX.X GB`, with sub-line `vs YYY GB cap · ZZ.Z GB headroom`)
   - **Rounds** (total, with breakdown sub-line)
   - **Cost** (`~X.X GPU·h`, with wallclock sub-line)
3. **`<section class="block">` — Per-feature contribution** — inline SVG waterfall chart + a legend + a "note-box" with the headline takeaway.
4. **`<section class="block">` — Throughput & memory trajectory** — two side-by-side inline SVG line charts (one TPS, one memory) inside a `.chart-row` grid. Each has a `chart-legend`.
5. **`<section class="block">` — Final plan** — `.plan-grid` (3 cards: parallelism / runtime / comm) + dark `pre.code` block with the override command + muted expected-result line.
6. **`<section class="block">` — Final bottleneck profile** — bottleneck class as a `pill--champion`, then `.bottleneck-grid` (8 metric cards), then a paragraph of context.
7. **`<section class="block">` — Decision trace** — full `table.data` with all rounds. Champion / promote rows highlighted (`background: #ddf4ff`), baseline row highlighted (`background: #f8fafc`), cap-breach mem values in red.
8. **`<section class="block">` — Lessons** — `.lessons` grid (2 cols on desktop, 1 on mobile), one `.lesson` card per lesson with mono-font `<span class="lesson__num">Lx</span>` badge.
9. **`<section class="block">` — Things to retry** — `table.data` with item / why / cost columns.
10. **`<section class="block">` — Artifacts** — `table.data` listing every run dir, champion + confirm rows highlighted.
11. **`<footer class="foot">`** — single line: "Generated from `<source.md path>` · driven by Primus Pilot · Cursor agent runtime · `<date>`".

## Design tokens (must match reference)

Inline these as CSS variables in `<style>` — do not change without updating the reference:

```css
--bg: #fafbfc;          --surface: #ffffff;       --surface-2: #f3f5f8;
--border: #e2e6ec;      --border-strong: #c8cfd8;
--text: #1c2128;        --text-muted: #57606a;    --text-dim: #8c959f;
--accent: #0969da;      --accent-2: #0550ae;
--green: #1a7f37;       --green-bg: #dafbe1;
--red: #cf222e;         --red-bg: #ffebe9;
--amber: #9a6700;       --amber-bg: #fff8c5;
--slate: #6e7781;
--shadow: 0 1px 2px rgba(31,35,40,.04), 0 4px 12px rgba(31,35,40,.06);
--mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
```

**Status pill mapping** (used in tables and bottleneck class headers):

| Pilot status | CSS class | Color |
|---|---|---|
| `completed` / passing run | `pill--good` | green |
| `shelved` / sub-bar gain | `pill--warn` | amber |
| `dead axis` / `pseudo_dead` / `failed` / `nan` / `cap_breach` | `pill--bad` | red |
| `promote` / `champion` / `confirms champion` | `pill--champion` | blue (`#ddf4ff`) |
| neutral / informational | `pill--neutral` | slate |

**Delta cell styling** in tables: `delta-pos` green, `delta-neg` red, `delta-zero` muted.

**Font rules**: numeric values in tables / KPIs / metric grids use `var(--mono)` and `font-variant-numeric: tabular-nums`. Body text uses system-ui.

## Chart math (precise — copy into the SVG)

All charts are **inline SVG with viewBox** for crisp scaling. No `width` attribute — the container CSS scales them responsively.

### Waterfall (per-feature contribution)

- `viewBox="0 0 1100 380"`, plot area x ∈ [100, 1060], y ∈ [40, 320]
- Pick Y range so it covers `[round_to_500_below(baseline), round_to_500_above(final_champion)]` with one extra gridline of headroom (reference uses 16,000–18,000 for a baseline of 16,467 and champion of 17,492).
- 5 horizontal gridlines (dashed, `#e2e6ec`), the bottom one solid. Y labels right-aligned at `x=92`, mono font, in `var(--text-dim)`.
- One bar per step in this order: baseline (full grey bar, fill `#9aa5b1`), one green floating bar per primus-defaults / Phase-B contributor (fill `#1a7f37`), final blue bar showing the 30-step confirmed champion (fill `#0969da`).
- Bar width 110px, gap 20px, starting at x=115.
- For each floating bar, draw a dashed connector line from the top of the previous bar to align with the start of the current bar.
- Above each bar: the increment `+ NNN` in green bold (12px), or the absolute value for baseline/champion in the bar color bold.
- Below each bar: feature name (slate, weight 500), then a 10px line with round id and Δ%.

### TPS-over-rounds line chart

- `viewBox="0 0 560 320"`, plot area x ∈ [55, 540], y ∈ [30, 275].
- Y range: span from `round_to_500_below(min_tps_in_completed_rounds)` to `round_to_500_above(max_tps_including_pseudo_dead)`. Reference uses 16,000–18,500.
- X axis: one x position per round, step = `(540-70)/(N_rounds-1)` (or fixed 30px per round if N≤16), starting at x=70.
- Champion lineage polyline: connect baseline → each promoted champion → final confirm point, stroke `var(--accent)` 2px, round caps.
- Round markers:
  - **baseline**: gray dot (`#6e7781`), r=5
  - **promoted / champion**: blue dot (`var(--accent)`), r=5 (r=6 with white stroke for the headline champion)
  - **confirm run**: blue dot with white stroke, r=6
  - **shelved**: amber dot (`#d4a017`), r=5
  - **dead axis**: red dot (`var(--red)`), r=5
  - **pseudo_dead / cap_breach**: red dot, r=6 with white stroke; annotate "cap breach" or "XXX.X GB" above the point in red 10px
  - **failed / NaN (no tps measurement)**: render an "✕" character at the chart center y for that x, red 13px bold (no dot)
- Round labels under x axis: `r0` ... `rN`, slate 9px, centered.

### Memory-over-rounds line chart

- Same viewBox / plot area as TPS chart.
- Y range 100–200 GB (reference). If max mem > 200, shift range up by 50 GB increments.
- **180 GB cap line**: solid red dashed line at the user cap with the label "180 GB cap" right-aligned. Use the actual cap from the source markdown's `Constraints met` section.
- HBM ceiling annotation (e.g. "192 GB HBM ceiling" for MI300X) in dim text above the cap line.
- Champion lineage polyline + per-round dots use the same color/size conventions as the TPS chart.
- Cap-breach round: red dot r=6 with white stroke, with the mem value annotated above in red bold.

### Legend

Under each chart, a `<div class="chart-legend">` with one `<span>` per category present in the chart (no empty legends). CSS provides the swatch via `::before`.

## Workflow

### Step 1 — Resolve and read the input

The user typically `@`-attaches the markdown or names it (`output/pilot/X.md`). Read it with `Read`. If the path is ambiguous, ask which session:

> Which Pilot session report should I render? (e.g. `output/pilot/deepseek_v2_lite-mi300x-1node-20260515.md`)

### Step 2 — Read the canonical reference

Read `output/pilot/deepseek_v2_lite-mi300x-1node-20260515.html`. This is the visual / structural baseline. Treat it as a template, not a literal copy — adapt content but keep the structure, CSS, and chart math.

### Step 3 — Extract structured data

Parse the markdown into a structured plan in your head:

```
session = { id, model_name, framework, cluster_shape, date, yaml_path }
kpis    = { net_gain_pct, mem_peak_gb, mem_cap_gb, mem_headroom_gb,
            rounds_total, rounds_breakdown_str, gpu_h, wallclock_str }
final_plan = {
  parallelism: { tp, pp, dp, ep, vpp, cp },
  runtime: { mbs, gbs, recompute, seq_len },
  comm: { overlap_grad_reduce, overlap_param_gather, env_diff },
  override_cmd: <bash block as-is from markdown>,
  expected: { tps, mem_gb, step_ms, mfu, notes }
}
bottleneck = { class, comm_ratio, bubble_ratio, overlap_ratio,
               moe_dispatch_ratio?, mem_peak_gb, gpu_util_avg,
               step_time_ms, mfu, narrative }
rounds = [ { id, parent, variable_changed, tps?, delta_parent_pct?,
             delta_vs_strict_pct?, mem_gb?, status, decision } ]
features = [ { name, round_id, delta_tps, delta_pct, cumulative_tps,
               note? } ]   // for waterfall, in promote order
lessons = [ { num, body_html, ...key_terms_as_code } ]
retries = [ { item, why, cost_str } ]
artifacts = [ { round_id?, description, path } ]
```

For rounds with no tps (NaN / failed / cap-breach without recorded number) keep `tps=null` and render `—` in the table + "✕" on the TPS chart.

### Step 4 — Compute chart geometry

Calculate Y ranges, x positions, and the bar/dot coordinates from the structured data using the formulas above. Don't eyeball — the chart precision is what makes the report look professional.

### Step 5 — Write the HTML file

Use `Write` to emit the full `<!DOCTYPE html>` ... `</html>` content. The file must be self-contained: all CSS in one `<style>` block in the head, all SVG inline in the body. No external resources.

Save to `output/pilot/<same-basename-as-input>.html`.

### Step 6 — Self-check (mandatory before finishing)

Re-read the output (or its line count) and verify:

- [ ] File starts with `<!DOCTYPE html>` and ends with `</html>`.
- [ ] No `http://` or `https://` URLs in `<link>`, `<script>`, `<img>` tags (data attributes / footer attribution text are fine).
- [ ] Hero contains the session id and model+cluster title.
- [ ] Exactly 4 KPI cards, in order: gain → mem → rounds → cost.
- [ ] Waterfall has `1 + N_increment + 1` bars (baseline + each contributor + final champion), and bar tops form a non-decreasing staircase (cumulative tps), barring intentional dips.
- [ ] Both round-trajectory charts (TPS and memory) have a dot or "✕" for every round in the source, in r0..rN order.
- [ ] Memory chart has the 180 GB (or actual) cap line annotated.
- [ ] Decision trace table has one row per round, with the champion row(s) highlighted via inline `style="background: #ddf4ff;"`.
- [ ] Override command block matches the markdown's bash block byte-for-byte (escape `<` `>` `&` if any, otherwise pass through).
- [ ] Number of `.lesson` cards equals the count in the markdown.
- [ ] Footer cites the source markdown path.

Then report to the user the absolute output path and a one-line content summary.

## Edge cases and conventions

| Case | How to render |
|---|---|
| Source markdown is missing the `Final bottleneck profile` block | Render the section with a muted "Profile not present in source report" note instead of fabricating zeros |
| Champion was promoted multiple times (e.g. r1, r2, r4 are all `keep on`) | Connect ALL of them in the TPS / memory polyline; mark only the final promote as `★ champion` |
| FP8 session (no MoE) | Drop `moe_dispatch_ratio` from the bottleneck grid (one fewer card); drop MoE-specific features from the waterfall |
| Dense session (no EP) | Final plan grid: still render `ep: 1` in the parallelism card for shape parity |
| Source markdown title is in Chinese | Keep the title as-is in the hero; UI labels (KPI labels, table headers) stay English |
| Mem cap is not 180 GB | Read it from `Constraints met` line (e.g. "≤ 200") and use the actual value everywhere — KPI sub, chart line, narrative |
| No `Things to retry` section | Drop that whole `<section>`, do not render an empty block |
| Round count > 20 | Switch the per-round chart x-step to `(540-70)/(N-1)`; bump label font to 8px |
| Multi-node session | Title becomes "X×8" (e.g. "MI300X 2×8"); cluster sub-line in hero meta reflects nodes count |

## Stylistic non-negotiables

- **Never** use a chart library (Chart.js, D3, ApexCharts, …) — inline SVG only, so the file works offline and in email clients.
- **Never** add Google Fonts or any external font — system-ui stack only.
- **Never** add JavaScript — the report is read-only static HTML.
- **Never** truncate the decision trace table — every round in the markdown gets a row.
- **Always** show shelved / dead / pseudo-dead / failed rounds in the trajectory charts — the search behaviour is part of the story, not just the winners.
- **Always** highlight the champion row and the confirm row in the decision trace via inline background color.
- **Always** make the override command copy-pasteable (proper `<pre>` block, no smart quotes).

## Acceptance criteria (one-line definition of done)

The output HTML opens in a browser with no network, fits a 1280×800 viewport without horizontal scroll, and a reader who has not read the source markdown can in 30 seconds answer: **(1) what was tuned, (2) by how much, (3) what the champion plan is, (4) which axes were dead / shelved**.
