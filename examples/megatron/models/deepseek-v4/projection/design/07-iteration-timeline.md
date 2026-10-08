# 07 — Iteration timeline (3-level composition view)

A visual, drill-down view of **how one training iteration's time is composed**,
layered from a single layer up to the whole pipeline schedule. It reuses the
projection math in `04-projection-math.md` (same controls, same per-layer times)
and adds only rendering + one pipeline-schedule simulator. Nothing here changes
the headline `iteration time` / `tokens/s/GPU` numbers; it explains them.

All times are microseconds (µs) unless noted, scaled to the active GPU tab by
Step 6 (`rowScaledTime`) exactly like the rest of the site.

## Why three levels

The iteration time is built bottom-up:

```
module time  --(sum per phase)-->  layer fwd/bwd        (Level 1)
layer times  --(map to PP/VPP)-->  per-device chunk cost (Level 2)
device costs --(1F1B schedule)-->  iteration timeline    (Level 3)
```

Each level answers one question:

- **Level 1 — where does a single layer's time go?** (attn / mlp / a2a)
- **Level 2 — how is work distributed across pipeline ranks?** (layer granularity)
- **Level 3 — how do the ranks overlap in time, and where is the bubble?**

## Module → category mapping (Level 1 granularity)

The default minimum granularity is three categories plus an explicit
"unattributed" bucket, derived from the `module` field (`03-json-schema.md`):

| category | modules | meaning |
|----------|---------|---------|
| `attn`   | `attn.norm`, `attn.proj`, `attn.core`, `attn.indexer`, `attn.misc` | attention (varies by `cr`) |
| `mlp`    | `moe.grouped_gemm`, `moe.shared_expert`, `moe.router` | expert / shared-expert compute (cr-independent) |
| `a2a`    | `moe.dispatch`, `moe.combine` | EP all-to-all (captured at EP=8) |
| `misc`   | any `*.misc` / unattributed | casts, scalar, control-flow; kept visible |

Rules:

- `attn.misc` counts as `attn` (it is attention-local unattributed time), while a
  standalone `misc` category is only used if a non-attn/non-moe module is
  unmapped. In practice every row maps to `attn` or `mlp`/`a2a`; the `misc`
  bucket surfaces `*.misc` share the same way `unattributedShare` does today.
- Because MoE is identical across `cr` (A10), only three representative layers are
  shown: `cr=0`, `cr=4`, `cr=128`. "Same params/category → show one" reduces to
  "one per cr".
- Optional drill-down expands `attn` into `core/indexer/proj/norm` and `mlp` into
  `grouped_gemm/shared_expert/router`.

## Level 2 — per-device chunk composition

Granularity is **one physical decoder layer**. The projection already maps every
layer to a `PP*VPP` chunk and a device (`04` Step 2). Level 2 exposes that map:

- Each device (PP rank) is a row; within it, chunks (VPP virtual chunks) are laid
  out in schedule order, each chunk a run of layers coloured by `cr`.
- Recomputed layers (their backward replays one forward, `04` Step 1) are marked
  (hatched / outlined) because they cost extra in `Db`.
- Non-layer parts are drawn on their owning device: `embedding` on device 0,
  `output`+`loss`+`MTP` on the last device.
- **Dedup:** devices with an identical ordered signature
  `(cr-list, recompute-flags, hasEmb, hasOut, hasMtp)` are drawn once, annotated
  `×N ranks (d..d)`.
- The **critical device** (`max Df` / `max Db`, the one that sets the pipeline
  critical path) is highlighted; each row shows its `Df`/`Db` and share of the
  critical stage, making load imbalance (the bubble's root cause) visible.

## Level 3 — pipeline schedule (Megatron-2 Figure 4 style)

A Gantt chart: y-axis = device, x-axis = time. Forward cells one colour, backward
another; VPP virtual chunks distinguished by lightness; bubbles are gaps.

Reference: Narayanan et al., "Efficient Large-Scale Language Model Training on
GPU Clusters Using Megatron-LM", arXiv:2104.04473, Figure 4 (default 1F1B on top,
interleaved below).

### Colour scheme

- forward = `--accent` (#4f8cff), backward = `--accent-2` (#36c08f).
- VPP chunk index shifts lightness (chunk 0 lightest → deeper for higher chunks),
  mirroring the paper's light/dark model-chunk shading.
- bubble = empty (optionally a faint diagonal hatch) with the fraction labelled.

### Schedule simulator

The site's analytic pipe time is
`pipe_compute = (GA + (PP-1)/VPP) * (Df_crit + Db_crit)` (`04` Step 3). To *draw*
the schedule we simulate 1F1B (optionally interleaved) microbatch ordering and
let bubbles emerge from the gaps.

Inputs: `PP`, `VPP`, `GA`, and per-virtual-chunk forward/backward cost. Each
virtual chunk `k` (device `k % PP`, vpp `⌊k/PP⌋`) uses the **exact** per-chunk
sums from `schedule.chunks` (`f_chunk[k]`, `b_chunk[k]`), with the non-layer
parts folded into the first (`embedding`) and last (`output`/`loss`/`MTP`) virtual
chunk so the drawn per-device length stays consistent with `Df`/`Db`. The
non-interleaved view collapses to one chunk per device with `f=Df[d]`, `b=Db[d]`.
Large `GA` is capped to `TL_VIS_GA_CAP` (48) microbatches for display only.

Algorithm (interleaved 1F1B, `VPP` model chunks per device, `k mod PP` device
map):

```
num_chunks   = PP * VPP
num_warmup   = min(GA, (PP - 1 - device) * ... )   # standard interleaved warmup
events[device] = ordered list of {kind:'F'|'B', mb, chunk, start, dur}
- time advances per device; a device starts an op when both its own timeline and
  the producing/consuming neighbour dependency are satisfied (F flows forward
  along devices, B flows backward), with p2p comm assumed zero (A4).
```

The simulator returns per-device event lists with `start`/`dur`; the drawn
iteration length `max_device(last_end)` is compared against the analytic
`pipe_compute` (pre-`calibFactor`). For a balanced schedule they agree; a
mismatch beyond tolerance is surfaced as a warning rather than hidden. The
analytic number remains the official iteration time; the Gantt chart is the
visualization and is scaled so its total width equals the analytic pipe time.

### Interleaving follows VPP

There is no separate interleaved/non-interleaved toggle: the schedule is driven
directly by the `VPP` control. `VPP=1` renders plain 1F1B (Figure 4 top);
`VPP>1` renders interleaved 1F1B (Figure 4 bottom), which shrinks the bubble from
`(PP-1)/GA` to `(PP-1)/(GA*VPP)` (A5). Change `VPP` in the projection controls to
compare.

## UI: layout + cross-level linkage

- **Layout toggle.** *Tabbed* shows one level at a time (L1/L2/L3 buttons);
  *Stacked (all + link)* renders all three top-to-bottom with section headings.
- **Drill-down linkage** (works in both layouts, most visible when stacked):
  - click a **cr** in Level 1 (or a layer cell in Level 2) → highlights that cr's
    layers in Level 2 and dims the rest; Level 1 emphasises the matching row.
  - click a **PP rank** in Level 2 (or a device row in Level 3) → highlights the
    linked rank in Levels 2 and 3, and Level 1 highlights the cr types present on
    that rank.
  - a "Clear" bar removes the selection.
- **Export.** Level 3's Gantt has *Export SVG* / *Export PNG* buttons; the
  serializer resolves the theme CSS variables and paints a background so the file
  is self-contained (good for slides / the paper-style figure).
- **Fit + zoom.** Level 3 fits the whole schedule in the panel at 1× (no
  scrollbar) for an at-a-glance overview. A zoom slider (1×–10×) widens the time
  axis so the per-cell microbatch numbers become readable; above 1× the chart
  scrolls horizontally. Zoom stretches only the time axis (font/row height fixed).
- **Cell tooltip.** Hovering a Gantt cell shows `compute <dur> µs` (the op's
  duration) and `starts @ <t> ms` (its start time measured from the iteration
  start). The x-axis is that same wall-clock time; gaps between cells are bubble.

## Consistency / self-checks

- Level 1 category sums per cr == `layerTimes(cr).fwd/bwd` (no time lost in
  categorization).
- Level 2 per-device `Df/Db` == `project().Df/Db` (same mapping, just detailed).
- Level 3 simulated iteration length ≈ analytic `pipe_compute` (balanced case);
  otherwise warn.
- All three levels react live to the shared projection controls (GPU tab, PP/VPP,
  GA/GBS/MBS/DP, recompute, manual layer-timing mode).

## Scope / caveats (inherit from 02-assumptions)

- `a2a` is captured at EP=8 intra-node; other EP values are not re-modeled (A7-A9).
- p2p PP comm is hidden; only the bubble is shown (A4).
- seq is fixed at the capture value (4096).
- MI455X tab rescales module times per Step 6 before all three levels are drawn.
