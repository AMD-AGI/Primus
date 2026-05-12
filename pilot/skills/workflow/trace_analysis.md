# Trace Analysis (Trace → Bottleneck Evidence)

**Status**: v0.5 (adds fwd / bwd / post-bwd-comm / optim phase split using
PyTorch autograd-engine + Optimizer.step markers; per-bucket roll-up grows
an ``avg_ms/kernel`` column and a sub-table per phase). Builds on v0.4
(generic, reusable; full kernel cost list, per-stream pipeline timeline,
pure-compute / pure-comm / overlap / bubble decomposition).
**Scope**: any consumer of a PyTorch chrome trace (tuning workflows, ad-hoc
performance reviews, regression triage, hardware enablement, etc.). The
output of this skill is plain documentation; nothing in the report is
specific to one workflow.

**Tool boundary**: `pilot.tools.trace_analyze.run(trace_meta)` — pure
analysis. The tool produces evidence; downstream consumers (e.g. a tuning
DIAGNOSE step) interpret it.

**Input** (`trace_meta` shape, see `profile.md`): a JSON document pointing
at one or more chrome-trace files captured around a single steady-state
iteration of training.

**Output (mandatory)**:
- `<run_dir>/trace_analysis.md` — human-readable report (the user-facing artifact)
- `<run_dir>/trace_analysis.json` — machine-readable; same data, full schema

The optional session-symlink (e.g.
`state/sessions/<sid>/trace_analyses/<run_id>.md`) is the responsibility
of the workflow that called this skill, not of this skill itself.

---

## 1. Goal

Given one chrome-trace JSON of one steady-state iteration, produce a
**covered, evidence-rich** description of what the iteration spent its time
doing. Coverage matters more than cleverness: **every kernel and every CPU
range in the iteration MUST be accounted for**. Unknown should be small (<
1% of self-time on MI355X workloads); large unknown means
`_trace_patterns.json` is under-specified.

The report does NOT decide the bottleneck. It only emits evidence; DIAGNOSE
maps evidence → verdict.

## 2. The single iteration boundary (CRITICAL)

The captured iteration is bounded by, in priority order:

1. **`ProfilerStep#N` user_annotation** (PyTorch profiler emits this around
   the entire `schedule.step()` window — both CPU + GPU). When only one iter
   is captured (warmup=5, capture=1, the default), THIS IS THE ONLY RELIABLE
   ITER BOUNDARY. Use the longest range matching `^ProfilerStep#\d+$` of
   `cat in {user_annotation, cpu_op}`.
2. Two adjacent `Optimizer.step#*` X-events: use the gap (end-of-first to
   start-of-second). Only valid if at least 2 iters captured.
3. ❌ **Single Optimizer.step is NOT an iter** — it's just the optim
   window inside one iter (~0.06% of iter for FP8 LLM). Falling back to it
   produces a 0.88ms "iter" that has 23 kernels. Earlier v0.1 of this
   analyzer did this and produced nonsense.
4. Last-resort: full GPU kernel timespan (warning emitted).

`iter_wallclock_us` = `range_end - range_start` of the chosen boundary.

## 3. Bucketing: every kernel goes into exactly one bucket

Patterns live in `pilot/tools/_trace_patterns.json` — edit there, no code
change needed. **Order matters; first match wins**, so put more specific
patterns first.

### Comm buckets

| Bucket | What it contains | Notes |
|---|---|---|
| `comm_collective` | RCCL collective kernels | ROCm collapses ALL collectives (alltoall, allreduce, allgather, reduce_scatter, broadcast, p2p) into a SINGLE `ncclDevKernel_Generic_1` — the device kernel name does NOT distinguish operation. Use CPU-side `nccl:*` user_annotations to disambiguate (pending feature). |
| `comm_moe_dispatch` | MoE expert-parallel dispatch/combine via `primus_turbo::deep_ep::*` | Logically EP comm but NOT RCCL. Includes routing helpers (`_permute_kernel`, `_unpermute_kernel`, `_row_id_map_pass_*`, `_indices_to_multihot_kernel`, `fused_scaling_group_sum_routing*`, `compute_group_offs_device`). |
| `comm_alltoall_nvidia` | NVIDIA-only NCCL alltoall (empty on ROCm) | Reserved bucket for portability. |
| `comm_allreduce_nvidia` | NVIDIA-only NCCL allreduce (empty on ROCm) | Reserved bucket. |
| `comm_allgather_nvidia` | NVIDIA-only NCCL allgather (empty on ROCm) | Reserved bucket. |
| `comm_reduce_scatter_nvidia` | NVIDIA-only NCCL reduce_scatter (empty on ROCm) | Reserved bucket. |
| `comm_p2p` | pipeline send/recv | `nccl::Send/Recv`, `pp_p2p_*`. Empty when PP=1. |

### Compute buckets

| Bucket | What it contains | Notes |
|---|---|---|
| `compute_attention` | flash attention | `aiter::fmha_*`, `flash_attn`, `fmha_v3`, `mha_*`, `mla_*` |
| `compute_norm_act` | layernorm/rmsnorm/softmax/SiLU/GeLU/SwiGLU | `rmsnorm_fwd/bwd`, `layernorm_*`, `te_norm`, `_softmax_`, `fused_swiglu`, `fused_silu`, `fused_gelu`, `triton.*sigmoid_silu` |
| `compute_gemm_fp8_grouped` | FP8 grouped GEMM for MoE (the dominant compute kernel for MoE FP8) | `QuantGroupedGemmKernel`, `quantgroupedgemm`, `compute_grouped_gemm_fp8`, `grouped_gemm_variable_k`, `primus_turbo::grouped_gemm` |
| `compute_gemm` | dense GEMM (rocBLAS Cijk_, hipblaslt, ck_tile) | `Cijk_`, `ck_tile.*gemm`, `ck_tile::gfx950_t`, `tensor::*::matmul`, `te_gemm`, `cutlass`, `cublasLt`, `hipblaslt`, `rocblas_*gemm`, `Hgemm_` |
| `compute_fp8_prep` | FP8 cast/scale/amax overhead | `_amax_reduce`, `_amax_compute`, `compute_scale_from_amax`, `_cast_transpose`, `fp8_cast`, `fp8_quant`, `primus_turbo::reduce_row_kernel`, `primus_turbo::unary_kernel`, `primus_turbo::compute_scale`. **Lump separately so we can see how much FP8 admin costs us.** |
| `compute_optim` | optimizer step (AdamW, FusedAdam, multi_tensor) | `_adam`, `adamw_`, `fusedadam`, `multi_tensor_apply` |
| `compute_other` | catch-all for elementwise / reduce / cat / pointwise math | `elementwise_kernel`, `vectorized_elementwise`, `reduce_kernel`, `CatArrayBatchedCopy`, `triton_poi_fused`, `triton_per_fused`, `triton_red_fused`, `cross_entropy_kernel`, `_kernel<` |

### Other

| Bucket | What it contains | Notes |
|---|---|---|
| `memcpy` | Explicit copies: `cudaMemcpy*`, `hipMemcpy*`, `Memcpy DtoD/DtoH/HtoD`, `Memset` | Usually small. |
| `schedule_overhead` | CPU-side launch latency, `c10d::wait`, NVTX | Currently NOT counted in compute/comm ratios because we filter to `cat in {kernel, gpu_op}` for the GPU-stream wall-time analysis. |

A kernel is matched against the patterns **case-insensitively** in the
order they appear in `_trace_patterns.json`. If no bucket matches, it goes
to `compute_other` AND its name is added to `top_unknown_names` for
visibility (so we see what to add).

## 4. Bucket roles for headline ratios

The `comm_ratio` and `compute_ratio` headlines auto-derive from bucket id
prefixes — bucket id starting with `comm_` → comm role, `compute_` →
compute role, `memcpy` → compute role, `schedule_overhead` → overhead
(excluded from headline ratios). **No code change needed when you add a
new `comm_xxx` or `compute_xxx` bucket** — it rolls up automatically.

## 5. Per-bucket metrics (the schema for `trace_analysis.json`)

```json
{
  "schema_version": "0.1",
  "trace_path": "...",
  "iter_wallclock_ms": 31999.30,
  "iter_boundary_source": "profiler_step",
  "total_kernel_count": 72121,
  "buckets": {
    "compute_gemm_fp8_grouped": {
      "self_time_ms": 18961.16,
      "self_time_pct": 0.5925,
      "wall_time_ms": 18961.16,
      "wall_time_pct": 0.5925,
      "kernel_count": 2496,
      "p50_kernel_ms": 4.86,
      "p99_kernel_ms": 24.31,
      "top_kernels": [
        {"name": "QuantGroupedGemmKernel<...>", "self_time_ms": 12126.29, "count": 416},
        ...
      ]
    },
    "comm_moe_dispatch": { ... },
    ...
  },
  "ratios": {
    "comm_ratio": 0.0734,
    "comm_collective_ratio": 0.0013,
    "comm_moe_dispatch_ratio": 0.0724,
    "comm_p2p_ratio": 0.0,
    "compute_ratio": 0.9212,
    "compute_gemm_ratio": 0.8024,           // grouped + dense
    "compute_attention_ratio": 0.0072,
    "compute_fp8_prep_ratio": 0.0326,
    "compute_norm_act_ratio": 0.0189,
    "compute_optim_ratio": 0.0006,
    "compute_other_ratio": 0.0596,
    "memcpy_ratio": 0.0,
    "bubble_ratio": 0.0061,
    "schedule_overhead_ratio": 0.0
  },
  "overlap": {
    "comm_compute_overlap_ms": 23.48,
    "comm_compute_overlap_pct_of_comm": 0.0100,
    "longest_serialized_comm_ms": 16.08
  },
  "memory": {
    "peak_alloc_mb": null,
    "peak_reserved_mb": null,
    "fragmentation_ratio": null
  },
  "pipeline": {
    "pp_size": null,
    "p2p_send_recv_pairs": 0,
    "estimated_bubble_pct": null
  },
  "coverage": {
    "kernel_self_time_ms": 32030.20,
    "unknown_self_time_ms": 25.50,
    "unknown_ratio": 0.0008
  },
  "top_unknown_names": [...],
  "warnings": []
}
```

Every entry that is `null` MUST be accompanied by a corresponding
`warnings[]` entry explaining why (e.g. `"memory.peak_alloc_mb=null because
trace lacks memory_alloc events; enable torch.profiler with
profile_memory=True"`).

## 6. Coverage discipline

After all buckets are filled, the analyzer asserts (soft):

```
coverage.unknown_ratio < 0.01   (1% of kernel self-time)
```

If it doesn't, the analyzer emits a `high_unknown_ratio` warning AND lists
the top 20 unaccounted kernel names in `top_unknown_names`. **A
high_unknown_ratio warning is the trigger for the next iteration of this
skill** — look at the unknowns, pick a pattern, add it to
`_trace_patterns.json`, and re-run.

## 7. Sanity checks (the report flags but doesn't fail)

| Check | Threshold | What it tells us |
|---|---|---|
| `iter_wallclock_ms` matches Primus log iter time | ± 5% | sanity: the captured iter is representative of steady state |
| `iter_boundary_source == "profiler_step"` | (warning if otherwise) | the chosen iter window is the canonical full step |
| `unknown_ratio` | < 0.01 | the pattern table is well covered |
| `kernel_count` | > 1000 (steady iter) | trace flushed completely; not truncated |
| `top_kernels[0].count > 1` per bucket | > 1 | not capturing only kernel-compile noise |

## 8. The Markdown report

Layout of `trace_analysis.md`. Section names are STABLE across versions
(no internal numbering like `8.1`). Sections appear in this order:

1. **Header** — iter wallclock, **per-phase wallclock chip** (fwd / bwd /
   post_bwd_comm / optim / tail with absolute ms and % of iter), boundary
   source, kernel count, coverage
2. **Per-iter cost — bucket roll-up** — `Overall` table + per-phase
   sub-tables (`Forward`, `Backward`, `Post-bwd comm`, `Optimizer`); each
   table has an `avg_ms/kernel` column (= self_ms / kernel count) so
   "small kernels with large total" vs "few large kernels" is visible at
   a glance
3. **Per-iter cost — full kernel list** — every distinct kernel name
   that ran in the iter, sorted by `wall_ms` desc; the "which exact
   kernel ate my iter?" view (configurable cap; default = full list)
4. **Compute / Comm / Overlap / Bubble decomposition** — the four
   mutually-exclusive wall-time slices of the iter (pure compute alone,
   pure comm alone, comm ∧ compute simultaneous, GPU idle)
5. **Iter pipeline timeline (per stream)** — one ASCII row per CUDA
   stream observed in the trace, plus a "comm presence" overlay row;
   shows what each stream was doing across the iter
6. **Headline ratios** — the bucket → ratio map (machine-friendly column)
7. **Per-bucket drill-down** — each bucket's top kernels (capped per
   bucket to keep the report navigable)
8. **Top unknown kernels** — anything that fell into the catch-all bucket
9. **Warnings** — anything the analyzer wants to flag (coverage,
   boundary fallback, truncated trace, etc.)

The same data lives in the JSON output under stable top-level keys
(`buckets`, `kernels_full`, `cost_breakdown`, `pipeline_timeline`,
`ratios`, `top_unknown_names`, `warnings`).

### Per-iter cost — bucket roll-up

One `Overall` table covering the whole iter, plus one sub-table per
phase (`Forward`, `Backward`, `Post-bwd comm`, `Optimizer`) when phase
markers were detected. Within each table: one row per bucket present in
that window, sorted by `wall_ms` desc. The `Overall` table also appends
a `<bubble>` row = `iter_wallclock − union(all_kernel_intervals)` so the
column sum reads close to `iter_wallclock_ms`. Per-phase tables omit the
bubble row (per-phase bubble is the phase wallclock minus union of
in-phase kernels, computable from the data; not currently rendered).

Each row has 8 columns:

| Col | Meaning |
|---|---|
| `Bucket` | bucket id |
| `wall_ms` | UNION of this bucket's kernel intervals inside the window (no double-counting cross-stream concurrency) |
| `wall %` | wall_ms ÷ window wallclock (iter wallclock for `Overall`; phase wallclock for sub-tables) |
| `self_ms` | SUM of every kernel's clipped `dur` |
| `avg_ms/kernel` | `self_ms / kernel_count` — single-kernel typical cost |
| `kernels` | number of kernel events in this bucket × window |
| `p50_ms`, `p99_ms` | per-kernel duration percentiles |

Phase tagging rule: a kernel is assigned to whichever phase its
**midpoint** falls into. This avoids splitting one kernel across phases
(which would inflate `kernels` and skew `avg_ms/kernel`). Kernels whose
midpoint is between `bwd_end` and `optim_start` land in `post_bwd`, etc.

Why both `wall_ms` and `self_ms`? `self_ms` = SUM of every kernel's
`dur` (double-counts cross-stream concurrency). `wall_ms` = UNION of those
intervals (no double counting). On a single stream they are equal; on
multi-stream the gap is the within-bucket parallelism.

Per-phase wallclock is computed from PyTorch user-annotation markers:

| Phase | Window | Detected from |
|---|---|---|
| `fwd` | `[iter_start, bwd_start]` | derived |
| `bwd` | `[first, last]` of `autograd::engine::evaluate_function:*` cpu_op | autograd engine |
| `post_bwd_comm` | `[bwd_end, optim_start]` | derived (typical: DDP grad allreduce) |
| `optim` | union of `Optimizer.step#*` events | user annotation |
| `tail` | `[max(bwd_end, optim_end), iter_end]` | derived (book-keeping) |

If the trace has no `autograd::engine::evaluate_function:*` events
(some training frameworks don't go through PyTorch's autograd engine),
the per-phase chip and sub-tables are omitted; `Overall` is still shown.

### Per-iter cost — full kernel list

Every distinct kernel name (chrome-trace `event.name`) that ran inside
the iter window, ranked by `wall_ms` desc. Default cap = unlimited; CLI
flag `--kernels-cost-list-cap N` truncates to top N if a smaller report
is wanted (recommended: 50 for a quick skim, no cap for forensic work).

```
| # | kernel | bucket | wall_ms | wall % | calls | p50 | p99 |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | ck_tile QuantGroupedGemmKernel [MT256x256x128] | compute_gemm_fp8_grouped | 12126.29 | 37.90% | 416 | 29.19 | 37.07 |
| 2 | ck_tile QuantGroupedGemmKernel [MT256x256x128] | compute_gemm_fp8_grouped |  6031.41 | 18.85% | 416 | 14.46 | 19.49 |
| 3 | ck_tile FmhaBwdDQDKDVKernel                    | compute_gemm             |  4498.39 | 14.06% | 216 | 20.83 | 20.98 |
| ... (every distinct kernel listed)
```

Truncation rules for the displayed name:
- Strip `void` prefix, `at::native::`, `c10::`, `transformer_engine::`
  namespace prefixes, and trailing `(arg, list, ...)`.
- For Itanium-mangled symbols (`_ZN…`), recognize known operator names
  (`QuantGroupedGemmKernel`, `FmhaBwdDQDKDVKernel`, etc.) and dig out the
  GEMM tile shape (`sequenceIJLi256ELi256ELi128EEE` → `MT256x256x128`).
- Hard cap at 80 chars; append `…` if truncated.
- The full mangled name is preserved in the JSON (`name`); the truncated
  string is in `name_short`.

JSON schema (top-level field `kernels_full`):

```json
"kernels_full": [
  {"name": "_ZN7ck_tile6kentry...", "name_short": "ck_tile QuantGroupedGemmKernel [MT256x256x128]",
   "bucket": "compute_gemm_fp8_grouped",
   "wall_ms": 12126.29, "wall_pct": 0.379, "self_ms": 12126.29,
   "calls": 416, "p50_ms": 29.189, "p99_ms": 37.069,
   "streams": [7]},
  ...
]
```

### Compute / Comm / Overlap / Bubble decomposition

The four mutually-exclusive slices of `iter_wallclock_ms`. Sum equals
`iter_wallclock_ms` (no double counting). Together they answer "what
fraction of the iter was actually waiting on compute alone, vs comm alone,
vs both at once, vs no GPU work?".

```
| Slice | wall_ms | wall % | what it is |
|---|---:|---:|---|
| pure compute       | 28800.50 | 90.00% | compute kernels live AND no comm kernels live |
| pure comm          |    40.19 |  0.13% | comm kernels live AND no compute kernels live |
| comm ∧ compute     |  2275.36 |  7.11% | both compute and comm kernels live simultaneously |
| bubble             |   195.20 |  0.61% | no GPU kernel live (idle GPU) |
| --- legacy --- | | | |
| total compute wall | 31075.86 | 97.11% | union of all compute-bucket intervals (= pure_compute + overlap) |
| total comm wall    |  2315.55 |  7.24% | union of all comm-bucket intervals (= pure_comm + overlap) |
| longest serialized comm window | 16.08 ms | --- | longest contiguous comm-only sub-window (bottleneck candidate) |
```

The first 4 rows partition `iter_wallclock_ms` exactly. The "legacy"
rows are the two pre-existing union totals (`comm_wall`, `compute_wall`)
that workflows already consume; included for backward compatibility with
the `overlap` JSON object.

JSON schema (top-level field `cost_breakdown`):

```json
"cost_breakdown": {
  "iter_wallclock_ms": 31999.30,
  "pure_compute_ms": 28800.50, "pure_compute_pct": 0.9000,
  "pure_comm_ms":       40.19, "pure_comm_pct":    0.0013,
  "overlap_ms":       2275.36, "overlap_pct":      0.0711,
  "bubble_ms":         195.20, "bubble_pct":       0.0061,
  "compute_wall_ms": 31075.86, "comm_wall_ms":     2315.55,
  "longest_serialized_comm_ms": 16.08
}
```

### Iter pipeline timeline (per stream)

Two-row `header + N rows + 1 overlay`:

- **Header**: `<iter_start>` … `<iter_end>` markers, plus the cell width.
- **One row per CUDA stream observed** in the trace (sorted by total
  kernel wall-time desc, so the busiest stream is on top). Each cell
  shows the glyph of the bucket whose kernels occupied the MOST wall time
  inside that cell, **on that specific stream**. Empty cell on that stream
  → space (the kernel was running on a DIFFERENT stream that cell). Zero
  GPU work anywhere → `.` (bubble).
- **Comm-presence overlay**: a final row that lights up `M` / `C` / `c`
  whenever the union of comm intervals (any stream) ≥ 5% of that cell
  wall. This survives the case where compute drowns out comm in the
  per-stream rows.

Glyph table (stable across reports for visual comparability):

```
G  compute_gemm_fp8_grouped
g  compute_gemm
A  compute_attention
N  compute_norm_act
F  compute_fp8_prep
O  compute_optim
o  compute_other
M  comm_moe_dispatch
C  comm_collective (RCCL ncclDevKernel*; per-collective subtype TBD)
c  comm_*_nvidia / comm_p2p (NCCL on nvidia or PP send/recv)
m  memcpy
s  schedule_overhead
.  bubble
?  mixed (no bucket > 30% of cell on that stream)
```

Stream IDs are the chrome-trace `tid` of the GPU-kernel events
(PyTorch chrome-trace uses one `tid` per CUDA stream on the GPU
process). Compute and comm typically live on different streams when the
framework configures them so (e.g. `compute_stream` and `comm_stream`).
Co-located buckets on one stream means everything is serialized on that
stream — usually a sign that the workload didn't enable a separate
comm stream.

Rendered example (deepseek_v2_lite-FP8, 80 cells × 400 ms each, 4 GPU
streams observed):

```
iter pipeline (80 cells × 400.0 ms each, total = 31999.30 ms)
                 0%                                                                       100%
  stream  7 →  GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG    (compute)
  stream 14 →  M  M  M MM MM    M  MM  MM   MM    M MM     MM   MM    MMMM  MM M    M M       (comm: deepep + RCCL)
  stream 22 →           o    o          o o   o            o            o     o                (compute_other tails)
  stream 26 →                                                                            o     (optimizer step)
  comm ≥5%  →  MMM  M M  MMM     M MMM  M M  MMMMMMM M MMMMMMM   MMMM  M   MMMMM  MM MMM M MM M
              └──────────────────────────────────────────────────────────────────────────────┘
              start of iter                                                        end of iter
```

JSON schema (top-level field `pipeline_timeline`):

```json
"pipeline_timeline": {
  "width_cells": 80,
  "cell_us": 399991,
  "streams": [
    {"stream": 7,  "wall_ms": 30640.0, "glyphs": "GGGG…", "kernels_top": ["ck_tile Quant…", …]},
    {"stream": 14, "wall_ms":  2315.5, "glyphs": "M  M…", "kernels_top": ["deep_ep::intranode::dispatch", …]}
  ],
  "comm_overlay_glyphs": "MMM M M MMM    …",
  "legend": {"G": "compute_gemm_fp8_grouped", "M": "comm_moe_dispatch", ".": "bubble", "?": "mixed"}
}
```

The `width` is configurable via CLI `--pipeline-width`. Default 80, valid
range 40..200. The per-stream view subsumes the v0.3 single-row "majority"
view (which can still be derived from the streams by taking the bucket
with max wall in each cell across all streams).

## 9. Failure modes the analyzer must handle gracefully

| Failure | Behavior |
|---|---|
| `trace_meta.json` missing | exit 2 USAGE; do NOT crash |
| Trace file is 0 bytes (flush race) | wait up to 30s; if still 0 bytes, exit 1 STAGE_FAILED with hint `re-run with profile enabled` |
| Trace truncated mid-iter (file < expected) | proceed with what we have; emit warning `trace_truncated`; coverage is best-effort |
| No `ProfilerStep#N` AND no Optimizer markers found | use wallclock fallback; emit warning `iter_boundary_fallback` |
| Single Optimizer marker only (one-iter trace without ProfilerStep) | DO NOT use it as iter; fall through to wallclock fallback (this was a v0.1 bug) |
| `pp > 1` but no p2p kernels in trace | emit warning `pp_layout_mismatch`; `pipeline.estimated_bubble_pct = null` |

## 10. Important known limitations

### 10a. RCCL collapses all collectives into one device kernel NAME, but the op type is in args
On ROCm, `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` is the
ONLY GPU kernel `name` for ALL collectives. **However**, the per-event
`args.Collective name` field (added by RCCL ≥ 2.18 / ROCm ≥ 6.0) directly
gives the operation type: `all_reduce`, `all_gather_into_tensor_coalesced`,
`reduce_scatter_tensor_coalesced`, `broadcast`, `alltoall`, `send`/`recv`.
And `args.Process Group Description` gives the parallel dimension:
`DATA_PARALLEL_GROUP_WITH_CP`, `TENSOR_MODEL_PARALLEL_GROUP`,
`PIPELINE_MODEL_PARALLEL_GROUP`, `EXPERT_PARALLEL_GROUP`, etc.

→ The analyzer SHOULD use `args.Collective name` to populate the
`comm_collective_*` sub-buckets (alltoall / allreduce / allgather /
reduce_scatter / broadcast / p2p). NO timestamp-based flow-event linking
needed. **This is a planned v0.4 feature**, not yet implemented; for now
`comm_collective` is one bucket on ROCm and the breakdown shows up only
in the CPU-side `nccl:*` user_annotation count distribution (printed at
the bottom of the report under `comm_collective_breakdown_unmatched`).

### 10b. NCCL annotation count >> kernel count
On the deepseek_v2_lite-FP8 baseline (1 iter, rank 0), there are ~205
`nccl:*` annotations but only 51 `ncclDevKernel_Generic_1` events. RCCL
fuses or batches some collectives at the device level. Do NOT assume
1:1 between annotation and kernel.

### 10c. `tensorboard_dir` overridden by Primus
`primus.backends.megatron.patches.args.tensorboard_path_patches.py` resets
`args.tensorboard_dir` to `<exp_root>/tensorboard` regardless of what we
inject. The `pilot.tools.profile.collect` step compensates by also
scanning `Primus/output/*/*/*/tensorboard` and symlinking the resulting
chrome-trace into `<run_dir>/profile/tb/`. Audit trail records the
override in `trace_meta.warnings`.

### 10d. `ProfilerStep#N` may not exist with `use_pytorch_profiler=False`
Some Primus configurations disable `use_pytorch_profiler` and use a custom
profiler. In that case, no `ProfilerStep` annotation is emitted and we
fall back to the Optimizer marker (priority 2). For a 1-iter capture,
that fallback won't work either, so capture at least 2 iters when
`use_pytorch_profiler` is False.

## 11. CLI contract

```
python -m pilot.tools.trace_analyze run \
    --trace-meta state/runs/<id>/profile/trace_meta.json \
    [--patterns  pilot/tools/_trace_patterns.json] \
    [--out-md    state/runs/<id>/trace_analysis.md] \
    [--out-json  state/runs/<id>/trace_analysis.json] \
    [--session   state/sessions/<sid>]      # if set, also write the session symlink
```

Exit codes: 0 success; 1 stage_failed (trace empty / corrupt past
recovery); 2 usage; 3 tool_error.

## 12. Iteration plan (status)

This skill v0.2 is **calibrated against `deepseek_v2_lite-FP8` on
MI355X**. Validated round-trips so far:

| Round | What we looked for | Outcome |
|---|---|---|
| Round 0 | run analyzer on the trace; check coverage | iter_boundary fallback was wrong (used single-Optimizer = 0.88ms); patterns missed RCCL collapse + deepep MoE |
| Round 1 | inspect top_unknown_names; update boundary detection + patterns | iter_wallclock 31999ms (matches Primus log 31987ms ± 0.04%), unknown_ratio 0.08%, MoE-EP comm correctly bucketed |
| Round 2 (v0.3) | report ergonomics: per-iter cost ms, top-N kernel ranking, end-to-end iter pipeline timeline | a single human-readable iter ms / wall % / kernel-count table at the top; cross-bucket top-20 kernels; ASCII pipeline showing forward/backward/optim phases |
| Round 3 (v0.4) | (a) make skill generic — drop pilot-specific section numbering / paths; (b) full kernel cost list (not just top-N); (c) per-stream pipeline rows (one per CUDA stream); (d) compute / comm / overlap / bubble decomposition that exactly partitions iter wallclock | report is now reusable in any consumer (not just pilot); kernel-list cap configurable; pipeline shows which stream did what; cost decomposition has zero double-counting |
| Round 4 (this round, v0.5) | (a) split iter wallclock into fwd / bwd / post_bwd_comm / optim phases via PyTorch autograd-engine markers; (b) per-bucket roll-up table grows an `avg_ms/kernel` column and per-phase sub-tables | header chip now answers "where does the iter time go?"; per-phase sub-tables answer "which bucket dominates each phase" (e.g. MoE comm density is 13.9% of fwd vs 7.2% overall — diagnosable insight) |
| Round 5 (todo) | use `ncclDevKernel*.args.Collective name` to split `comm_collective` into alltoall / allreduce / allgather / reduce_scatter / broadcast / p2p sub-buckets, AND tag by `args.Process Group Description` (TP / DP / EP / PP) | needed so that consumers can tell TP-bound vs DP-bound vs EP-bound |
| Round 6 (todo) | run on a TP/DP-only baseline (no MoE) | confirm `comm_moe_dispatch` bucket is empty and `comm_collective_allreduce` populates instead |
| Round 7 (todo) | run on a `pp > 1` config | confirm `comm_p2p` populated, pipeline timeline shows clear bubble pattern |
