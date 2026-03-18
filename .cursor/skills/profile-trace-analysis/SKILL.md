---
name: profile-trace-analysis
description: Analyze PyTorch profiler trace JSON files from Megatron/Primus training runs. Use when the user wants to profile training performance, analyze GPU kernel timing, generate per-operator statistics, identify bottlenecks, or compare forward/backward execution order.
---

# Profile Trace Analysis

Parse PyTorch profiler `.pt.trace.json` files from Megatron/Primus training and produce structured performance reports.

## When to Use

- User runs training with `--profile True --use_pytorch_profiler True`
- User asks to analyze a trace file or profile result
- User wants to identify performance bottlenecks in training

## Step 1: Locate Trace File

```bash
find output/ -name "*.pt.trace.json" -ls 2>/dev/null | head -5
```

Trace files are typically at:
```
output/<team>/<user>/<exp_name>/tensorboard/*.pt.trace.json
```

## Step 2: Parse and Classify Kernels

Use this Python script to extract GPU kernel statistics. Run it **inside the training container** via `primus-cli slurm --single`.

### Kernel Classification Map

Map raw kernel names to human-readable operator categories:

| Pattern in kernel name | Category |
|----------------------|----------|
| `nccl` | NCCL |
| `fmha_fwd` | Attn-FWD |
| `fmha_bwd`, `FmhaBwdOGradDotO` | Attn-BWD |
| `deep_ep` | DeepEP |
| `rmsnorm_fwd` | RMSNorm-FWD |
| `rmsnorm_bwd` | RMSNorm-BWD |
| `rotary_fwd` | RoPE-FWD |
| `rotary_bwd` | RoPE-BWD |
| `cast_transpose` | FP8-CastTranspose |
| `topk` (case-insensitive) | MoE-TopK |
| `unpermute` | MoE-Unpermute |
| `permute` (not `unpermute`) | MoE-Permute |
| `Cijk` + `F8` | GEMM-FP8 |
| `Cijk` (no `F8`) | GEMM-BF16 |
| `multi_tensor_apply` | Optimizer |
| `triton` | Triton-Fused |
| `fillBuffer` | MemSet |
| `MEMORY_COPY` | MemCopy |
| `vectorized`, `elementwise` | Elementwise |
| everything else | Other |

### Phase Detection

Use `ProfilerStep#N` events to find the profiled step boundary, then use CPU annotations to split Forward vs Backward:

- `forward_step` in `schedules.py` → Forward phase
- `backward_step` in `schedules.py` → Backward phase
- Everything else → Other (optimizer, grad reduce, PP comm)

## Step 3: Generate Reports

Produce **five** reports and save all to `<exp_dir>/profile_analysis.md`.

### Report Header (MUST use this exact format)

Parse the trace filename to extract experiment config. The filename pattern is:
```
primus-megatron-exp[<exp_name>]-rank[<rank>].<trace_id>.pt.trace.json
```

The `<exp_name>` contains hyphen-separated key-value pairs like `dsv3-pretrain-mbs_2-gbs_512-PP_4-EP_8-VPP_1-...`. Extract known keys (`PP`, `EP`, `MBS`/`mbs`, `GBS`/`gbs`, `VPP`/`vpp`, `turbodeepep`, `legacygg`, `recompute_num_layers`, `profile`) and format exactly as:

```markdown
# Profile Analysis

| Item | Value |
|------|-------|
| Trace | `<full_trace_filename>` |
| Step | ProfilerStep#N (X.XX ms) |
| PP | 4 |
| EP | 8 |
| MBS | 2 |
| GBS | 512 |
| VPP | 1 |
| TurboDeepEP | True |
| LegacyGG | True |
| Recompute layers | 0 |

---
```

Rules:
- Title is always `# Profile Analysis`, no experiment name appended
- Config table uses `| Item | Value |` columns, one row per known config key
- Only include keys that exist in the filename
- Step row shows the ProfilerStep name and its wall duration in ms
- Separator `---` after the table before Report A

### Report A: Phase Summary Table

**This is the first thing readers see after the header — must give an instant high-level picture.**

Report A is always the first report, placed immediately after the header `---`. Format exactly as:

```markdown
# Report A: Phase Summary

| Phase | Time (ms) | % |
|-------|-----------|---|
| Forward | X.XX | X.X% |
| Backward | X.XX | X.X% |
| Other | X.XX | X.X% |
| **Total** | **X.XX** | **100%** |

- Forward microbatches: N
- Backward microbatches: N
- Total GPU kernel time: X.XX ms
- Step wall time (ProfilerStep): X.XX ms
- GPU utilization (kernel / wall): X.X%
```

Rules:
- Total row uses bold (`**`), all other rows plain
- Times in ms with 2 decimal places, percentages with 1 decimal
- Bullet list below the table gives context: microbatch counts, kernel vs wall time, GPU utilization
- GPU utilization = total kernel time / step wall time × 100% (can exceed 100% due to multi-stream overlap)

### Report B: Forward Layer Analysis

Extract one **clean** forward transformer layer (skip the first layer which may have PP warmup). Merge fine-grained kernel groups into **semantic operators** — the goal is ~15-20 rows, not 70+.

#### Operator Merging Rules — Derived from Trace Module Hierarchy

The trace contains CPU annotations (`python_function` / `nn.Module:` events) that reveal the Megatron module tree. Use this hierarchy to map GPU kernels to semantic operators.

**Step 1: Read the module tree from the trace.** The expected DSV3 structure is:

```
TransformerLayer
├── _forward_attention                              ← 2nd-level split
│   ├── RMSNorm (input_layernorm)
│   └── MLASelfAttention
│       ├── get_query_key_value_tensors
│       │   ├── TELinear (kv_a_proj)               ← KV down-projection
│       │   ├── TELinear (kv_b_proj)               ← KV up-projection
│       │   └── qkv_up_proj_and_rope_apply
│       │       ├── TELayerNormColumnParallelLinear (q_a_proj)   ← Q compress
│       │       ├── TELayerNormColumnParallelLinear (q_b_proj)   ← Q decompress
│       │       └── fused_mla_yarn_rope_apply ×2                ← RoPE
│       ├── TEDotProductAttention                  ← FlashAttention
│       └── TERowParallelLinear                    ← Output projection
└── _forward_mlp                                    ← 2nd-level split
    ├── RMSNorm (pre_mlp_layernorm)
    └── MoELayer
        ├── router_and_preprocess
        │   └── PrimusTopKRouter (routing)
        ├── dispatch → DeepEP(dispatch)
        ├── experts_compute
        │   ├── SharedExpertMLP                    ← Shared expert (overlapped with dispatch)
        │   ├── token_permute
        │   ├── GroupedMLP                         ← Expert FFN1 + SwiGLU + FFN2
        │   └── token_unpermute
        └── combine → DeepEP(combine)
```

**Step 2: Map kernels to modules.** For each kernel, find its parent CPU annotation (`nn.Module:` or `python_function`) to determine which module it belongs to. Then merge kernels by module into semantic operators.

**Step 3: Name operators using `Module(sub-op)` format.**

| Module path in trace | Merged operator name |
|---------------------|---------------------|
| `_forward_attention` → `RMSNorm` | RMSNorm(Attn) |
| `MLASelfAttention` → `get_query_key_value_tensors` | MLASelfAttention(QKV proj) — merge all MLA projection kernels |
| `MLASelfAttention` → `qkv_up_proj_and_rope_apply` → `rope_apply` | MLASelfAttention(RoPE) |
| `MLASelfAttention` → `TEDotProductAttention` | MLASelfAttention(FlashAttn) |
| `MLASelfAttention` → `TERowParallelLinear` | MLASelfAttention(O proj) |
| `_forward_mlp` → `RMSNorm` | RMSNorm(MoE) |
| `MoELayer` → `PrimusTopKRouter` | MoELayer(TopKRouter) |
| `MoELayer` → `dispatch` | MoELayer(dispatch) |
| `MoELayer` → `SharedExpertMLP` | MoELayer(SharedExpert) — if present |
| `MoELayer` → `token_permute` | MoELayer(token_permute) |
| `MoELayer` → `GroupedMLP` → FFN1 | MoELayer(GroupedMLP FFN1×N) |
| `MoELayer` → `GroupedMLP` → SwiGLU | MoELayer(GroupedMLP SwiGLU) |
| `MoELayer` → `GroupedMLP` → FFN2 | MoELayer(GroupedMLP FFN2×N) |
| `MoELayer` → `token_unpermute` | MoELayer(token_unpermute) |
| `MoELayer` → `combine` | MoELayer(combine) |
| Small Elementwise/Other/MemSet between major ops | Absorb into adjacent module or drop |

**Fallback**: If CPU annotations are missing, fall back to kernel category classification (GEMM-FP8/GEMM-BF16/Attn-FWD etc.) with positional heuristics.

#### Output Format (exact)

**1. Pipeline diagram** — use **tree format** mirroring the Megatron module hierarchy. Each row shows `#N` referencing the table below, and timing. Mark `[gap X.XXms]` for idle gaps > 0.5ms:

```
TransformerLayer
├── _forward_attention
│   ├── #1  RMSNorm(Attn)                     0.10 ms
│   └── MLASelfAttention
│       ├── #2  QKV proj                       0.80 ms   (8 kernels)
│       ├── #3  RoPE                           0.30 ms
│       ├── #4  FlashAttn                      1.28 ms   ★
│       └── #5  O proj                         0.73 ms
├── _forward_mlp
│   ├── #6  RMSNorm(MoE)                      0.10 ms
│   └── MoELayer
│       ├── #7  TopKRouter                     1.02 ms
│       ├── #8  dispatch (DeepEP)              0.79 ms
│       │       [gap 0.76ms — cross-node all-to-all]
│       ├── #9  token_permute                  0.17 ms
│       ├── GroupedMLP
│       │   ├── #10  FFN1×N                    6.30 ms   ★ 1.97× overlap
│       │   ├── #11  SwiGLU                    0.12 ms
│       │   └── #12  FFN2×N                    2.69 ms   ★ 1.97× overlap
│       ├── #13 token_unpermute                0.18 ms
│       └── #14 combine (DeepEP)               0.77 ms
```

Rules:
- Tree structure uses `├──` / `└──` / `│` box-drawing characters
- Each leaf node is a merged operator with `#N` referencing the statistics table
- Show kernel time in **us** (microseconds) right-aligned, `★` marks top-3 by time
- Module-level nodes (MLASelfAttention, MoELayer, GroupedMLP) are non-leaf groupings without `#N`
- `[gap X.XXms — cause]` inline after the operator that precedes the gap

**2. Per-Operator Statistics table:**

```markdown
| # | Operator | Kernels | Wall (us) | Kernel (us) | % | Overlap | Top Kernel |
|---|----------|---------|-----------|-------------|---|---------|------------|
```

Rules:
- **All times in us (microseconds)**, integer format (no decimals)
- Operator column uses semantic names from merging rules above
- Rows with ≥5% share: use **bold** for the entire row
- Overlap = kernel_sum / wall_time. Show as `X.XX×` if > 1.0, otherwise `-`
- **Top Kernel** column: show the raw kernel function name (truncated to ~50 chars) of the longest-running kernel in this merged operator — this helps verify correctness (e.g. `_ZN5aiter32fmha_fwd_hd192_hd128_bf16_causalE`)
- Total row at bottom with **bold**
- ~15-20 rows max per layer

**3. Overlap Analysis** section — explain multi-stream parallelism for Expert GEMM:

```markdown
## Overlap Analysis

- **FFN1 (N experts)**: kernel sum = X.XXms, wall = X.XXms → **X.XX× overlap**
- **FFN2 (N experts)**: kernel sum = X.XXms, wall = X.XXms → **X.XX× overlap**
```

**4. Idle Gaps** table — list gaps > 0.1ms between major operators with cause:

```markdown
## Idle Gaps

| Between | Gap (ms) | Cause |
|---------|----------|-------|
| MoE-TopK → DeepEP(dispatch) | X.XX | CPU scheduling + DeepEP buffer prep |
| DeepEP(dispatch) → MoE-Permute | X.XX | Cross-node all-to-all comm wait |
```

**5. Key Takeaways** — 3-5 numbered insights with percentages and actionable recommendations.

### Report C: Backward Layer Analysis

Same structure as Report B but for one clean backward layer. BWD reverses the module order:

```
TransformerLayer BWD (reversed)
├── _forward_mlp BWD
│   └── MoELayer BWD
│       ├── MoELayer(dispatch)                     ← re-dispatch for BWD
│       ├── MoELayer(token_permute)
│       ├── MoELayer(GroupedMLP dFFN2×2N)          ← 2N kernels (dX + dW)
│       ├── MoELayer(GroupedMLP SwiGLU BWD)
│       ├── MoELayer(GroupedMLP dFFN1×2N)          ← 2N kernels (dX + dW)
│       ├── MoELayer(grad_acc)
│       ├── MoELayer(token_unpermute)
│       ├── MoELayer(combine)
│       ├── MoELayer(Router wgrad)
│       └── RMSNorm-BWD(MoE)
├── _forward_attention BWD
│   ├── MLASelfAttention(O wgrad)
│   ├── MLASelfAttention(FlashAttn BWD)
│   ├── MLASelfAttention(RoPE BWD)
│   ├── MLASelfAttention(QKV wgrad)
│   └── RMSNorm-BWD(Attn)
```

Note: BWD has 2N expert GEMM kernels per layer (N for dX activation grad, N for dW weight grad).

Include: pipeline diagram, per-operator table, overlap analysis, idle gaps, key takeaways, and a **FWD vs BWD comparison table**:

```markdown
## FWD vs BWD Comparison

| Category | FWD (ms) | BWD (ms) | BWD/FWD |
|----------|----------|----------|---------|
```

Note: BWD has 2N expert GEMM kernels per layer (N for dX activation grad, N for dW weight grad).

## Step 4: Identify Bottlenecks and Optimizations

After generating reports, analyze and recommend:

| What to check | How to check | Possible action |
|---------------|-------------|-----------------|
| GEMM-BF16 dominance (>40%) | Expert FFN running in BF16 | Check if `moe_use_legacy_grouped_gemm` can be False (FP8 expert GEMM) |
| NCCL in Other phase (>15%) | PP send/recv not overlapped | Enable `overlap_p2p_comm_warmup_flush` |
| DeepEP idle gaps (>1ms between TopK and dispatch) | CPU scheduling overhead | Enable `turbo_sync_free_moe_stage >= 2` |
| Elementwise >5% | Many small unfused kernels | Check fusion flags: `bias_swiglu_fusion`, `moe_permute_fusion` |
| Expert GEMM overlap < 2× | Only 2-stream parallel | Check `use_turbo_grouped_mlp` |
| BWD Expert GEMM overlap << FWD | dX/dW serialization in BWD | Check grouped GEMM BWD stream config |

## Step 5: Generate Interactive HTML Timeline

Generate `<exp_dir>/layer_timeline.html` with **both Forward and Backward** single-layer timelines.

### HTML Style Specification (MUST follow exactly)

**Theme & Fonts:**
- Dark background: `#0a0e17`, text color: `#e0e6f0`
- Google Fonts: `JetBrains Mono` (headings, labels, monospace) + `DM Sans` (body)
- `@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=DM+Sans:wght@400;500;700&display=swap')`

**Color Palette for Categories:**
```javascript
const C = {
  norm: "#6366f1",   // RMSNorm — indigo
  cast: "#8b5cf6",   // FP8 Cast+Transpose — purple
  fp8:  "#06b6d4",   // GEMM FP8 — cyan
  rope: "#f59e0b",   // RoPE — amber
  attn: "#ef4444",   // Flash Attention — red
  bf16: "#10b981",   // GEMM BF16 (Expert) — emerald
  moe:  "#f97316",   // MoE Control (TopK/Permute/Unpermute) — orange
  ep:   "#ec4899",   // DeepEP Dispatch/Combine — pink
  tri:  "#a3e635"    // Triton Fused — lime
};
```

**Layout Structure:**
1. `<h1>` title — `JetBrains Mono`, 22px, bold, color `#7dd3fc`
2. `.sub` subtitle — `JetBrains Mono`, 13px, color `#64748b`
3. **Chart area** — rows of `.row` (flex, 34px height):
   - `.lbl` (210px wide, right-aligned, `JetBrains Mono` 11px, color `#94a3b8`)
   - `.bc` (flex:1, relative positioned container for bars)
   - `.bar` (absolute positioned, 26px height, rounded 4px, colored by category, hover brightness 1.3 + white outline)
   - `.gap` (hatched pattern: `repeating-linear-gradient(90deg, transparent 0 4px, #ffffff06 4px 8px)`)
4. **Time axis** — `.tick` labels (`JetBrains Mono` 10px, color `#475569`) + `.grid` lines (`#1e293b`)
5. **Legend** — flex row of color swatches (14px rounded squares) with labels
6. **Statistics table** — full width, `JetBrains Mono` 11px:
   - `th`: color `#7dd3fc`, border-bottom `2px solid #1e293b`
   - `td`: color `#cbd5e1`, border-bottom `1px solid #1e293b`
   - Columns: `#`, `Operator` (colored by category, bold), `Cnt`, `Wall(ms)`, `Kernel(ms)`, `%`, `Overlap`, inline bar `.bi`
7. **Notes** — 12px, color `#64748b`, highlight class `.hl` with color `#f59e0b` bold

**CSS Classes (exact):**
```css
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0e17;color:#e0e6f0;font-family:'DM Sans',sans-serif;padding:32px;min-height:100vh}
h1{font-family:'JetBrains Mono',monospace;font-size:22px;font-weight:700;color:#7dd3fc;margin-bottom:4px}
.sub{font-size:13px;color:#64748b;margin-bottom:28px;font-family:'JetBrains Mono',monospace}
.row{display:flex;align-items:center;margin-bottom:3px;height:34px}
.lbl{width:210px;min-width:210px;font-family:'JetBrains Mono',monospace;font-size:11px;text-align:right;padding-right:14px;color:#94a3b8;white-space:nowrap}
.bc{flex:1;position:relative;height:26px}
.bar{position:absolute;height:26px;border-radius:4px;display:flex;align-items:center;justify-content:center;font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:600;color:#fff;text-shadow:0 1px 2px rgba(0,0,0,.6);min-width:3px;cursor:default;transition:filter .1s}
.bar:hover{filter:brightness(1.3);outline:2px solid #fff4}
.gap{position:absolute;height:26px;background:repeating-linear-gradient(90deg,transparent,transparent 4px,#ffffff06 4px,#ffffff06 8px);border-radius:3px}
.axis{display:flex;margin-top:10px;padding-left:210px;height:22px;position:relative}
.tick{position:absolute;font-family:'JetBrains Mono',monospace;font-size:10px;color:#475569;transform:translateX(-50%)}
.grid{position:absolute;width:1px;background:#1e293b;height:700px;bottom:0;transform:translateX(-50%);pointer-events:none;z-index:-1}
.legend{display:flex;flex-wrap:wrap;gap:14px;margin-top:24px;padding-left:210px}
.li{display:flex;align-items:center;gap:6px;font-size:12px;color:#94a3b8}
.ld{width:14px;height:14px;border-radius:3px}
table{margin-top:32px;border-collapse:collapse;width:100%;max-width:1000px;font-family:'JetBrains Mono',monospace;font-size:11px}
th{text-align:left;color:#7dd3fc;font-weight:600;padding:7px 10px;border-bottom:2px solid #1e293b}
td{padding:5px 10px;border-bottom:1px solid #1e293b;color:#cbd5e1}
tr:hover td{background:#1e293b44}
.bi{height:16px;border-radius:3px;min-width:1px}
.note{margin-top:20px;font-size:12px;color:#64748b;line-height:1.6;max-width:900px;padding-left:210px}
.note b{color:#94a3b8}
.hl{color:#f59e0b;font-weight:600}
```

**JavaScript Data Format:**
```javascript
// Each operator as object with: label, start(ms), end(ms), wall(ms), kernel_sum(ms), kernel_count, group_key
const S = [
  {l:"RMSNorm(Attn)", s:0.41, e:0.44, w:0.033, k:0.033, n:1, g:"norm"},
  // ... one entry per operator in execution order
];
```

**Rendering Logic:**
- Bar width = `((op.end - tMin) / (tMax - tMin)) * chartWidth` where `chartWidth = 750`
- Show `wall_ms` text inside bar if bar pixel width > 50
- Tooltip on hover: operator name, wall time, kernel sum, kernel count, overlap ratio
- Gap hatched rectangles between consecutive operators when gap > 0.02ms
- Axis ticks at integer ms values within the visible range
- Table total row in bold with `color: #7dd3fc`
- Notes section with overlap explanation and key findings using `.hl` highlight spans

**Tabbed Interface (FWD + BWD):**
- Add tab buttons above the chart with matching dark theme
- Active tab: `background: #1e293b`, `color: #7dd3fc`, `border-bottom: 2px solid #7dd3fc`
- Inactive tab: `background: transparent`, `color: #64748b`
- Each tab contains its own chart, axis, legend, table, and notes section

## Notes

- Trace files are large (500MB-1GB). Use streaming JSON parse or load entirely into memory.
- Always use the **first** `ProfilerStep` occurrence for analysis (typically step 6).
- For MoE models, Expert GEMM count = `num_local_experts` (e.g. 32 for DSV3 with EP=8).
- BWD has 2× expert GEMM kernels per layer (dX + dW) compared to FWD.
- BF16 GEMM kernel names start with `Cijk_A` (hipBLASLt format on AMD).
- FP8 GEMM kernel names contain `F8` (e.g. `Cijk_Alik_Bljk_F8BS`).
