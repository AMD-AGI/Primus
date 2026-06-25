# 01 — Overview & methodology

## Goal

Project DeepSeek-V4 (Flash / Pro) training throughput (iteration time,
TFLOP/s/GPU, tokens/s/GPU) for a real multi-hundred-GPU job, from a small set of
**single-layer** MI355X traces, then scale the result to MI455X by theoretical
hardware ratios.

## Why single-layer, per-cr traces

DeepSeek-V4 layers come in three attention flavours selected by the per-layer
`compress_ratio` (`cr`):

| cr   | attention branch                              |
|------|-----------------------------------------------|
| 0    | dense + sliding-window attention (SWA)         |
| 4    | CSA (compressed sparse attention, top-k via Indexer) |
| 128  | HCA (compressed attention, full pool visibility) |

The MoE block is **identical across all `cr`** (cr only changes attention). So
three single-layer traces — one per cr — fully characterise the per-layer cost,
and the full model is `Σ over layers (attention[cr_of_layer] + moe)`.

Profiling a *single* layer per cr (instead of a full model) keeps:
- **memory** within the MI355X budget at the production `seq=4096`, and
- **attribution clean**: with one cr in the trace, the dense attention kernels
  (which are shared across cr types in a multi-cr run) belong unambiguously to
  that cr.

## What we measure vs what we model

**Measured on MI355X (from traces):**
- per-module forward / backward time for one layer of each cr,
- TFLOPs for the three compute-bound kernel classes (`gemm`, `grouped_gemm`,
  `attn`); everything else is treated as memory-bound,
- embedding, output/logits, loss (the non-layer parts), taken once,
- the optimizer-step cost per unit parameter (for scaling).

**Modeled on top (in the website):**
- full layer count and exact `cr` schedule (Flash 43L, Pro 61L),
- PP / VPP partitioning -> per-stage critical path + pipeline bubble,
- EP dispatch/combine cost (no overlap in current stack),
- gradient-accumulation (GA) and DP behaviour (DP/PP comm assumed hidden),
- activation recompute (add a forward pass to the recomputed layers' backward),
- optimizer step scaled to per-rank parameter count for the target sharding,
- MI455X = MI355X breakdown scaled by compute / memory-bandwidth ratios.

## Capture configuration (the projection trace)

The profiling script (`script/deepseek_v4_layer_trace-projection.sh`) deliberately
differs from `run_deepseek_v4_pro_muon.sh`:

| knob              | projection value | why |
|-------------------|------------------|-----|
| `seq_length`      | 4096             | production per-microbatch token count |
| `num_layers`      | 1                | clean per-layer attribution; fits memory at seq 4096 |
| `compress_ratios` | `[CR]`           | one cr per trace |
| optimizer         | `adam` + distributed optimizer | compute is optimizer-independent; dist-opt = zero1 |
| overlap grad/param | off            | num_layers=1 breaks Megatron's chained param-sync; compute already clean |
| `global_batch_size` | `2 * DP`       | GA=2 (see below) |
| recompute         | off (`recompute_num_layers 0`) | capture pure fwd / pure bwd |
| profiler          | on, `with_stack=True`, window iter 6->7 | map kernels -> nn.module |

### Why overlap-off + GA=2 + take min

We capture with the distributed optimizer's comm overlap **off**. Two reasons:

- with `num_layers=1`, Megatron's chained param-gather sync trips an assertion
  (`param_and_grad_buffer.start_param_sync`) when overlap is on; and
- with overlap off, **every microbatch's compute is already overlap-free** —
  exactly the clean per-kernel time we want. In a real run GA is large so the
  vast majority of microbatches are clean anyway, and the projection assumes DP
  comm is hidden (A2), so there is no need to measure the contaminated overlap.

GA=2 (two microbatches) plus a late steady profiler window lets the parser
compute per-microbatch time as `sum(kernel_durations) / num_microbatches`.
This keeps scalar control-flow stalls (for example Indexer top-k syncs) that the
full-model calibration shows are real per-layer costs, while avoiding warm-up
and autotune iterations.

## Non-overlap assumptions in the current stack

- **No MoE deepep-comm + grouped-gemm overlap.** dispatch + grouped_gemm +
  combine are summed directly.
- **EP dispatch/combine has no overlap** with compute; counted in full.
- **DP / PP comm assumed hidden** (only the PP bubble remains). Optimistic; first
  thing to revisit at calibration time.

See `02-assumptions.md` for the complete list, and `04-projection-math.md` for
the formulas.
