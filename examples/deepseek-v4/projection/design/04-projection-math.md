# 04 — Projection math

This is the exact derivation the website implements (`site/assets/app.js`). All
inputs come from the breakdown JSON (`03-json-schema.md`); all assumptions are in
`02-assumptions.md`. Times in seconds unless noted.

Notation:
- `cr ∈ {0, 4, 128}`, `n[cr]` = number of layers of that cr (`cr_layer_counts`).
- A `<Breakdown>` has `forward` / `backward` module lists. Define
  `T(bd, phase) = Σ_module bd[phase][m].time` (sum of clean module times).

## Step 0 — per-layer and non-layer base times (MI355X)

For each cr:
```
layer_fwd[cr] = T(attention[cr], forward) + T(moe, forward)
layer_bwd[cr] = T(attention[cr], backward) + T(moe, backward)
```
EP dispatch/combine are included as memory-bound module rows inside `moe`
(A7/A8), so they are already in these sums — do not add `comm` again (the `comm`
field is informational for the UI).

Non-layer (taken once, A15):
```
emb_fwd = T(embedding, forward);  emb_bwd = T(embedding, backward)
out_fwd = T(output, forward) + T(loss, forward)
out_bwd = T(output, backward) + T(loss, backward)
```

MTP (if `mtp_num_layers > 0`, A17):
```
mtp_inner_fwd = layer_fwd[mtp_cr] ; mtp_inner_bwd = layer_bwd[mtp_cr]
mtp_out_fwd   = out_fwd          ; mtp_out_bwd   = out_bwd
mtp_eh_time   ≈ output_time * (mtp_eh_proj_flops / output_flops)

mtp_fwd = mtp_num_layers * (mtp_inner_fwd + mtp_out_fwd) + mtp_eh_time / 3
mtp_bwd = mtp_num_layers * (mtp_inner_bwd + mtp_out_bwd) + 2 * mtp_eh_time / 3
```
The current Flash Megatron FLOPs anchor uses `mtp_cr=4`; a future dedicated MTP
trace can replace the `eh_proj` and inner-layer approximation.

## Step 0b — manual layer-time mode (optional, UI-only)

The site exposes a **layer-timing mode** toggle in the controls panel:

- `trace` (default): `layer_fwd/bwd[cr]` come from the breakdown JSON exactly as
  in Step 0.
- `manual`: the user types `layer_fwd[cr]` / `layer_bwd[cr]` directly (one
  fwd/bwd pair per `cr ∈ {0,4,128}`, in µs) and those values replace the
  trace-derived per-layer times. This is a what-if calculator: you supply the
  per-layer cost and the site reuses Steps 1-5 unchanged to derive the full-model
  iteration time, bubble, optimizer, tokens/s and TFLOP/s.

Rules baked into the implementation:

- **Granularity is per-`cr`** (same as the data model — `cr` only changes
  attention, the MoE block is shared), not per physical layer. The `cr` schedule,
  PP/VPP layout and recompute still expand these per-cr times to the full model.
- **Per-GPU storage.** A hand-entered time already targets one GPU, so manual
  values are stored separately for MI355X and MI455X and the
  MI355→MI455 scaling (Step 6) is **bypassed** in manual mode. Switching the GPU
  tab edits that GPU's own set.
- **Prefill from trace.** Entering manual mode (or switching GPU within it)
  seeds any unset field with the current trace-derived value, so toggling never
  changes the result until you actually edit a number. Unset/blank fields keep
  falling back to trace.
- **Time only, FLOPs stay analytic.** Manual input overrides time but not FLOPs;
  `TFLOP/s/GPU` continues to use the V4 analytic model FLOPs (Step 5), so it
  stays meaningful.
- **Scope.** Manual covers the three per-cr decoder layers **and** the non-layer
  parts — embedding (PP stage 0), output / loss / MTP (last PP stage) — each as a
  fwd/bwd pair. Any field left unset falls back to its trace-derived value, so you
  can override just the parts you care about. FLOPs stay analytic regardless.

## Step 1 — recompute (A16)

If a layer is recomputed, its backward replays one forward:
```
layer_bwd_eff[cr] = layer_bwd[cr] + recompute_factor[cr] * layer_fwd[cr]
```
`recompute_factor` ∈ {0,1} per layer. The site exposes `none`, `full`, and
`first-n`; `first-n` adds one forward replay to the first N decoder layers owned
by each physical PP stage, matching the common Megatron
`recompute_num_layers=N` block pattern.

## Step 2 — map layers to PP stages / VPP chunks (A6)

Inputs: `PP`, `VPP`, optional `pipeline_layout`. Total model chunks
`C = PP * VPP`. If `pipeline_layout` is provided, parse Megatron-style `t` /
`t*N` stage specs (for example `Et*10|t*11|t*11|t*11mL`) and assign virtual
chunk `k` to device `k mod PP`. Otherwise build the ordered layer list from
`compress_ratios`, slice it into `C` contiguous chunks, and use the same
`k mod PP` mapping. The UI validates that an explicit layout has exactly
`PP*VPP` stages and exactly `num_layers` decoder layers; invalid layouts block
projection instead of silently falling back. For device `d`:
```
Df[d] = Σ_{chunks on d} Σ_{layer in chunk} layer_fwd[cr(layer)]
Db[d] = Σ_{chunks on d} Σ_{layer in chunk} layer_bwd_eff[cr(layer)]
```
Add non-layer parts to their devices:
```
Df[0]      += emb_fwd ;  Db[0]      += emb_bwd
Df[PP-1]   += out_fwd ;  Db[PP-1]   += out_bwd
Df[PP-1]   += mtp_fwd ;  Db[PP-1]   += mtp_bwd
```
Critical device:
```
Df_crit = max_d Df[d] ;  Db_crit = max_d Db[d]
```
(Using per-device max is an upper-bound for imbalanced stages; for a balanced
schedule it is exact.)

## Step 3 — pipeline iteration time (A4/A5)

With gradient accumulation `GA` microbatches and interleaved VPP, the steady
1F1B time on the critical device is:
```
pipe_compute = (GA + (PP - 1) / VPP) * (Df_crit + Db_crit)
```
- `GA * (Df_crit + Db_crit)` is the steady throughput term;
- `(PP-1)/VPP * (Df_crit + Db_crit)` is the bubble (fraction `(PP-1)/(GA*VPP)`).
- PP=1 ⇒ `pipe_compute = GA * (Df_crit + Db_crit)` (no bubble).

`GA = GBS / (DP * MBS)`. The site takes `GBS`, `MBS`, `DP` as inputs (or derives
`DP = world_size / (PP * TP * CP)` with EP ⊆ DP).

## Step 4 — optimizer step (A1/A3)

Per-iteration, once, zero1-sharded, memory-bound:
```
local_model_params  = total_params / (PP * TP)
per_rank_opt_params = local_model_params / DP
opt_bytes           = per_rank_opt_params * bytes_per_param
opt_time            = opt_bytes / hbm_bandwidth / opt_efficiency
```
`total_params` is computed from `model_config` for the full model (dense +
expert params + untied embedding/output). PP/TP determine the average local
model-parameter ownership; CP does not shard parameters. EP is represented in
the full expert count and cancels with the data-replica count for ZeRO-1
optimizer sharding, so the average optimizer shard is `total/(PP*TP*DP)`.
`bytes_per_param` is the full Adam mixed-precision step traffic (default 30B:
reads + writes), and `opt_efficiency` is tunable. The measured
`optimizer.time_us` carried in the JSON is displayed as a sanity reference.

## Step 5 — totals (A2/A4: DP & PP comm hidden)

```
iter_time = pipe_compute + opt_time
```
Throughput:
```
tokens_per_iter      = GBS * seq_length            (= GA * DP * MBS * seq)
tokens_per_s         = tokens_per_iter / iter_time
tokens_per_s_per_gpu = tokens_per_s / world_size
```
TFLOP/s/GPU (matmul-flops convention, A14): per-microbatch model compute FLOPs
```
F_mb = Σ_cr n[cr] * (flops_fwd[cr] + flops_bwd[cr]) + nonlayer_flops
       where flops_*[cr] = Σ_module (module.flops or 0) over the cr breakdown
       + mtp_inner + mtp_eh_proj + mtp_extra_logits + mtp_hc_head
flops_per_iter   = F_mb * GA * DP            (all microbatches, all DP replicas)
TFLOP_s_per_gpu  = flops_per_iter / iter_time / world_size / 1e12
```
`tokens_per_s_per_gpu` is the headline metric (independent of FLOP convention);
`TFLOP/s/GPU` is reported for comparison with Primus' own logging.

## Step 6 — MI455X scaling (A20-A23)

Rescale every module time before re-running Steps 0-5:
```
ratio_compute = peak_tflops_bf16[MI355] / peak_tflops_bf16[MI455]
ratio_memory  = hbm_bandwidth[MI355]   / hbm_bandwidth[MI455]

time'(module) = module.time * ratio_compute / compute_efficiency   if compute_bound
              = module.time * ratio_memory                          if memory_bound
```
`compute_efficiency` (default 1.0) is the MFU knob (A22). FLOPs are unchanged
(same math), so MI455 `tflops` rises by `1/ratio_compute * compute_efficiency`.
Optimizer scales by `ratio_memory`. Comm not rescaled (A23).

## Step 7 — self-consistency check (A24)

Configure `PP=1, VPP=1, EP=8, DP=1, MBS=1, GA=GA_capture` and confirm the model's
`iter_time` matches the measured single-node iteration time within tolerance. The
site shows this check on the MI355X page when capture metadata is present.

## Worked control set (website inputs)

GPU page (MI355X / MI455X), then: `world_size`, `PP`, `VPP`, `EP`, `DP` (or
derive), `CP`, `TP`, `MBS`, `GBS` (or `GA`), recompute mode, and the tunables
`opt_efficiency`, `compute_efficiency`, `bytes_per_param`. Every intermediate
(`layer_fwd/bwd`, `Df/Db` per stage, `pipe_compute`, bubble %, `opt_time`,
`iter_time`, `tokens/s/gpu`, `TFLOP/s/gpu`) is displayed step by step.
