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

## Step 1 — recompute (A16)

If a layer is recomputed, its backward replays one forward:
```
layer_bwd_eff[cr] = layer_bwd[cr] + recompute_factor[cr] * layer_fwd[cr]
```
`recompute_factor` ∈ {0,1} per cr (site control: none / full / selective). v1
exposes none | full; full sets it to 1 for all cr.

## Step 2 — map layers to PP stages / VPP chunks (A6)

Inputs: `PP`, `VPP`. Total model chunks `C = PP * VPP`, layers per chunk
`Lc = num_layers / C` (assume divisible; otherwise distribute remainder to the
first chunks). Build the ordered layer list from `compress_ratios`, slice it into
`C` contiguous chunks (Megatron interleaved assigns chunk `k` to device
`k mod PP`). For device `d`:
```
Df[d] = Σ_{chunks on d} Σ_{layer in chunk} layer_fwd[cr(layer)]
Db[d] = Σ_{chunks on d} Σ_{layer in chunk} layer_bwd_eff[cr(layer)]
```
Add non-layer parts to their devices:
```
Df[0]      += emb_fwd ;  Db[0]      += emb_bwd
Df[PP-1]   += out_fwd ;  Db[PP-1]   += out_bwd
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
per_rank_opt_params = total_params / DP
opt_bytes           = per_rank_opt_params * bytes_per_param * rw_factor
opt_time            = opt_bytes / hbm_bandwidth / opt_efficiency
```
`total_params` is computed from `model_config` (dense + expert params; experts
are EP-sharded so per-rank expert params = experts/EP). `bytes_per_param`,
`rw_factor` (read+write ≈ 2), `opt_efficiency` are tunable; the measured
`optimizer.time_us` is used to calibrate `opt_time_per_param` for the MI355X
page and as a sanity check.

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
