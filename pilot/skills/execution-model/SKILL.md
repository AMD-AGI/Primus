---
name: execution-model
description: Closed-form formulas to estimate per-step compute time, communication time, pipeline bubble, overlap, and peak memory of a Primus training plan. Use whenever you need to predict TPS / step_time / mem_peak before running a candidate, sanity-check a measured metric against expectation, decide if a candidate fits memory before launching it, or compare two parallelism shapes on paper. Keywords: execution model, T_step, T_comp, T_comm, T_bubble, T_overlap, mem estimate, OOM prediction, MFU, peak TFLOPS, allreduce time, alltoall time.
---

# Execution Model — Step-Time and Memory Formulas

A small set of closed-form formulas the agent uses to predict `T_step` and `Mem_peak` for a candidate plan, *before* burning a GPU·h on it. The formulas are coarse on purpose: their job is to (1) reject obviously-bad candidates and (2) give `expected_gain` numbers for ranking. Real numbers come from the `run-and-profile` subagent.

The cluster constants (`peak_tflops`, `bandwidth_gbs`) come from the `preflight` skill's output (`output/pilot/cluster-<id>.md`). Never hardcode hardware peaks in chat.

## Step Time Decomposition

```
T_step = T_comp + T_comm + T_bubble - T_overlap
```

### T_comp — compute time

```
T_comp = model_flops_per_step / (num_gpus × peak_tflops × η_comp)

model_flops_per_step ≈ 6 × P × tokens_per_step                # dense
                     ≈ 6 × P_active × tokens_per_step          # MoE, where P_active = embedding + non-MoE + topk × per-expert
tokens_per_step      = gbs × seq_len
P                    = total parameters
```

`η_comp` is the achieved fraction of peak (MFU); rule of thumb:

| Setup | Typical η_comp |
|---|---|
| Dense, BF16, well-tuned, single-node | 0.45 – 0.60 |
| Dense, BF16, 8–16 nodes | 0.35 – 0.50 |
| MoE, BF16, well-tuned | 0.25 – 0.40 |
| Anything fp8 with mature kernels | +5–10 pp on top |

Use the BASELINE run to back-fit `η_comp`; subsequent candidate predictions reuse the fitted value.

### T_comm — communication time

```
T_comm = T_AR(grad_size / dp_shard) + T_AT(moe_msg) + T_AG(zero_shard) + T_RS(zero_shard)
```

For each collective `T_coll(bytes)`:

```
T_coll(B) ≈ α_coll + B / BW_eff(scope, coll, B)
```

| Scope (chosen by parallelism shape) | When |
|---|---|
| `intra_node` | TP / EP groups within one node |
| `inter_node` | DP / FSDP groups across nodes (when intra-node is excluded) |
| `world` | Mixed shape that uses the full N×gpus_per_node ring |

Look up `BW_eff` from `cluster.rccl_baseline.<scope>.collectives.<coll>.roll_up.median_bw_gbs` at the message-size bucket nearest `B`. For ranking candidates, use `median_bw_gbs`; for OOM-style worst-case bounds, use `min_bw_gbs`.

Common message sizes:

| Collective | Bytes |
|---|---|
| AllReduce(grad) per DP step (no ZeRO) | 2 × P × bytes_per_param / dp |
| ReduceScatter+AllGather (ZeRO-1/2) | 2 × P × bytes_per_param / dp (each half) |
| AllToAll (MoE dispatch) per layer | 2 × hidden × topk × tokens_per_rank × bytes_per_act |
| AllGather (TP weight) per layer | hidden × hidden / tp × bytes_per_param |

### T_bubble — pipeline bubble

```
T_bubble = (pp - 1) / (pp - 1 + M) × T_comp_per_microbatch_chain
M        = gbs / (mbs × dp)               # number of microbatches per stage
```

When VPP (interleaved 1F1B with `vpp` virtual stages per device) is on:

```
T_bubble_vpp ≈ T_bubble / vpp
```

Hard limits:

- `pp = 1` ⇒ `T_bubble = 0`
- `M < pp` ⇒ severe bubble; reject the candidate (too few microbatches for the pipeline depth)

### T_overlap — overlapped slack

```
T_overlap = min(T_comm_overlappable, T_comp_spare)
```

Overlappable comms (when the corresponding flag is on):

| Comm | Overlappable when |
|---|---|
| DP gradient AllReduce | `--overlap_grad_reduce True` |
| ZeRO param AllGather | `--overlap_param_gather True` |
| MoE AllToAll | dispatch overlap impl present (model-side) |
| TP weight AllGather | sequence-parallel + async TP enabled |

`T_comp_spare` ≈ `T_comp × (1 - α_comp_busy)`; use `α_comp_busy ≈ 0.85` as a default cap.

## Memory Estimation

```
Mem = M_param + M_grad + M_optim + M_act + M_buffer
```

| Term | Formula |
|---|---|
| `M_param` | `P × bytes_per_param / (tp × pp × ep_for_experts)` |
| `M_grad`  | same as `M_param` (BF16 grad) |
| `M_optim` | Adam = 2 × FP32 × params shard = `8 × P / dp × {1 if ZeRO-1 else dp_factor}` |
| `M_act`   | `f(seq, hidden, mbs, layers/pp, recompute) × bytes_per_act` |
| `M_buffer` | nccl/comm buffers + workspace ≈ 2–8 GB |

Activation memory rough form (per-stage, one microbatch in flight):

```
M_act_one ≈ k_act × layers_per_stage × seq × hidden × mbs × bytes_per_act
```

`k_act` defaults:

| Recompute | k_act |
|---|---|
| `none` | 6–10 (model-dependent) |
| `selective` | 2–4 |
| `full` | 1 (only the input checkpoint) |

For 1F1B with VPP, the in-flight count is `≈ pp × vpp`, so:

```
M_act_total ≈ M_act_one × min(pp × vpp, M)
```

### OOM gate

Reject any candidate where:

```
predicted_Mem_peak > 0.92 × hbm_capacity_gb
```

(0.92 leaves headroom for fragmentation + temporary kernel allocations.)

## Calibration: how the agent fits the constants

The five free constants (`η_comp`, `α_overlap`, `α_comp_busy`, `k_act`, `α_coll` per collective) are fit from the BASELINE run, not guessed:

1. After BASELINE finishes, take measured `tps`, `step_time_ms`, `comm_ratio`, `bubble_ratio`, `mem_peak_gb`.
2. `η_comp_fit = (1 - comm_ratio - bubble_ratio + overlap_ratio) × T_comp_predicted_at_η=1 / step_time_ms`
3. `k_act_fit = mem_peak_gb − (M_param + M_grad + M_optim + M_buffer) / per-microbatch activation footprint`
4. Carry the fitted values forward; recalibrate when champion changes by > 1.5× tps (different operating regime).

## Worked example — Dense Llama 8B, 16 nodes, MI300X

Constants from preflight: `peak_tflops_bf16 = 1300`, `world_AR median_bw_gbs[256MB] = 22.3`, `hbm = 192 GB`.

Plan: tp=8 pp=1 dp=16 mbs=2 gbs=512 seq=8192 vpp=1 recompute=selective overlap=true

```
P = 8e9
tokens_per_step = 512 × 8192 = 4.2e6
model_flops = 6 × 8e9 × 4.2e6 = 2.0e17 FLOPs
T_comp_predicted_at_η=1 = 2.0e17 / (128 × 1300e12) = 1.20 s
With η_comp = 0.45:    T_comp = 2.67 s
T_AR(grad) = 2 × 8e9 × 2 / 16 = 2 GB → @22.3 GB/s ≈ 90 ms (single AR)
T_comm ≈ 90 ms
T_bubble = 0 (pp=1)
T_overlap ≈ min(90, 2670 × 0.15) = 90 ms (fully overlappable)
T_step ≈ 2.67 + 0.09 - 0.09 = 2.67 s
predicted_tps = tokens_per_step / T_step / num_gpus
             = 4.2e6 / 2.67 / 128 ≈ 12300 tok/s/GPU
```

Candidate: same but mbs=4 (reduce M from 32 to 16)

```
T_comp roughly unchanged (kernels bigger but more efficient → η_comp +5%)
M_act doubles → check predicted_Mem_peak ≤ 0.92 × 192 ?
If yes → predicted gain ≈ +5%, run it.
```

## Important Notes

- The model is a **ranker**, not a precise predictor. Aim for ±20% on `T_step`, ±15% on `Mem_peak`. A 5% predicted gain is noise; a 30% predicted gain is a confident bet.
- **Always require a measured BASELINE before trusting predictions**. Pre-baseline predictions use the table defaults and have ±50% error — useful only for spotting impossible candidates (e.g. predicted Mem 400 GB on a 192 GB GPU).
- **Confidence drops near regime boundaries**: at low `M` (close to `pp`), bubble dominates non-linearly; at very large `gbs`, comm scales differently. Mark such candidates with `confidence ≤ 0.6`.
- **Model is NOT for kernel-level decisions**. It does not know about flash-attn version, fusion impl, tile size. Those go in `optimize-compute` and are searched empirically.
- **Cite the formula in the candidate's `notes`**: e.g. "predicted +6% from T_bubble: (pp-1)/(pp-1+M) drops 0.18→0.09 with vpp 1→2". Makes the audit trail self-explaining.
