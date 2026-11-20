# MoE expert replication (dynamic redundant experts)

With expert parallelism, every EP rank waits for the busiest one: the dispatch, the expert GEMMs and the combine of a MoE layer all finish when the rank with the most routed tokens finishes. Expert replication gives each rank a few **replica slots**. Before the experts run, a planner copies the hottest experts into slots on lightly loaded ranks and moves part of their tokens there; after the backward, the replicas' weight gradients go back to the owning rank. Routing, the router's statistics and the optimizer are unchanged: a replica is a temporary copy, not a new parameter, and it has no optimizer state.

It pays off when the per-rank load is uneven and stays uneven long enough to amortize moving the weights. On a DeepSeek-V3 4-layer proxy at EP8 on MI355X it makes a step 21% faster under heavily skewed synthetic routing (16% with the DeepSeek expert bias on) and 1.6% faster with real DeepSeek-R1 routing, whose imbalance at EP8 is mild and changes every micro-batch ([Results](#results)). The imbalance of real routing grows with EP size, which is where replication is expected to help most. It is off by default.

Design discussion and the alternatives considered: [AMD-AGI/Primus#1502](https://github.com/AMD-AGI/Primus/issues/1502).

## Prerequisites

- MoE layers on Primus-Turbo: the DeepEP token dispatcher (`use_turbo_deepep: true`) and the Turbo grouped GEMM (`use_turbo_grouped_gemm: true`, `moe_grouped_gemm: true`), `turbo_grouped_gemm_backend` `default` or `triton`.
- A Primus-Turbo whose `grouped_gemm` takes the weights as a list of segments ([AMD-AGI/Primus-Turbo#675](https://github.com/AMD-AGI/Primus-Turbo/pull/675)). With an older Primus-Turbo, enabling replication raises an error at startup; rebuild Primus-Turbo from a commit that has it (the Primus CI image is built from the commit pinned as `PRIMUS_TURBO_COMMIT` in `.github/workflows/ci.yaml`).
- BF16 experts, tensor and expert tensor parallel size 1, pipeline parallel size 1, no token dropping.

Combinations that are rejected at startup (`rocm_arg_validation.py`), each with the reason in the error message: MegaMoE (`use_turbo_mega_moe`), FP8 / FP4 experts and the fused grouped MLP (`turbo_fused_grouped_gemm`), `moe_expert_capacity_factor`, `moe_router_force_load_balancing_type: even`, `overlap_moe_expert_parallel_comm`, `delay_wgrad_compute`, Megatron FSDP and torch FSDP2, CUDA graphs, the deprecated MoE layer (`use_deprecated_20241209_moe_layer`), and `overlap_grad_reduce` (the expert gradient buckets are reduced during the backward, before the replica gradients reach their owners). Activation recompute and `turbo_sync_free_moe_stage` 2 and 3 work with it (planning still reads the routed counts on the host once per plan).

## Design

```
router: top-k over E logical experts (aux loss, expert bias and their counts stay logical)
  |
  v
ReplicaScheduler (one per MoE layer, in the DeepEP token dispatcher)
  |-- planner     per-rank counts [R, E] -> slot table [R, B] + rows each replica takes
  |-- transport   push slot weights to the replicas, return slot gradients to the owners (NCCL point to point)
  '-- storage     B slots per rank and expert linear: weights, and a main_grad in the owners' dtype
  |
  v
remap: logical expert e -> physical expert p; each rank owns L = E / R experts followed by B slots
  |
  v
DeepEP dispatch over R * (L + B) physical experts (destination rank p // (L + B); kernels unchanged)
  |
  v
expert GEMMs: one segmented grouped_gemm over [own experts (L), slots (B)]
  |
  v
DeepEP combine (unchanged)
```

### Physical experts and the remap

Each rank owns `L = E / R` experts and holds `B` slots, so DeepEP is built for `R * (L + B)` physical experts and finds a token's destination rank as `p // (L + B)`, without kernel changes. After the router picks its top-k logical experts, the scheduler rewrites each `(token, expert)` route to the owner's copy or to one replica, following the plan. Probabilities stay attached to their routes, so the combine is unchanged. When a plan builds no replica, every route stays on its owner.

The split of an expert's routes among its owner and replicas is deterministic: routes are numbered in a global order (source rank, then token) and cut at the planned row counts. Every rank computes the same split from the same counts, with no extra communication (as MoonEP and UltraEP do).

### Planners

The planner sees the per-rank route counts of the layer, `[R, E]`, gathered with one all-gather, and returns which expert each slot holds and how many rows each copy takes.

| Planner | What it does | Origin |
| --- | --- | --- |
| `ultraep` | Approximate balance with quotas: binary-searches the smallest per-rank load cap, between the mean load and the current busiest rank, for which every overloaded rank can shed the rows above the cap into free slots on other ranks, with at least `moe_replica_min_rows` rows per new replica. Builds no replica unless the busiest rank drops by at least `moe_replica_min_gain` of the mean load. Optional per-replica cost in rows (`moe_replica_cost`, the `max(load + lambda * replicas)` objective of MegaFlux) and quotas in multiples of a GEMM tile (`moe_replica_block`). | UltraEP |
| `moonep` | Exact balance: every rank computes exactly the mean load (`S * K` rows with S tokens per rank and top-K, when all ranks route the same number of tokens). Each rank's replicas come from one overloaded rank, so it needs at least `L` slots. | MoonEP |

### When to plan: placement

- **`per_step`** (default): plan at the first micro-batch of each optimizer step and keep the slots for the step. Later micro-batches re-split each expert's rows among its owner and replicas in the first plan's proportions, on the device, so no count goes back to the host. Weights move once per layer and step.
- **`per_microbatch`**: plan every micro-batch (MoonEP's original use). It follows routing that changes every micro-batch but moves weights much more often. `moe_replica_replan_above` makes it sticky: a micro-batch first re-splits its rows among the current replicas and keeps them while the busiest rank stays within that multiple of the mean load; above it, it plans again and pushes only the changed slots.

### Weights, gradients and the expert GEMM

- **Expert GEMM.** The layer calls Primus-Turbo `grouped_gemm(x, [weights, slot_weights], m_splits)` once per pass: the own experts and the slots are two segments of one launch, over the rows of one input in the remapped order. Forward, dgrad and wgrad each stay a single kernel.
- **Slot gradients.** The slot weights carry a `main_grad` in the dtype of the owners' `main_grad` (`main_grads_dtype`), so with `turbo_grouped_gemm_wgrad_accum_fusion` the wgrad GEMM accumulates into both segments' `main_grad` in the same launch; without it, a hook adds the slot gradients into their `main_grad`.
- **Transfers.** Slot weights are pushed with NCCL point-to-point right after planning and waited for just before the expert GEMM. Slot gradients are returned in the backward of the step's last micro-batch and waited for at the end of the step, before `finalize_model_grads` (a patch wraps it); the owner adds them into its `main_grad` in source-rank order. `moe_replica_async_comm: false` makes both synchronous at their point of use.
- **Storage.** Every expert linear of every MoE layer keeps its own `B` slots, allocated right after the model is built and wrapped (the log reports their size): per slot, one expert's weights and as much again for their gradients. The slots are plain tensors, not parameters or buffers, so checkpoints do not change. See [Memory](#memory).

### Correctness rules

- **Router statistics stay logical.** The expert bias update and the aux loss use the router's counts over logical experts.
- **Collectives depend only on state all ranks agree on** (the slot tables, whether a plan has replicas), never on whether one rank happens to have rows or gradients; otherwise point-to-point transfers wait on each other forever.
- **Activation recompute replays the remap.** Megatron's first forward under checkpointing runs without grad; the scheduler tells training from evaluation by the `training` flag of the expert modules, plans in that forward and replays the same remap in the recompute.
- **A slot returns its gradients before it changes hands.** When a new plan gives a slot another expert or leaves it empty, the gradients it has collected go back to their owner first; otherwise they would be lost and later added to the slot's next expert.
- **Every step ends with the gradient return.** If a step's micro-batches outnumber `get_num_microbatches()` (the wrapped `finalize_model_grads` did not run), the scheduler raises instead of dropping replica gradients.

## Configuration

Replication is off by default. Arguments (in the model's `overrides`, defaults in `primus/configs/models/megatron/language_model.yaml`):

| Argument | Default | Meaning |
| --- | --- | --- |
| `moe_replica_planner` | `null` (off) | `ultraep` or `moonep` |
| `moe_replica_placement` | `per_step` | `per_step` or `per_microbatch` |
| `moe_replica_slots` | `4` | Slots per rank and expert linear (`moonep` needs at least `num_experts / expert_model_parallel_size`) |
| `moe_replica_min_rows` | `1024` | `ultraep`: rows a new replica must take |
| `moe_replica_min_gain` | `0.1` | `ultraep`: build replicas only if the busiest rank drops by this fraction of the mean load |
| `moe_replica_cost` | `0` | `ultraep`: rows a new replica costs its rank |
| `moe_replica_block` | `1` | `ultraep`: quotas in multiples of this many rows (GEMM tile rows) |
| `moe_replica_replan_above` | `0` | `per_microbatch`: keep the slots while the re-split busiest rank stays within this multiple of the mean load; `0` plans every micro-batch |
| `moe_replica_async_comm` | `true` | Overlap the weight push and the gradient return; `false` makes both synchronous at their point of use |
| `moe_replica_log_interval` | `0` | Log the busiest rank's load before and after planning every N plans |

The defaults were chosen on the DeepSeek-V3 proxy of [Results](#results) (MI355X, EP8, 4 micro-batches per step) with an earlier version of this code. `per_step`, `min_gain`, `cost`, `block` and `async_comm` were compared there; `slots` and `min_rows` follow UltraEP (2-4 slots per rank) and were not swept.

### Memory

The slots take, on every GPU:

```
MoE layers x moe_replica_slots x (fc1 + fc2 elements of one expert) x (weight bytes + main_grad bytes)
```

For DeepSeek-V3 (hidden 7168, expert FFN 2048 with SwiGLU, so 44.0 M elements per expert; BF16 weights and `main_grads_dtype: bf16`, 176 MB per slot and layer):

| Model | MoE layers | `moe_replica_slots` | Slot memory per GPU |
| --- | ---: | ---: | ---: |
| 4-layer proxy | 4 | 4 | 2.8 GB (0.7 GB per layer) |
| Full DeepSeek-V3 | 58 | 4 | 40.9 GB |
| Full DeepSeek-V3, `moonep` at EP8 (needs `slots >= num_experts / EP = 32`) | 58 | 32 | 327 GB |

`moonep` with its one-slot-per-local-expert requirement therefore does not fit a full-depth model on one GPU (MI355X has 288 GB); it is meant for shallow models until slots can be shared across layers. The startup check logs the estimate and stops when it exceeds the GPU's memory.

Example (DeepSeek-V3 proxy, EP8, one node):

```yaml
overrides:
  use_turbo_deepep: true
  use_turbo_grouped_gemm: true
  moe_replica_planner: ultraep
  moe_replica_placement: per_step
  moe_replica_slots: 4
```

## Results

8 x MI355X, DeepSeek-V3 4-layer proxy, EP8, 256 experts, top-8, BF16 (`main_grads_dtype: bf16`), sequence 4096, MBS 2, GBS 64 (8192 tokens per rank per micro-batch, 4 micro-batches per step), Turbo DeepEP at 80 CUs, `overlap_grad_reduce: false`, mock data. Synthetic skew: `moe_router_force_biased: -0.7`, with the DeepSeek expert bias off (`moe_router_enable_expert_bias: false`) unless the row says on. Step time of rank 7, mean over the steps after the first 10.

| Routing and setting | No replicas (ms/step) | With replicas (ms/step) | Change |
| --- | ---: | ---: | ---: |
| Synthetic skew, `ultraep`, `per_step`; 30-step runs, median of 3 interleaved rounds | 1339.9 | 1058.4 | -21.0% |
| Same skew, 100-step runs, `ultraep`, `per_step` | 1339.2 | 1056.9 | -21.1% |
| Same skew, 100-step runs, `ultraep`, `per_microbatch` | 1339.2 | 1105.6 | -17.4% |
| Same skew, 100-step runs, `moonep`, `per_microbatch`, 32 slots | 1339.2 | 1105.6 | -17.4% |
| Same skew, `turbo_sync_free_moe_stage` 2 / 3, 20-step runs, `ultraep`, `per_step` | 1343.4 / 1481.9 | 1063.4 / 1181.8 | -20.8% / -20.3% |
| Same skew, expert bias on (as in the DeepSeek-V3 recipes), 30-step runs, median of 3 interleaved rounds, `ultraep`, `per_step` | 1240.3 | 1046.8 | -15.6% |
| Same skew, expert bias on, all 4 layers recomputed, 20-step runs, `ultraep`, `per_step` | 1593.6 | 1306.4 | -18.0% |
| Real routing (DeepSeek-R1 routes on wikitext, decoder layers 13, 33, 47 and 26 replayed), 100-step runs, `ultraep`, `per_step` | 1093.8 | 1075.8 | -1.6% |
| Same real routing, `moonep`, `per_microbatch`, 32 slots | 1093.8 | 1152.0 | +5.3% |

- Without replicas nothing changes: 1339.9 ms against 1339.4 ms on the same code without replication support (+0.04%, same Primus-Turbo, 3 interleaved rounds; +0.1% with the expert bias on).
- Planning brings the busiest rank to 1.00-1.01 times the mean load. With the expert bias on, the bias also evens the load out as training goes on (layer 0: 2.32, 1.93, 1.58 and 1.34 times the mean at steps 1, 26, 51 and 76), so replicas save less (-15.6% over 30 steps, -8.5% over 100). The real routes start at 1.11-1.21; replay picks the recorded experts, so the expert bias does not change them.
- Loss and grad norm with replicas stay within the spread between two runs without them (relative, maximum over 100 steps). Two runs without replicas: 1.7e-4 in loss and 2.6e-3 in grad norm. `ultraep` per step: 2.4e-4 and 1.3e-3; `ultraep` per micro-batch: 2.0e-4 and 5.7e-3 (one step where the logged grad norm, printed to 3 significant digits, differs in its last digit); `moonep` per micro-batch: 2.4e-4 and 2.8e-3. With the expert bias on, `moonep` per micro-batch: 2.3e-4 and 1.3e-3 against 3.2e-4 and 2.8e-3 between two runs without replicas. With the real routes, `moonep` per micro-batch differs by 7.5e-4 in loss against 2.3e-4 between two runs without replicas (grad norm within the spread). All 4 layers recomputed: 3.2e-5 and 2.1e-4 over 20 steps; sync-free stage 2 and 3: 1.7e-5 / 1.9e-4 and 4.3e-5 / 7.8e-4 over 20 steps.
- Real routing at EP8 leaves little to gain and its hot experts change every micro-batch, so planning per step only catches the persistent part, and planning every micro-batch moves more weights than it saves. On MI355X one 88 MB expert takes about 1.5 ms over one XGMI link (about 58 GB/s).
- The per-rank imbalance of the same real routes grows with EP (offline, all 58 MoE layers): 1.17 at EP8, 1.24 at EP16, 1.46 at EP32, 1.86 at EP64. A model of the step from measured costs estimates about 0% at EP16, -3% to -5% at EP32 and -8% to -12% at EP64 for `per_step` planning, with replicas kept within a node; multi-node runs are needed to confirm it.

## Limitations

- No net gain at EP8 with real routing; larger EP needs multi-node runs. The planners do not yet keep replicas within a node.
- Pipeline parallelism and `overlap_grad_reduce` are rejected. With `overlap_grad_reduce` the replica gradients would have to reach their owners before each expert gradient bucket starts its reduction; on the proxy above (expert data parallel size 1) adding them at the end of the step lost them (grad norm off by 68%).
- Not with MegaMoE, tensor parallelism, FP8 / FP4 experts or token dropping.
- MI455X (FlyDSL DeepEP and grouped GEMM) is not covered; the segmented GEMM interface leaves room for it.

## References

- MoonEP: exact per-micro-batch replica planning.
- UltraEP (`Dots-Infra/UltraEP`): quota planning with a minimum replica size; replicas as extra expert parameters in Megatron.
- Megatron-LM #8500 (MoEScheduler): planner, transport and storage as separate pieces.
- MegaFlux (arXiv 2610.00671): replica cost in the planning objective and tile-aligned quotas.
