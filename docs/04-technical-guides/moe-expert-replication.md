# MoE expert replication (dynamic redundant experts)

With expert parallelism, every EP rank waits for the busiest one: the dispatch, the expert GEMMs and the combine of a MoE layer all finish when the rank with the most routed tokens finishes. Expert replication gives each rank a few **replica slots**. Before the experts run, a planner copies the hottest experts into slots on lightly loaded ranks and moves part of their tokens there; after the backward, the replicas' weight gradients go back to the owning rank. Routing, the router's statistics and the optimizer are unchanged: a replica is a temporary copy, not a new parameter, and it has no optimizer state.

It pays off when the per-rank load is uneven and stays uneven long enough to amortize moving the weights. On a DeepSeek-V3 4-layer proxy on 8 x MI355X it makes a step 1.6-3.2% faster with real routing at EP8, whose imbalance is mild and changes every micro-batch, 5.6-15.8% faster when the 8 GPUs replay the routing of the busiest node of an EP64 run, and 21% faster under heavily skewed synthetic routing ([Results](#results)). The imbalance of real routing grows with EP size, which is where replication is expected to help most. It is off by default.

Design discussion and the alternatives considered: [AMD-AGI/Primus#1502](https://github.com/AMD-AGI/Primus/issues/1502).

## Prerequisites

- MoE layers on Primus-Turbo: the DeepEP token dispatcher (`use_turbo_deepep: true`) and the Turbo grouped GEMM (`use_turbo_grouped_gemm: true`, `moe_grouped_gemm: true`), `turbo_grouped_gemm_backend` `default` or `triton`; on MI455X (gfx1250) also `flydsl` (`bf16:flydsl`), which needs a Primus-Turbo whose FlyDSL grouped GEMM takes weight segments and at most 64 local experts plus slots per GPU (its wgrad's group cap: a pinned backend does not fall back to Triton).
- A Primus-Turbo whose `grouped_gemm` takes the weights as a list of segments ([AMD-AGI/Primus-Turbo#675](https://github.com/AMD-AGI/Primus-Turbo/pull/675)). With an older Primus-Turbo, enabling replication raises an error at startup; rebuild Primus-Turbo from a commit that has it (the Primus CI image is built from the commit pinned as `PRIMUS_TURBO_COMMIT` in `.github/workflows/ci.yaml`).
- BF16 experts, tensor and expert tensor parallel size 1, no token dropping. Pipeline parallelism, also interleaved (virtual stages), and the EP communication overlap (`overlap_moe_expert_parallel_comm`, with Megatron's schedules) work with both placements ([Pipeline parallelism](#pipeline-parallelism)).

Combinations that are rejected at startup (`rocm_arg_validation.py`), each with the reason in the error message: MegaMoE (`use_turbo_mega_moe`), FP8 / FP4 experts and the fused grouped MLP (`turbo_fused_grouped_gemm`), `moe_expert_capacity_factor`, `moe_router_force_load_balancing_type: even`, `delay_wgrad_compute`, Megatron FSDP and torch FSDP2, CUDA graphs, the deprecated MoE layer (`use_deprecated_20241209_moe_layer`), `moe_router_grad_merge`, `mlp_chunks_for_training` above 1 (several MoE calls per micro-batch), `num_microbatches_with_partial_activation_checkpoints`, Primus's zero-bubble runtime (`patch_zero_bubble`), the `patch_primus_pipeline` schedules together with `overlap_moe_expert_parallel_comm`, `tensor_parallel_num_weight_shards` above 1, `num_distributed_optimizer_instances` above 1, and `overlap_param_gather` without the distributed optimizer. Models whose MoE layer subclasses Megatron's `MoELayer` (their own experts or forward) raise an error when the layer is built, and so does a MoE layer without the Primus-Turbo DeepEP dispatcher (for example when the Primus-Turbo patches did not apply), instead of running without replicas. Activation recompute (also with `overlap_grad_reduce`) and `turbo_sync_free_moe_stage` 2 and 3 (tested with the overlap off) work with it (planning still reads the routed counts on the host once per plan).

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
- **Slot gradients.** The slot weights carry a `main_grad` in the dtype of the owners' `main_grad` (as in Megatron's DDP: fp32 with `accumulate_allreduce_grads_in_fp32`, otherwise the parameter dtype; checked against the owners' at first use), so with `turbo_grouped_gemm_wgrad_accum_fusion` the wgrad GEMM accumulates into both segments' `main_grad` in the same launch; without it, a hook adds the slot gradients into their `main_grad`.
- **Transfers.** Slot weights are pushed with NCCL point-to-point right after planning and waited for just before the expert GEMM. Slot gradients are returned in the backward of the step's last micro-batch: posted right after each expert GEMM's backward and waited for once the gradient of the tokens the dispatcher takes (the MoE layer's input) is computed, so they are in the owners' `main_grad` before `finalize_model_grads`; the owner adds them in source-rank order. `moe_replica_async_comm: false` makes both synchronous at their point of use.
- **Overlapped gradient reduction and parameter all-gather.** With `overlap_grad_reduce`, DDP would start reducing (and scaling) an expert gradient bucket as soon as the owners' weight gradients exist, before the replica gradients reach their `main_grad`. A patch on DDP's backward hook (`megatron.moe.replica_ddp`) therefore leaves the owners' weights unmarked on the last micro-batch, and the end of the step marks each of them ready once, after the replica gradients are in, so the reduction still overlaps the rest of the backward. With `overlap_param_gather` (distributed optimizer), the expert linears' forward pre-hook waits for their parameters' all-gather only after the weights were pushed to the replicas, so the push first waits for that all-gather the same way. Megatron's pipeline schedules (with `align_grad_reduce`, the default) may run every backward of a stage in `no_sync` and start a model chunk's reduction themselves after its last backward, which comes after the end of the step; then there is nothing to mark.
- **Storage.** Every expert linear of every MoE layer keeps its own `B` slots, allocated on the training GPU when the MoE layer is built (the startup check logs their estimated size): per slot, one expert's weights and as much again for their gradients. The slots are plain tensors, not parameters or buffers, so checkpoints do not change. See [Memory](#memory).

### Pipeline parallelism

A pipeline schedule runs the forwards of later micro-batches before the backward of an earlier one (1F1B keeps up to `pipeline_model_parallel_size` of them in flight per stage), but each MoE layer still runs the forwards, and separately the backwards, of a step in micro-batch order. Each backward therefore uses what its own forward saw:

- **End of the step.** The identity whose backward ends the step records in its forward whether its micro-batch is the step's last; it also checks that the backwards come in micro-batch order and raises otherwise, since the slots and the end of the step depend on it.
- **Slots.** With `per_microbatch`, later forwards may have reassigned the slots before an earlier micro-batch's backward. An identity on each expert GEMM's output restores, in its backward, the slot table of its forward before the GEMM's backward runs: the gradients of the slots that change go back to their owners and the forward's weights are pushed again (as UltraEP restores its replica weights). With `per_step` the slot table does not change within a step and nothing moves.
- **Recompute.** The recompute takes its forward's slot table along; the planner keeps comparing new plans with the latest one.
- **Pipeline warmup.** Primus's `pp_warmup` runs one forward and backward on synthetic data before training; the replica schedulers forget it (its slot gradients are dropped) so the first step counts its micro-batches from zero.
- **EP communication overlap** (`overlap_moe_expert_parallel_comm`). Megatron then runs a layer piece by piece (router and dispatch preprocessing, dispatch, experts, combine) and interleaves one micro-batch's forward pieces with another's backward pieces on a compute and a communication stream. The end of the step sits in the dispatch preprocessing and the restore on the expert GEMM output, so both are on that path, and a layer's micro-batches still run their forwards, and separately their backwards, in order. With `moe_replica_async_comm`, a forward's weight push may still be in flight when the restore of an earlier micro-batch pushes again; the restore waits for it first.

Verified end to end ([Results](#results)): Megatron's 1F1B and interleaved schedules with both placements, and the `patch_primus_pipeline` schedules `1f1b`, `1f1b-interleaved`, `zero-bubble` and `zbv-formatted` with `per_step` and `communication_method: batch_p2p` (the default `async_p2p` did not start on the test machine, with or without replicas). Other `pp_algorithm` values and Primus's zero-bubble runtime (`patch_zero_bubble`) are rejected. The zero-bubble schedules defer weight gradients of the linears that support it; the Turbo grouped linear computes the expert and slot weight gradients in the backward.

### Correctness rules

- **Router statistics stay logical.** The expert bias update and the aux loss use the router's counts over logical experts.
- **Collectives depend only on state all ranks agree on** (the slot tables, whether a plan has replicas), never on whether one rank happens to have rows or gradients; otherwise point-to-point transfers wait on each other forever.
- **Activation recompute replays the remap.** Megatron's first forward under checkpointing runs without grad; the scheduler tells training from evaluation by the `training` flag of the expert modules, plans in that forward and replays the same remap in the recompute.
- **A slot returns its gradients before it changes hands.** When a new plan gives a slot another expert or leaves it empty, the gradients it has collected go back to their owner first; otherwise they would be lost and later added to the slot's next expert.
- **Every step ends with the gradient return.** If a step's micro-batches outnumber `get_num_microbatches()` (a step did not end in the backward of its last micro-batch), the scheduler raises instead of dropping replica gradients.

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

For DeepSeek-V3 (hidden 7168, expert FFN 2048 with SwiGLU, so 44.0 M elements per expert; BF16 weights and BF16 gradients, as with `main_grads_dtype: bf16`, which leaves `accumulate_allreduce_grads_in_fp32` off; 176 MB per slot and layer):

| Model | MoE layers | `moe_replica_slots` | Slot memory per GPU |
| --- | ---: | ---: | ---: |
| 4-layer proxy | 4 | 4 | 2.8 GB (0.7 GB per layer) |
| Full DeepSeek-V3 | 58 | 4 | 40.9 GB |
| Full DeepSeek-V3, `moonep` at EP8 (needs `slots >= num_experts / EP = 32`) | 58 | 32 | 327 GB |

`moonep` with its one-slot-per-local-expert requirement therefore does not fit a full-depth model on one GPU (MI355X has 288 GB); it is meant for shallow models until slots can be shared across layers. The startup check logs the estimate; slots that do not fit fail with an out-of-memory error when the MoE layers are built.

Example (DeepSeek-V3 proxy, EP8, one node):

```yaml
overrides:
  use_turbo_deepep: true
  use_turbo_grouped_gemm: true
  moe_replica_planner: ultraep
  moe_replica_placement: per_step
  moe_replica_slots: 4
```

## Benchmarking with recorded routing

Mock weights route almost uniformly, which hides the imbalance replication is for. To benchmark with a real model's routing, record it with Megatron's router tracer and replay it on the proxy:

1. Record: run the real model with `moe_routing_trace_path: <dir>` (and `moe_routing_trace_max_training_iters` to limit the steps), without activation recompute of the MoE layers and without evaluation, so that every record is one training forward. Every rank writes `<dir>/router_trace_rank<r>.jsonl` with the top-k experts of every MoE layer.
2. Replay: run the proxy on the same number of ranks with `moe_routing_replay_path: <dir>`. Each rank reads its own file; each MoE layer replays the recorded decoder layer with the same number, or the ones listed in `moe_routing_replay_layers`, in layer order (e.g. `[10, 30, 44, 23]` drives a 4-layer proxy with four layers of a 61-layer model). Every training forward takes the next recorded tokens of its layer and wraps around; evaluation routes by the router's own scores and leaves the recorded position as it is.

The router still computes its scores; Megatron's routing replay picks the recorded experts and takes their probabilities from those scores. The replayed routes need the same top-k and expert count as the proxy. Not with `disable_primus_topk_router` (replay lives in Primus's top-k router), `moe_router_fusion`, `moe_use_fused_router_with_aux_score`, forced or biased routing, recompute of MoE layers, PP > 1 or MTP.

## Results

8 x MI355X, one node, EP8, BF16, DeepSeek-V3 4-layer proxy: 256 experts, top-8, `main_grads_dtype: bf16`, sequence 4096, MBS 2, GBS 64 (4 micro-batches per step, 8192 tokens per rank per micro-batch), Turbo DeepEP at 80 CUs, distributed optimizer, `overlap_grad_reduce` and `overlap_param_gather` off unless noted, mock weights. `ultraep` with the defaults (4 slots, `min_rows` 1024, `min_gain` 0.1), `per_step`. The DeepSeek expert bias is off (`moe_router_enable_expert_bias: false`) unless the row says on. Step time on rank 7: median of 3 interleaved runs per arm (20- or 30-step runs, steps after the 10th), or the mean after the 10th step of one 100-step run where marked. Real routing is replayed with `moe_routing_replay_path` ([Benchmarking with recorded routing](#benchmarking-with-recorded-routing)). Primus-Turbo [#675](https://github.com/AMD-AGI/Primus-Turbo/pull/675) at `8eadf346`; Primus: this feature's code at revisions measured during the review of [AMD-AGI/Primus#1508](https://github.com/AMD-AGI/Primus/pull/1508); the synthetic-skew row and the recompute run after the last change to the replica path, the other rows earlier (re-measuring the synthetic skew after the later revisions moved it by less than 0.1%).

| Routing | Busiest rank / mean without replicas -> with them | No replicas (ms/step) | With replicas (ms/step) | Change |
| --- | --- | ---: | ---: | ---: |
| Balanced: EP64 node, late training with the expert bias | 1.05 -> 1.05 | 1002.5 | 1006.7 | +0.4% |
| Real: EP8, DeepSeek-R1 routes on wikitext, model layers 13 / 33 / 47 / 26, 100-step run | 1.11-1.21 | 1093.8 | 1075.8 | -1.6% |
| Real: EP8, same routes, model layers 10 / 30 / 44 / 23 | 1.31 | 1106.7 | 1071.1 | -3.2% |
| Real: EP64 node, training without the expert bias | 1.31 -> 1.03 | 1097.2 | 1036.0 | -5.6% |
| Real: EP64 node, DeepSeek-R1 routes on mixed text (web, math, code, books) | 1.44 -> 1.05 | 1157.5 | 1048.3 | -9.4% |
| Real: EP64 node, early training | 1.71 -> 1.08 | 1262.2 | 1067.2 | -15.4% |
| Real: EP64 node, DeepSeek-R1 routes on wikitext | 1.68 -> 1.05 | 1254.0 | 1056.2 | -15.8% |
| Synthetic skew (`moe_router_force_biased: -0.7`) | 1.6-2.3 -> 1.00-1.01 | 1342.2 | 1059.9 | -21.0% |
| Same skew, expert bias on (as in the DeepSeek-V3 recipes) | | 1240.3 | 1046.8 | -15.6% |
| Same skew, expert bias on, 100-step run | | 1131.4 | 1039.3 | -8.1% |

- EP64 node: the real routes are split as in EP64 (64 ranks with 4 experts each, 8 nodes) and the 8 GPUs play the node with the busiest rank, each GPU getting the same routed tokens per layer and micro-batch as its EP64 rank (within 1%). This measures the expert compute inside one node; a real EP64 run also has imbalance between nodes, which replicas confined to a node cannot remove, and inter-node all-to-all, so it gains less (estimates in [#1502](https://github.com/AMD-AGI/Primus/issues/1502)). "Early training" and "training without the expert bias" are routes recorded while training an 8-layer DeepSeek-V3-shaped model from scratch on FineWeb-Edu; the R1 routes are DeepSeek-R1-0528's.
- The per-rank imbalance of the same real R1 routes grows with EP (offline, all 58 MoE layers): 1.17 at EP8, 1.24 at EP16, 1.46 at EP32, 1.86 at EP64.
- With the expert bias on, the bias itself evens out a fixed skew during training (layer 0: 2.32, 1.93, 1.58, 1.34 times the mean at steps 1, 26, 51, 76), so replicas save less over a run: -15.6% over 30 steps, -8.1% over 100.
- Real routing at EP8 starts at 1.11-1.31 and its hot experts change every micro-batch, which leaves little to gain. Planning per micro-batch then moves more weights than it saves: `moonep` per micro-batch with 32 slots is 5.3% slower on the R1 routes (1093.8 -> 1152.0 ms, 100-step run). On MI355X one 88 MB expert takes about 1.5 ms over one XGMI link (about 58 GB/s).
- Other modes on the synthetic skew, expert bias off: `ultraep` and `moonep` (32 slots) per micro-batch -17.4% each (100-step runs, means: 1339.2 ms without replicas, 1105.6 ms with either; two independent runs whose means happen to agree, medians 1104.3 and 1106.7 ms); `turbo_sync_free_moe_stage` 2 / 3 -20.8% / -20.3% (20-step runs); all 4 layers recomputed, expert bias on, -17.9% (1595.3 -> 1309.1 ms, 20-step run).
- With `overlap_grad_reduce` and `overlap_param_gather` on (same session, 3 interleaved 30-step rounds per arm): 1327.2 -> 1044.6 ms (-21.3%), against 1341.7 -> 1059.5 ms (-21.0%) with both off; the overlap itself saves 1.1% without replicas and 1.4% with them. Loss and grad norm with replicas stay within twice the spread between runs without them (loss 9.3e-5 against 6.3e-5, grad norm 9.2e-4 against 7.1e-4). At EP4 (expert data parallel size 2, same 8 GPUs; 2 runs without replicas, 1 with): 1415.0 -> 1358.2 ms (-4.0%), loss 6.0e-5 and grad norm 8.1e-4 against 7.1e-5 and 1.0e-3 between the runs without replicas. Comparing every slot with its owner (sums of the weights) after each step's weight push at expert data parallel size 2: no mismatch in 232 checks with the wait for the all-gather, 112 mismatches without it. With the overlap on, all 4 layers recomputed and the expert bias on (20-step runs): 1577.2 -> 1292.7 ms (-18.0%), loss 1.6e-5 against 2.3e-5 between runs without replicas.
- Pipeline parallelism, PP2 x EP4 on the same 8 GPUs (expert data parallel size 1, 8 micro-batches per step), overlap on, `pp_warmup` on: `per_step` 1453.7 -> 1404.9 ms (-3.4%, median of 6 interleaved 30-step rounds), `per_microbatch` 1420.5 ms (-2.3%, one run); interleaved with one layer per virtual stage, 1412.7 -> 1383.3 ms (-2.1%) per step and 1422.6 ms per micro-batch (0.7% slower); all 4 layers recomputed, 1798.1 -> 1720.6 ms (-4.3%) per step and 1733.7 ms (-3.6%) per micro-batch. The gain is smaller than at EP8 because each of 4 EP ranks averages over twice as many experts, so the same skew leaves less imbalance. `patch_primus_pipeline` with `communication_method: batch_p2p`, per step: `1f1b` 1467.0 -> 1421.1 ms (-3.1%), `1f1b-interleaved` 1479.3 -> 1423.5 ms (-3.8%), `zero-bubble` 1437.5 -> 1390.8 ms (-3.2%), `zbv-formatted` 1632.7 -> 1552.9 ms (-4.9%). Loss and grad norm with replicas stay within twice the spread between runs without them in every one of these (PP2 per step, 7 runs with replicas and 11 without: loss 1.5e-4 against 1.1e-4, grad norm 1.4e-3 against 1.2e-3); `zbv-formatted` logs no loss, only the grad norm was compared. Comparing every slot with its owner after each restore (PP2, per micro-batch): no mismatch in 964 checks, 284 of them after the restore moved weights.
- EP communication overlap (`overlap_moe_expert_parallel_comm`), same setup as the pipeline runs: PP1 at EP8, 1245.2 -> 1048.5 ms (-15.8%, 3 interleaved 30-step rounds) per step and 1161.4 ms (-6.7%) per micro-batch; PP2 x EP4 interleaved with one layer per virtual stage, 1371.2 -> 1371.7 ms (+0.0%; the 3 rounds with replicas spread from 1352.5 to 1395.7 ms) per step and 1445.9 ms (5.5% slower) per micro-batch. Without replicas the overlap saves 6.2% at PP1 (1326.8 -> 1245.2 ms) and 2.9% interleaved (1412.7 -> 1371.2 ms); with them it saves almost nothing (1045.3 -> 1048.5 ms and 1383.3 -> 1371.7 ms; not profiled). Loss and grad norm with replicas within twice the spread between the 3 runs without them in all four (PP1 per step: loss 8.6e-5 against 5.2e-5, grad norm 9.4e-4 against 7.1e-4). Comparing every slot with its owner after each restore (interleaved, per micro-batch): no mismatch in 964 checks, 312 of them after the restore moved weights.
- Off by default, and off it costs nothing: 1342.2 ms against 1342.0 ms on `main` (`a4775075b`) with the same Primus-Turbo, 3 interleaved 30-step rounds (+0.01%). The loss is identical to `main` for the first 2 steps and then within the spread between runs of `main` (6.1e-5 against 5.0e-5 relative over 30 steps).
- Loss and grad norm with replicas stay within the spread between two runs without them (relative, maximum over 100 steps). Two runs without replicas: loss 1.7e-4, grad norm 2.6e-3. `ultraep` per step 2.4e-4 and 1.3e-3; `ultraep` per micro-batch 2.0e-4 and 5.7e-3 (one step where the logged grad norm, printed to 3 significant digits, differs in its last digit); `moonep` per micro-batch 2.4e-4 and 2.8e-3; with the expert bias on, `moonep` per micro-batch 2.3e-4 and 1.3e-3 against 3.2e-4 and 2.8e-3 between two runs without replicas. With the real routes, `moonep` per micro-batch differs by 7.5e-4 in loss against 2.3e-4 between two runs without replicas, with the grad norm within the spread. All 4 layers recomputed: 3.2e-5 and 2.1e-4 over 20 steps; sync-free stage 2 / 3: 1.7e-5 / 1.9e-4 and 4.3e-5 / 7.8e-4.

## Limitations

- Small gains with real routing at EP8 (1.6-3.2%). The EP64 rows emulate the busiest node only; a real EP64 run needs multi-node measurements, and the planners do not yet keep replicas within a node.
- Under pipeline parallelism, `per_microbatch` returns slot gradients and pushes weights once more per expert linear and backward (the restore); `per_step` adds nothing.
- Planning copies the routed counts to the host (once per step with `per_step`, every micro-batch with `per_microbatch`), which stalls the host while the EP communication overlap would run ahead.
- Not with MegaMoE, tensor parallelism, FP8 / FP4 experts or token dropping.
- The step ends in the backward of the tokens each MoE layer's dispatcher takes (the layer's input), so in training that input must require a gradient: with every parameter before the MoE layers frozen, replication raises an error.
- MI455X with a FlyDSL BF16 grouped GEMM takes at most 64 local experts plus slots per GPU (the FlyDSL wgrad's group cap), so EP2 runs of a 128-expert model replicate on `default` / `triton` only.

## References

- MoonEP: exact per-micro-batch replica planning.
- UltraEP (`Dots-Infra/UltraEP`): quota planning with a minimum replica size; replicas as extra expert parameters in Megatron.
- Megatron-LM #8500 (MoEScheduler): planner, transport and storage as separate pieces.
- MegaFlux (arXiv 2610.00671): replica cost in the planning objective and tile-aligned quotas.
