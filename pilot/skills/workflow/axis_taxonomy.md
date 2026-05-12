# Axis Taxonomy

**Status**: v1
**Consumed by**: `pilot.tools.diagnose` (validates `candidate_axes[].axis` is known and well-typed), `pilot.tools.replan` (picks search strategy from axis types), `pilot.tools.constraint` (enforces structural validity).

This file is the **single registry** of every tunable axis Pilot knows about. The engine refuses to emit any axis not listed here.

---

## 1. Axis types

Type drives both **search strategy** and **promotion budget** (a `cluster_shared` change has wider blast radius than a `weakly_local` one).

| Type | Definition | Example | Strategy hint |
|---|---|---|---|
| `cluster_shared` | Affects every job on the cluster, not just this run. Once changed, the EnvBaseline must be re-validated. | `NCCL_IB_HCA`, fabric MTU | Champion-Challenger |
| `weakly_local` | Per-process env or knob; safe to flip per-run with no cross-job effect; gain is usually < 10%. | `NCCL_BUFFSIZE`, `PYTORCH_HIP_ALLOC_CONF`, `turbo_deepep_use_comm_stream` | Per-Plan |
| `strongly_local` | Per-run knob with stronger effect (often > 10%); still no structural model change. | `NCCL_MIN_NCHANNELS`, `MOE_PERMUTE_FUSION`, `recompute_method` | Per-Plan + Pruning |
| `structural` | Changes the partitioning of the model or batch; invalidates memory predictions, requires fresh constraint check, and may invalidate prior CandidatePool. | `tp`, `pp`, `ep`, `cp`, `mbs`, `gbs`, `seq_length` | Successive_Halving (or Champion-Challenger when ≤ 3 candidates) |

## 2. Axis catalog

### 2.1 Parallelism (all `structural`)

| Axis | YAML key | Type | Notes |
|---|---|---|---|
| `tensor_model_parallel_size` | `tensor_model_parallel_size` | structural | typical 1, 2, 4, 8 |
| `pipeline_model_parallel_size` | `pipeline_model_parallel_size` | structural | implies bubble; needs `train_iters >= pp` |
| `expert_model_parallel_size` | `expert_model_parallel_size` | structural | for MoE; `tp×pp×ep×cp ≤ world` |
| `context_parallel_size` | `context_parallel_size` | structural | for very long seq |
| `virtual_pipeline_model_parallel_size` | `virtual_pipeline_model_parallel_size` | structural | reduces bubble at the cost of more comm |
| `micro_batch_size` | `micro_batch_size` | structural | candidates constrained by `gbs % (mbs×dp) == 0` |
| `global_batch_size` | `global_batch_size` | structural | rarely tuned; affects optimizer dynamics |
| `seq_length` | `seq_length` | structural | tuning down to fit memory |

### 2.2 Recompute & memory (mostly `strongly_local`)

| Axis | YAML key | Type | Notes |
|---|---|---|---|
| `recompute_granularity` | `recompute_granularity` | strongly_local | `null|selective|full` |
| `recompute_method` | `recompute_method` | strongly_local | `null|uniform|block` |
| `recompute_num_layers` | `recompute_num_layers` | strongly_local | only with `block` |
| `optimizer_offload` | `optimizer_offload` | strongly_local | trades compute for HBM |
| `MOE_BUFFER_PCT` | env `MOE_BUFFER_PCT` | weakly_local | dispatcher buffer share |

### 2.3 Communication (mix)

| Axis | YAML key / env | Type | Notes |
|---|---|---|---|
| `overlap_grad_reduce` | `overlap_grad_reduce` | weakly_local | safe to flip on/off |
| `overlap_param_gather` | `overlap_param_gather` | weakly_local | same |
| `gradient_accumulation_fusion` | `gradient_accumulation_fusion` | weakly_local | Primus turbo path overrides this; flipping has no effect there |
| `turbo_deepep_use_comm_stream` | `turbo_deepep_use_comm_stream` | weakly_local | observed neutral on single-node EP=8 (deepep_comm_stream_neutral) |
| `turbo_deepep_num_cu` | `turbo_deepep_num_cu` | weakly_local | candidates `[64, 80, 96]`; observed +0.7% from 64→80 on EP=8 |
| `moe_shared_expert_overlap` | `moe_shared_expert_overlap` | weakly_local | **HARD CONSTRAINT**: must be `false` when `use_turbo_deepep=true` (deepep_x_sharedovrlp_mutex) |
| `moe_router_force_load_balancing` | `moe_router_force_load_balancing` | weakly_local | **HARD CONSTRAINT**: must be `true` when `use_turbo_deepep=true` (deepep_x_real_router_hang) |
| `NCCL_BUFFSIZE` | env | strongly_local | candidates `[8M, 16M, 32M]` |
| `NCCL_MIN_NCHANNELS` | env | strongly_local | candidates `[1, 4, 8, 16]` |
| `NCCL_NET_GDR_LEVEL` | env | weakly_local | only when IB present |
| `NCCL_IB_DISABLE` | env | strongly_local | mutex with `NCCL_NET_GDR_LEVEL` |
| `RCCL_MSCCL_ENABLE` | env | weakly_local | algorithm pick |
| `NCCL_IB_HCA` | env | cluster_shared | re-validate baseline before promoting |

### 2.4 Compute kernels (mostly `weakly_local`)

| Axis | YAML key / env | Type | Notes |
|---|---|---|---|
| `attention_kernel` | `attention_backend` | weakly_local | flash, sdpa, math |
| `MOE_PERMUTE_FUSION` | env | strongly_local | when supported |
| `fp8_e4m3_fnuz` | `fp8_format` | strongly_local | when hardware supports |
| `tensor_layout` | `nhwc_supported` etc | weakly_local | rarely a tuning axis in transformers |

### 2.5 Allocator (`weakly_local`)

| Axis | YAML key / env | Type | Notes |
|---|---|---|---|
| `PYTORCH_HIP_ALLOC_CONF` | env | weakly_local | `expandable_segments`, `max_split_size_mb` |
| `PYTORCH_CUDA_ALLOC_CONF` | env | weakly_local | NV-side equivalent |

## 3. Exhausted-neighborhood radius

Re-Plan dedups against `PlanGraph.exhausted_neighborhoods` using `(parent_plan_id, axis, value)` keys. The "value-equivalence" radius depends on type:

| Type | Radius | Example |
|---|---|---|
| `structural` | exact match | `mbs=14` and `mbs=16` are different |
| `strongly_local` | exact match | `NCCL_BUFFSIZE=8M` and `=16M` are different |
| `weakly_local` | bucketed for numeric (within 25%) | `num_cu=80` covers `[64..100]` |
| `cluster_shared` | exact match per cluster_id | absolute, scoped by cluster |

This prevents thrashing on tiny env-noise candidates while still letting structural sweeps explore meaningfully.

## 4. Cross-references

- `skills/optimization/comm/SKILL.md` consumes the comm axes.
- `skills/optimization/moe/SKILL.md` consumes the MoE axes (with the two HARD CONSTRAINT pairs above).
- `skills/optimization/memory/SKILL.md` consumes the recompute/offload axes.
- `pilot/tools/constraint.check` enforces the `tp×pp×ep×cp ≤ world` and `gbs % (mbs×dp)` rules.

## 5. How to add a new axis

1. Append a row in §2.x (pick the right subsection).
2. If the axis has a HARD CONSTRAINT, document it inline AND add a rule to `pilot/tools/constraint.check`.
3. If the axis is `structural`, the engine will request a Re-Plan that re-runs `constraint.check` before EXECUTE.
4. If the axis is `cluster_shared`, the engine will request `Champion-Challenger` strategy.
5. (Optional) Add a fixture to `tests/pilot/test_diagnose.py` so future regressions catch a wrong type.

The `pilot.tools.diagnose` engine reads this file at startup; new axes are picked up on next invocation, no engine restart needed.
