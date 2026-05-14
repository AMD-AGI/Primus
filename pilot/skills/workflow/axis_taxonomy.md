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

### 2.6 FP8 / precision (mostly `strongly_local`)

| Axis | YAML key / env | Type | Notes |
|---|---|---|---|
| `fp8_recipe` | `fp8_recipe` | strongly_local | `tensorwise|delayed|blockwise`. **Empirically dominant on MI355X**: switching default `tensorwise` → `delayed` removed per-iter amax in the FP8 hot path and yielded +24.85% on DeepSeek-V2-Lite (session `20260513T024603Z`, R4). Engine prior should be high when `compute_fp8_prep_ratio > 0.02`. |
| `accumulate_allreduce_grads_in_fp32` | `accumulate_allreduce_grads_in_fp32` | weakly_local | precision/comm trade; rarely net-positive when allreduce is already overlapped |
| `attention_softmax_in_fp32` | `attention_softmax_in_fp32` | weakly_local | numerical robustness vs throughput; flip false only when stability is verified |
| `attention_dropout` | `attention_dropout` | weakly_local | float; setting 0.0 skips dropout RNG path on every attention block |

### 2.7 Megatron fusion knobs (mostly `weakly_local`)

| Axis | YAML key / env | Type | Notes |
|---|---|---|---|
| `apply_rope_fusion` | `apply_rope_fusion` | weakly_local | fuses RoPE into attention; observed +2.61% (session R9) when attention is ≥3% of iter |
| `bias_activation_fusion` | `bias_activation_fusion` | weakly_local | only meaningful when MLP has bias; check model spec first |
| `bias_dropout_fusion` | `bias_dropout_fusion` | weakly_local | same caveat as above |
| `masked_softmax_fusion` | `masked_softmax_fusion` | weakly_local | reduces small-kernel storm; usually safe to leave on |

### 2.8 CUDA graphs (`strongly_local`; **stack-blocked on MI355X+Megatron+DeepEP today**)

| Axis | YAML key / env | Type | Notes |
|---|---|---|---|
| `enable_cuda_graph` | `enable_cuda_graph` | strongly_local | **HARD MUTEX** with `cuda_graph_impl` (Megatron `arguments.py` rejects them together) |
| `external_cuda_graph` | `external_cuda_graph` | strongly_local | runs but DeepEP intranode dispatch is not capture-friendly → `HIP error: invalid argument` |
| `cuda_graph_impl` | `cuda_graph_impl` | strongly_local | enum vs str bug in `arguments.py:958` (`'in <string>' requires string ...`) |
| `cuda_graph_scope` | `cuda_graph_scope` | weakly_local | inert when `cuda_graph_impl` cannot be enabled |

The whole family is recorded for engine completeness (so DIAGNOSE can flag launch-bubble even when it can't propose a fix). REPLAN should attach `axis_meta.known_blocker = "cuda_graph_family"` and downrank priority by 0.1 until the upstream bug lands.

### 2.9 Pipeline-only knobs (require `pp >= 2`)

| Axis | YAML key / env | Type | Notes |
|---|---|---|---|
| `defer_embedding_wgrad_compute` | `defer_embedding_wgrad_compute` | weakly_local | **HARD CONSTRAINT**: `pp >= 2` (Megatron asserts) |
| `overlap_p2p_communication` | `overlap_p2p_communication` | weakly_local | **HARD CONSTRAINT**: `pp >= 2` AND `virtual_pipeline_model_parallel_size > 1` |
| `overlap_param_gather_with_optimizer_step` | `overlap_param_gather_with_optimizer_step` | weakly_local | requires DistOpt; gain only meaningful with PP |

### 2.10 Host-launch / runtime tuning (`weakly_local` env)

| Axis | YAML key / env | Type | Notes |
|---|---|---|---|
| `manual_gc` | `manual_gc` | weakly_local | yields the GC tax to a deterministic interval |
| `manual_gc_interval` | `manual_gc_interval` | weakly_local | int; only meaningful when `manual_gc=true` |
| `OMP_NUM_THREADS` | env | weakly_local | empirically +3.34% at `OMP_NUM_THREADS=4` (session R7); the pytorch default of 1-or-cpu_count is bad on EPYC + MI355X |
| `GPU_MAX_HW_QUEUES` | env | weakly_local | observed neutral; `2` is often enough for MI355X |
| `MIOPEN_FIND_MODE` | env | weakly_local | `FAST` skips heuristic search; safe default |

### 2.11 RCCL extras (mostly `strongly_local` env)

| Axis | YAML key / env | Type | Notes |
|---|---|---|---|
| `RCCL_PROTO` | env | strongly_local | `LL|LL128|Simple`; algorithm pick affects collective shape |
| `RCCL_ALGO` | env | strongly_local | `Ring|Tree|CollnetDirect`; usually `Tree` on small world, `Ring` on big |
| `RCCL_NTHREADS` | env | weakly_local | int; tune only when comm is steady-state bound |
| `TORCH_NCCL_HIGH_PRIORITY` | env | weakly_local | priority hint; observed neutral on MI355X |

### 2.12 ROCm / HSA (`strongly_local` env, with one DANGER)

| Axis | YAML key / env | Type | Notes |
|---|---|---|---|
| `HSA_NO_SCRATCH_RECLAIM` | env | weakly_local | `1` keeps scratch resident; observed neutral but cheap to try |
| `HSA_ENABLE_INTERRUPT` | env | strongly_local | **DANGER**: setting `0` measured as **-13.28% TFLOPS** (session R9 `t_r9_c2`). Engine must NEVER emit `0` unless explicitly requested with `axis_meta.acknowledge_regression=true`. |

### 2.13 MoE extras (`weakly_local`, with mutex)

| Axis | YAML key / env | Type | Notes |
|---|---|---|---|
| `moe_router_dtype` | `moe_router_dtype` | weakly_local | **HARD CONSTRAINT**: must be `fp32` (or unset) when `use_turbo_deepep=true` (DeepEP only supports float32 probs) |
| `turbo_sync_free_moe_stage` | `turbo_sync_free_moe_stage` | weakly_local | gated by Primus turbo path |

### 2.14 Hard mutexes / required-companion table (consumed by `constraint.check`)

The engine enforces this table verbatim — emitting a candidate that violates a row counts as `INVALID_CONFIG` *before* execution.

| Rule id | Trigger | Required | Source |
|---|---|---|---|
| `MUTEX-CG-IMPL` | `cuda_graph_impl` set | `enable_cuda_graph` MUST be unset | Megatron `arguments.py` assert |
| `REQ-PP-DEFER-EMB` | `defer_embedding_wgrad_compute=true` | `pipeline_model_parallel_size >= 2` | Megatron assert |
| `REQ-PP-OVRLP-P2P` | `overlap_p2p_communication=true` | `pipeline_model_parallel_size >= 2` AND `virtual_pipeline_model_parallel_size > 1` | Megatron assert |
| `MUTEX-DEEPEP-ROUTER` | `use_turbo_deepep=true` | `moe_router_dtype` ∈ {unset, `"fp32"`} | DeepEP runtime check |
| `MUTEX-DEEPEP-SHAREDOVRLP` | `use_turbo_deepep=true` | `moe_shared_expert_overlap=false` | empirical (deepep_x_sharedovrlp_mutex) |
| `REQ-DEEPEP-LBAL` | `use_turbo_deepep=true` | `moe_router_force_load_balancing=true` | empirical (deepep_x_real_router_hang) |
| `MUTEX-PROFILE-HIPBLASLT` | `profile=true` (default) | `PRIMUS_HIPBLASLT_TUNING` ∈ {unset, `"0"`} | known incompatibility (profile.md §3) |
| `WARN-HSA-INTERRUPT-OFF` | env `HSA_ENABLE_INTERRUPT=0` | warn only; require `axis_meta.acknowledge_regression=true` | empirical (-13.28% TFLOPS) |

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
