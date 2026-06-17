# MI355X 128k MoE NaN — Reproduction & Root-Cause Investigation

## TL;DR (Conclusion)

On **MI355X (gfx950) / ROCm 7.2.1**, the MoE **fused grouped-GEMM** in the deepest
MoE layer emits **sparse NaN / `~3.4e38` (bf16-max) values from finite,
normal-magnitude inputs** (`X_absmax ≈ 6`, `W_absmax ≈ 0.1`) under the
highly-imbalanced, very-large-M grouped layout produced by long context +
unbalanced routing (per rank: 64 expert groups, ~520k total rows, groups of a
few thousand rows). The NaN then floods the backward graph → `grad_norm = NaN`.

**Root cause: the fused grouped-GEMM makes an out-of-bounds (OOB) GPU memory
access — a memory-safety bug, not a numeric miscompute.**

- **Disabling the caching allocator (`PYTORCH_NO_CUDA_MEMORY_CACHING=1`) turns the
  sparse NaN into a hard HIP "illegal memory access" fault at the same grouped
  fc1.** With the allocator on, the OOB lands in a cached arena and reads stale
  bytes (`~3e38`) → sparse NaN (e.g. 82 of 5.3e8 elements); with it off it faults.
- **Replaying the exact captured `(X, W, m_splits)` offline through
  `te.GroupedLinear` is FINITE** (absmax ≈ 5.44, matches fp32/bf16
  `torch.matmul`) — the OOB region happens to be in-bounds/zeroed for a fresh,
  exactly-sized allocation, so the single op alone does not reproduce.
- It is **not a race**: `AMD_SERIALIZE_KERNEL=3` + `HIP_LAUNCH_BLOCKING=1` and
  `turbo_deepep_use_comm_stream=false` both still fail.

Firmly ruled out: model weights / initialization (clean at iter 0),
input/activation explosion (`X ~ 6`, fp32 ref `~5`), bf16 precision (bf16
`torch.matmul` ≈ fp32), dispatcher choice (DeepEP **and** alltoall both
reproduce), and any deterministic GEMM arithmetic error.

> ⚠️ Investigation note: earlier conclusions ("deterministic grouped-GEMM kernel
> miscompute", then "uninitialized read") were **revised** — first after the
> offline `te.GroupedLinear` replay came back finite, then after
> `PYTORCH_NO_CUDA_MEMORY_CACHING=1` produced a hard IMA, pinning it to an OOB
> access. See [§7.4](#74-offline-single-op-replay-does-not-reproduce) and §8.

> ✅ **Update (2026-06-17): the OOB has now been isolated to one specific
> hipBLASLt solution and reproduced standalone.** Further drill-down on the **same
> qwen3 reproduction** pinned the fault to hipBLASLt **solution index `332814`**,
> which is *both* numerically wrong (`norm_error ≈ 0.07`, all other ~100 solutions
> ≈ `3e-5`) *and* writes out of bounds past the output. It is selected by the
> hipBLASLt heuristic only for a narrow `n` range, which is why it looks
> data-dependent. Two standalone reproducers now exist (a `hipblaslt-bench`
> one-liner → wrong result, and a VMM guard-page microbenchmark → the exact
> `Memory access fault … Write access to a read-only page`). This **refines the
> §7.4/§9 "offline does not reproduce" finding**: the offline replay missed it
> only because it re-ran the heuristic (picking a *good* solution) into a roomy,
> exactly-sized buffer. See **[§11](#11-update-isolated-to-a-specific-hipblaslt-solution-standalone-repro)**.

---

## 1. Original Symptom (as reported)

- **Failing benchmark:** `qwen-qwen3-6-35b-a3b … seq131072` on 8×MI355X.
- At `seq_len = 131072` the MoE fused grouped-GEMM (`TEColumnParallelGroupedLinear`,
  hipBLASLt) intermittently produces **NaN from finite, normal-magnitude inputs**
  (`x_absmax ≈ 20–50`, weights `≈ 0.05–0.17`), poisoning gradients → `grad_norm = NaN`.
- Reproduces with **mock/random tokens** too (the router still routes them into
  highly imbalanced expert groups, max ~130k tokens/expert).
- **DeepEP dispatch amplifies** it; `alltoall` mitigates but does not eliminate.
- Does **not** reproduce at short context / uniform routing.

> Note: the original benchmark model is `Qwen/Qwen3.6-35B-A3B` (kv_channels=128).
> The in-repo reproduction below uses the repository model
> `qwen3_5_35B_A3B` (gated-delta-net linear attention, 256 experts, top-8,
> kv_channels=256, num_query_groups=2). The MoE grouped-GEMM path is identical;
> see the parallelism deviation note in [§3](#3-in-repo-reproduction-config).

---

## 2. Environment / Versions

| Component | Version |
|---|---|
| Container image | `docker.io/rocm/primus:v26.3` |
| ROCm | 7.2.1 |
| torch | 2.10.0+git94c6e04 (`torch.version.hip` = 7.2.53211) |
| transformer-engine | 2.12.0.dev0+40434cf6 |
| megatron-core | 0.18.0+d978c6bf1 |
| primus_turbo | 0.2.0+3cd482d |
| nvidia-modelopt | 0.44.0 |
| GPU | 8× MI355X (gfx950) |

Reproduction node: SLURM `mi355-gpu-14`, container `nanrepro_weihuan`, single
node × 8 GPUs.

---

## 3. In-repo Reproduction Config

Config: `examples/megatron/configs/MI355X/qwen3_5_35B_A3B-BF16-nanrepro.yaml`
(derived from the working `qwen3_5_35B_A3B-BF16-pretrain.yaml`).

Key NaN-triggering settings:

| Lever | Value | Purpose |
|---|---|---|
| `seq_length` | 131072 | long context → huge per-expert M |
| `tensor_model_parallel_size` | 2 | (deviation, see below) |
| `expert_model_parallel_size` | 4 | 64 local experts / rank |
| `context_parallel_size` | 1 | (CP unsupported, see §6) |
| `use_turbo_deepep` | true | DeepEP dispatcher (`flex`) |
| `moe_grouped_gemm` | true | enable grouped GEMM |
| `moe_use_legacy_grouped_gemm` | false | → TE `TEColumnParallelGroupedLinear` |
| `use_turbo_grouped_mlp/gemm` | false | not turbo path |
| `turbo_sync_free_moe_stage` | 0 | stage≥2 would force turbo grouped gemm |
| `moe_router_force_load_balancing` | false | allow imbalance |
| `moe_router_load_balancing_type` | none | no forced balancing |
| `moe_aux_loss_coeff` | 0.0 | no aux balancing |
| `micro_batch_size` / `global_batch_size` | 1 / 4 | DP=4, single microbatch |
| precision | bf16 | |
| `mock_data` | true | random tokens still reproduce |

**Parallelism deviation:** the report used TP=4, but the in-repo
`qwen3_5_35B_A3B` has `num_query_groups=2`, which GQA requires to be divisible
by TP, so TP=4 is invalid for this model. The reproduction uses **TP=2 / EP=4**
(world=8 → DP=4). All NaN-triggering factors (long seq, DeepEP, TE grouped GEMM,
imbalanced routing, EP sharding) are preserved.

**Two unrelated config fixes** were needed for the run to start / not OOM:
1. `overlap_grad_reduce: false` + `overlap_param_gather: false` — required because
   `use_distributed_optimizer=false` (otherwise `validate_args` asserts).
2. `create_attention_mask_in_dataloader: false` + `num_workers: 1` — at
   seq=131072 a `[s, s]` bool attention mask is ~17 GB/sample and OOM-kills the
   dataloader worker (flash/TE attention does not need it).

### How to run (inside the container)

```bash
cd /workspace/Primus
EXP=examples/megatron/configs/MI355X/qwen3_5_35B_A3B-BF16-nanrepro.yaml \
  bash examples/run_pretrain.sh
```

Result: **NaN at iteration 1**, `found NaN in local grad norm for bucket #0 in
backward pass` on multiple ranks. Reproduces every run.

---

## 4. Debug Instrumentation Used (temporary; reverted)

During the investigation, temporary env-gated instrumentation was added to locate
the fault. **These edits have been reverted and are not part of the committed
change set** (only `tools/grouped_gemm_nan_repro.py`, the configs, and this doc
are kept). They are documented here for reproducibility — recreate them if you
need to re-run the localization.

| Where | Purpose | Env gate |
|---|---|---|
| forward non-finite tracer + in-situ fp32/bf16-vs-kernel verify + artifact save (new module, wired from the `get_model` patch) | find first non-finite forward; prove matmul-finite vs kernel-NaN; dump `(X,W,m_splits)` | `PRIMUS_NAN_TRACE=1`, `PRIMUS_NAN_SAVE=1`, `PRIMUS_NAN_SAVE_RANK` |
| per-expert fwd/bwd NaN dump for the Primus-Turbo grouped MLP path (new module, wired from `PrimusGroupedMLP`) | dump offending expert slice | `PRIMUS_NAN_DUMP=1` |
| param-name tagging + iter-0 weight-stats scan in the `get_model` patch | rule out bad init; name params for the grad scan | `PRIMUS_NAN_DUMP=1`, `PRIMUS_NAN_TRACE=1` |
| all-param NaN-`main_grad` scan at the grad-norm check (`param_and_grad_buffer.check_grads`, a Megatron submodule file) | locate which params' grads are NaN | `PRIMUS_NAN_DUMP=1` |
| `tools/grouped_gemm_nan_repro.py` | **kept** — standalone single-op reproducer / value-distribution dump | — |

The decisive controls used only stock env vars (no code): `AMD_SERIALIZE_KERNEL=3`,
`HIP_LAUNCH_BLOCKING=1`, `PYTORCH_NO_CUDA_MEMORY_CACHING=1`, `HIPBLASLT_LOG_LEVEL=2`.

---

## 5. Experiment Matrix

All runs use the in-repo config above (TP2/EP4/CP1, seq 131072, bf16, mock data,
no load balancing) varying a **single** lever at a time.

| # | Variant | Lever changed | Result |
|---|---|---|---|
| 1 | baseline | TE grouped GEMM (hipBLASLt) | **NaN @ iter 1** (backward grad-norm) |
| 2 | `-turbogg` | `use_turbo_grouped_gemm: true` | **NaN @ iter 1** (identical) |
| 3 | `-legacygg` | `moe_use_legacy_grouped_gemm: true` | **NaN** (fc1 out `3.38e38`) |
| 4 | `-nogg` | `moe_grouped_gemm: false` (SequentialMLP) | **HIP illegal memory access** (crash; inconclusive — that path is itself unstable under this layout) |
| 5 | `-alltoall` | `use_turbo_deepep: false` (alltoall) | **NaN @ iter 1** (identical, dense buffer 124–126/463) |
| 6 | `-cp2` / `-cp4` | `context_parallel_size: 2/4` | **Not runnable**: `Gated delta net does not support context parallel` |
| 7 | `-nocommstream` | `turbo_deepep_use_comm_stream: false` | **NaN** (iter 2) — comm-stream overlap not the cause |
| 8 | serialize env | `AMD_SERIALIZE_KERNEL=3` + `HIP_LAUNCH_BLOCKING=1` | **NaN** (iter 2) — not a kernel-launch race |

**Takeaways:** kernel-implementation–agnostic (TE/turbo/legacy all NaN),
dispatcher-agnostic (DeepEP and alltoall both NaN), and **not** a race
(serialization + comm-stream-off both still NaN). CP cannot be used on this model.

---

## 6. Localization

### 6.1 Full-parameter gradient scan

At the grad-norm check (`check_grads`), with `PRIMUS_NAN_DUMP=1`, scanning every
parameter's `main_grad`:

- **Dense param buffer:** ~**216–218 / 463** params have fully-NaN grad
  (`nan_frac=1.0`), spanning **layers 0–18** (layers 19–39 clean), across
  `self_attention` (125), `mlp.shared_experts` (54), `mlp.router` (19),
  `pre_mlp_layernorm` (19), `embedding` (1).
- **Grouped expert params** (`mlp.experts.*`, in a separate expert grad buffer)
  did **not** trigger the NaN scan.

Interpretation: the NaN is generated deep in the backward graph and floods the
shared residual gradient to all lower layers — initially misleading, since it
makes nearly half the dense params look guilty.

### 6.2 Forward non-finite tracer

With `PRIMUS_NAN_TRACE=1` (forward hooks only, no autograd perturbation), the
**first** non-finite forward tensor is consistently:

```
layers.39.mlp.experts.linear_fc1  shape=(~520000, 1024)
   nan=True  finite_absmax ≈ 3.32–3.39e38      (bf16 max ≈ 3.39e38)
   | IN  absmax ≈ 6      nan=False inf=False    (input finite, normal)
   | W   weight0..3 absmax ≈ 0.10  nan=False inf=False   (weights finite, normal)
```

i.e. the deepest layer's grouped fc1 output explodes to the bf16 ceiling. With
the trace hooks present the abort sometimes shifts to iteration 2–3 / forward
loss; without them it is iteration-1 backward grad-norm. Same underlying defect,
timing perturbed by hooks (consistent with the "intermittent" nature).

### 6.3 Initialization check (rules out bad init)

iter-0 (pre-training) weight scan: **0 params with NaN/Inf**, top `absmax = 2.766`
(`self_attention.A_log`, expected); expert weights `~0.1`. Initialization is
clean. At the moment of failure the weights are still `~0.1` (finite).

---

## 7. Evidence — kernel output vs correct math (and offline non-repro)

### 7.1 In-situ: kernel NaN vs matmul finite (same in-memory tensors)

When fc1 output is non-finite during training, recomputing the offending expert
groups with plain `torch.matmul` (`X_i @ W_iᵀ`) on the **exact same in-memory
tensors**, in both fp32 and bf16:

```
BAD-GROUP expert=5  rows≈5500   X_absmax≈5.7  W_absmax≈0.10
   REF fp32         absmax ≈ 4.65   finite=True
   REF bf16 (torch) absmax ≈ 4.66   finite=True          ← bf16 is fine too
   KERNEL (grouped) nan=True        finite_absmax ≈ 3.2e38   ← grouped output corrupted
```

Observed across ranks 1/2/3/5/6/7 and layers 11/17/24/31/34/35/39, experts
(5, 9, 23, 26, 40, 49, 54, 60, 61, …).

### 7.2 The corruption is SPARSE

The grouped output is mostly correct — only **~tens of elements** are NaN
(e.g. **82 of 5.3e8** total; per bad group 41 / 28 / 13). Quantiles of the
in-training kernel output match the reference up to q99.9; only the extreme tail
is `~3e38`. This is not a wholesale miscompute.

### 7.3 Not empty-groups / padding / precision

- `sum(tokens_per_expert) == out_rows`, `empty_groups = 0`, no NaN in any padding
  tail → not a buffer-sizing / empty-group issue.
- The **largest** group is computed correctly; broken groups are mid-sized.
- Correct magnitude (`~5`) fits easily in bf16 → not overflow / dynamic range.

### 7.4 Offline single-op replay does NOT reproduce

Replaying the **captured** `(X, W, m_splits)` through `te.GroupedLinear` offline
(see `tools/grouped_gemm_nan_repro.py`) is **finite**:

```
INPUT   X: min=-5.81 max=6.16 mean=-0.003 std=1.00 absmax=6.16  nan=0
        W: std=0.02 absmax=0.11  nan=0
REF fp32        : absmax=5.44  nan=0   non-finite groups: []
REF bf16(torch) : absmax=5.44  nan=0   non-finite groups: []
TE grouped rerun: absmax=5.44  nan=0   non-finite groups: []   ← NO REPRO offline
KERNEL (training): absmax=3.24e38  nan=82  non-finite groups: [5, 26, 60]
```

So the **captured input is clean and a fresh grouped-GEMM on it is correct**. The
in-training NaN is therefore **not** a deterministic function of `(X, W, m_splits)`
— it is injected by the surrounding fused MoE pipeline at run time.

---

## 8. Root Cause (current best understanding)

**The fused MoE grouped-GEMM performs an out-of-bounds (OOB) GPU memory access**
under the large-M / many-group / heavily-imbalanced layout. It is a
**memory-safety bug**, not a numeric/precision miscompute.

The decisive control: **disabling the caching allocator turns the sparse NaN into
a hard HIP fault** at the very same op:

```
PYTORCH_NO_CUDA_MEMORY_CACHING=1  →  torch.AcceleratorError: HIP error:
   an illegal memory access was encountered
   at  experts.py:367  self.linear_fc1(...)   # grouped fc1 forward
```

This explains every prior observation coherently:

| Condition | What the OOB access lands in | Symptom |
|---|---|---|
| caching allocator ON (default) | a large cached arena → readable stale bytes (`~3e38`) | **sparse NaN**, no crash |
| caching allocator OFF | unmapped / freed pages | **hard IMA fault** |
| offline single-op, exactly-sized fresh tensors | still inside allocation / zeroed | **finite, no repro** |

Control runs (all consistent with an OOB, none a race):

| Control | Result |
|---|---|
| `AMD_SERIALIZE_KERNEL=3` + `HIP_LAUNCH_BLOCKING=1` | still NaN → not a kernel race |
| `turbo_deepep_use_comm_stream: false` | still NaN → not comm-stream overlap |
| alltoall (no DeepEP) | still NaN → not DeepEP |
| `PYTORCH_NO_CUDA_MEMORY_CACHING=1` | **IMA crash at grouped fc1** → OOB exposed |

| Hypothesis | Verdict | Evidence |
|---|---|---|
| Weight initialization | ❌ ruled out | iter-0 weights clean; weights `~0.1` at failure |
| Activation / input explosion | ❌ ruled out | fc1 input `absmax≈6` finite; fp32 ref `~5` |
| bf16 precision / overflow | ❌ ruled out | bf16 `torch.matmul` ≈ fp32 ≈ `~5`, finite |
| Dispatcher choice (DeepEP / alltoall) | ❌ ruled out | both reproduce |
| Deterministic grouped-GEMM miscompute | ❌ ruled out | offline replay on captured `(X,W,m_splits)` is finite |
| Async / stream-ordering race | ❌ ruled out | serialization + comm-stream-off both still NaN |
| **Out-of-bounds access in fused grouped-GEMM** | ✅ **leading root cause** | caching-allocator-off → hard IMA at grouped fc1; sparse NaN otherwise; not offline-reproducible; not a race |

Notes:
- All three grouped-GEMM implementations (TE / Primus-Turbo / legacy) surface it
  in training (same `~3.38–3.39e38` tail), and the `moe_grouped_gemm: false`
  (SequentialMLP) variant also crashed with an IMA — consistent with a shared
  memory-safety defect in the MoE expert GEMM path under this layout.
- The corruption first appears at the **deepest** MoE layer (39) and floods backward.
- Broken rows sit within mid-sized groups (consistent with a per-group
  tile-boundary OOB), and the largest group is computed correctly.

### 8.1 Exact faulting call (pinpointed)

Under `PYTORCH_NO_CUDA_MEMORY_CACHING=1 AMD_SERIALIZE_KERNEL=3 HIP_LAUNCH_BLOCKING=1
TORCH_USE_HIP_DSA=1`, the synchronous fault localizes to TE's ROCm grouped GEMM →
hipBLASLt:

```
megatron.core.extensions.transformer_engine.TEColumnParallelGroupedLinear.forward
  → transformer_engine/pytorch/module/grouped_linear.py:221  general_grouped_gemm
  → cpp_extensions/gemm.py:299                                tex.te_general_grouped_gemm
  → TransformerEngine/common/gemm/rocm_gemm.hip:1373          hipblaslt_gemm
  → HIPBLASLT Error: 6  +  HIP "illegal memory access"
```

The grouped GEMM is dispatched as a **sequence of per-expert `rocblaslt_matmul`
calls** (one per expert, `n` = that expert's token count). Representative
descriptors from the failing step (bf16 in/out, fp32 compute):

```
rocblaslt_matmul  A=[R_16BF rows=512 cols=2048 ld=512] transA=OP_T
                  B=[R_16BF rows=512 cols=N    ld=512] transB=OP_N
                  C=D=[R_16BF rows=2048 cols=N ld=2048]  computeType=COMPUTE_32F
                  workSpaceSizeInBytes=268435456  alpha=1 beta=0
   → effective m=2048, k=512, n = per-expert tokens
   → N varies widely per expert: 5039, 6018, 7431, 9462, 11369, 13114, …
```

i.e. a batch of many back-to-back bf16 grouped matmuls with **highly variable,
large `n`** is what triggers the OOB / `HIPBLASLT Error 6` inside hipBLASLt on
gfx950.

---

## 9. Deliverables & Reproducer

- **Artifact:** `output/nan_dump/grouped_fc1_repro_rank7.pt` (~3.47 GB)
  - `X` `[521714, 2048]` bf16, `weights` (64 per-expert `[1024, 2048]` bf16),
    `tokens_per_expert` (m_splits), `kernel_output` (the in-training output, has
    sparse NaN), `dtype`. Captured from `layers.39.mlp.experts.linear_fc1`, rank 7.
- **Script:** `tools/grouped_gemm_nan_repro.py` — prints input/output value
  distributions, re-runs `te.GroupedLinear`, and compares to fp32/bf16 matmul.

```bash
python tools/grouped_gemm_nan_repro.py output/nan_dump/grouped_fc1_repro_rank7.pt
```

> ⚠️ Important: the artifact captures the **post-hoc clean input** + the corrupted
> training output. Replaying the op offline is **finite** (does not reproduce),
> which is itself the key evidence that the bug is a run-time hazard, not a
> deterministic GEMM defect. A standalone bug report to hipBLASLt should therefore
> focus on the **in-training fused path / concurrency**, not the isolated GEMM.

---

## 10. Mitigations / Next Steps

Async/race ruled out; the OOB is confirmed by the caching-allocator-off IMA.
Remaining steps:

1. **Pinpoint the OOB with a sanitizer** (highest priority): re-run the grouped
   fc1 under `PYTORCH_NO_CUDA_MEMORY_CACHING=1` + `AMD_SERIALIZE_KERNEL=3` +
   `TORCH_USE_HIP_DSA=1`, and/or `rocgdb` / `compute-sanitizer`-equivalent, to get
   the faulting address and the exact kernel + access. The fault is already
   localized to `experts.linear_fc1` (grouped fc1) at the deepest MoE layer.
2. **Inspect the grouped layout / `m_splits` handling** in the grouped-GEMM
   wrapper: the broken rows are within mid-sized groups, consistent with a
   per-group tile-boundary read/write past the group/buffer end (e.g. group sizes
   rounded up to a tile multiple, or an offset computed with a too-narrow integer
   for ~520k-row inputs).
3. **Immediate training workaround:** bound per-group M via
   `moe_expert_capacity_factor`, or re-enable `moe_router_force_load_balancing` /
   aux-loss, to avoid the extreme imbalanced layout that triggers the OOB.
4. **File against the owning component** (TE / hipBLASLt / Primus-Turbo grouped
   GEMM) as a **memory-safety / OOB** bug, with the IMA stack and the artifact.

> The standalone artifact's `te.GroupedLinear` replay is finite (the OOB region is
> in-bounds for a fresh allocation), so the bug report must reproduce inside the
> training pipeline (or under `PYTORCH_NO_CUDA_MEMORY_CACHING=1`, which faults
> deterministically), not via the isolated op on its own.

**Bug report target (pinpointed):** the fault is in hipBLASLt, reached via
`TransformerEngine/common/gemm/rocm_gemm.hip:1373 hipblaslt_gemm` →
`rocblaslt_matmul` (per-expert grouped GEMM, bf16 / fp32-compute, transA=T
transB=N, m=2048, k=512, variable large n = 5k–13k). Repro recipe: run the
training config under `PYTORCH_NO_CUDA_MEMORY_CACHING=1 AMD_SERIALIZE_KERNEL=3
HIP_LAUNCH_BLOCKING=1 HIPBLASLT_LOG_LEVEL=2` and the last logged `rocblaslt_matmul`
before `HIPBLASLT Error: 6` is the offending call.

---

## 11. Update — Isolated to a specific hipBLASLt solution (standalone repro)

Further drill-down on the **same qwen3 reproduction** (this run: container
`rocm/primus:v26.3`, ROCm 7.2.1, hipBLASLt `100300` git `c4b2dc9869`) pinned the
fault all the way down to a **single buggy hipBLASLt solution**. This both confirms
the OOB root cause of §8 and removes the §9/§10 caveat that "the isolated op does
not reproduce" — it does, once the offending solution is forced.

### 11.1 Crash and how the faulting rank was identified

Symptom (qwen3 repro run, `PYTORCH_NO_CUDA_MEMORY_CACHING=1`):

```
Memory access fault by GPU node-3 (Agent handle: 0x172bc4c0) on address
0x744388610000. Reason: Write access to a read-only page.
```

- `GPU node-X` is the **HSA/KFD topology node id**, not `LOCAL_RANK`. From
  `rocminfo`, `Node 3 = Agent 4 = BDFID 1280 = PCI 05:00.0`.
- With `HIPBLASLT_LOG_MASK=48` (api+bench) and `HIPBLASLT_LOG_FILE=/tmp/hipblaslt_%i.log`,
  8 per-PID logs were produced. The faulting process is the **one whose virtual
  address arena contains the fault address**: only PID `4680`'s matmul pointers
  were `0x7443…`, and `0x744388610000` sits just past its last output buffer
  `C=D=0x744387040800`. → crashing rank = **PID 4680**.

### 11.2 Exact faulting GEMM (extracted from the bench/api log)

```
bf16, transA=OP_T, transB=OP_N, in-place C==D, alpha=1 beta=0, computeType=COMPUTE_32F
A=[R_16BF rows=2048 cols=1024 ld=2048]   → m=1024, k=2048
B=[R_16BF rows=2048 cols=5407 ld=2048]   → k=2048, n=5407 (dynamic per-expert tokens)
C=D=[R_16BF rows=1024 cols=5407 ld=1024]  workSpaceSizeInBytes=268435456
```

(Note: this is the qwen3 MoE expert **`linear_fc1`** grouped GEMM — output width
`m=1024` matches the first non-finite tensor of §6.2/§7.4 (`...experts.linear_fc1`,
shape `(~520000, 1024)`), with `k=2048` = hidden and `n=5407` = dynamic per-expert
tokens. The coarser `m=2048, k=512` descriptor quoted in §8.1 was an earlier
approximate capture of the same per-expert GEMM family; the precise faulting
problem is the one above. Same path, same bf16 T/N in-place grouped matmul into
hipBLASLt, same OOB symptom.)

### 11.3 hipBLASLt-bench: the heuristic picks a numerically WRONG solution

Replaying the exact problem with `hipblaslt-bench` (`--algo_method heuristic
--requested_solution 1 --workspace 268435456`) selects **solution index `332814`**
(a split-K / StreamK "SK3" UserArgs kernel,
`Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT128x192x128_…`). Validation (`-v`):

| solution_index | norm_error | atol/rtol |
|---|---|---|
| **332814** (heuristic top-1) | **0.0726** | **failed** |
| 335459 (any other) | 3.46e-05 | passes |

`--algo_method all -v` over **all ~100 solutions**: every other solution is
`≈ 2–5e-5` (correct); **only 332814 is ~2000× off**. So the default hipBLASLt
heuristic hands back a broken kernel for this shape.

```bash
# wrong-result repro (no custom code needed):
hipblaslt-bench --api_method c -m 1024 -n 5407 -k 2048 \
  --lda 2048 --ldb 2048 --ldc 1024 --ldd 1024 --transA T --transB N \
  --a_type bf16_r --b_type bf16_r --c_type bf16_r --d_type bf16_r \
  --compute_type f32_r --alpha 1 --beta 0 \
  --algo_method index --solution_index 332814 -v
#  → norm_error≈0.0726, atol/rtol failed   (335459 → ~3.4e-5, passes)
```

### 11.4 Why it looks data-dependent (and why §7.4 offline missed it)

The heuristic selects a *different* solution per `n`, and only `n≈5407` lands on
the broken 332814:

| n (per-expert tokens) | heuristic solution | norm_error | verdict |
|---|---|---|---|
| **5407** | **332814** | **0.0726** | ✗ broken |
| 6170 | 332681 | 3.4e-05 | ✓ |
| 6545 | 332815 | 2.9e-05 | ✓ |
| 7484 | 332652 | 3.5e-05 | ✓ |
| 9747 | 332650 | 3.4e-05 | ✓ |
| 4096 | 333644 | 3.6e-05 | ✓ |

Reducing the workspace (0 / 32 / 128 / 256 MB) does **not** help — `n=5407` still
selects 332814. This explains the "intermittent / only certain routing" behaviour
(§1) and why the §7.4 offline `te.GroupedLinear` replay was finite: re-running the
heuristic on a slightly different captured layout (or buffer) selects a *good*
solution, and the roomy fresh allocation hides any overshoot.

### 11.5 Standalone reproduction of the *fault* (not just wrong numbers)

`hipblaslt-bench` allocates roomy separate buffers, so 332814's OOB write is
absorbed → only `norm_error`. To reproduce the **hard fault** in isolation, a
microbenchmark (`repro_332814.cpp`) forces solution 332814 via
`hipblaslt_ext::getAlgosFromIndex({332814})` and backs the output `D` with the HIP
**virtual-memory API**: `D` mapped read-write, immediately followed by a
**reserved-but-unmapped guard** region.

```
D base        = 0x75454c600000
D logical end = 0x75454d08f800   (sizeD = 10.56 MB)
RW mapped end = 0x75454d090000
Launching matmul with solution 332814 ...
Memory access fault by GPU node-2 on address 0x75454d0a0000.
   Reason: Write access to a read-only page.        ← fault ~64 KB past D end
```

Decisive A/B with the **same harness/guard**, only the solution changes:

| solution | result | exit |
|---|---|---|
| **332814** | `Memory access fault … Write access to a read-only page` | 134 |
| 335459 | `Synchronized OK (no fault)` | 0 |

This is the missing standalone fault repro and is unambiguous: solution 332814
writes past the output `D`. (ROCm reports a reserved-unmapped VMM page as
"read-only", matching the original training message verbatim.) The smaller
overshoot here (~64 KB, guard placed right after `D`) vs ~21.8 MB in training
(`fault_addr − C/D base`) is just guard placement — the larger training overshoot
was absorbed by ~21 MB of neighbouring writable buffers before hitting a read-only
page.

Build/run inside the container:

```bash
hipcc -std=c++17 repro_332814.cpp -o repro_332814 \
  -I/opt/rocm/include -L/opt/rocm/lib -lhipblaslt -lamdhip64
./repro_332814 332814   # → Memory access fault (write to read-only page)
./repro_332814 335459   # → Synchronized OK
```

Source: `repro_332814.cpp` (repo root).

### 11.6 Conclusion & mitigation (refines §8 / §10)

- **Root cause (pinned):** hipBLASLt `100300` (ROCm 7.2.1, gfx950) **solution
  `332814`** is broken for `bf16, transA=T transB=N, m=1024, k=2048` — it computes
  wrong values **and** writes out of bounds. The heuristic selects it for a narrow
  `n` band (observed `n≈5407`), so it surfaces only under imbalanced routing that
  happens to produce such a per-expert token count. In training the OOB write hits
  a read-only page → `Memory access fault`; with the caching allocator on it reads
  stale `~3e38` bytes → sparse NaN (exactly §8's two-symptoms-one-bug picture).
- **Standalone repro now exists** (both numeric §11.3 and fault §11.5), so a
  hipBLASLt bug report no longer needs the full training pipeline — attach the
  `hipblaslt-bench --solution_index 332814 -v` line and `repro_332814.cpp`.
- **Mitigation:** force hipBLASLt to avoid 332814 via
  `HIPBLASLT_TUNING_OVERRIDE_FILE` (pin the offending `(T/N, m, k, bf16)` problem
  to a known-good solution, e.g. 335459); plus the layout-level workarounds in §10
  (`moe_expert_capacity_factor` / load balancing) to avoid the triggering `n`.
- **Follow-up:** apply the same `hipblaslt-bench --algo_method all -v` sweep across
  the other qwen3 per-expert GEMM shapes (the `linear_fc2` down-projection, and the
  `linear_fc1` over its full `n` range) to enumerate which `(shape, n)` combinations
  the heuristic maps onto a broken solution.

---

## Appendix A — Config variants produced

```
examples/megatron/configs/MI355X/
  qwen3_5_35B_A3B-BF16-nanrepro.yaml            # baseline: TE grouped GEMM (repro)
  qwen3_5_35B_A3B-BF16-nanrepro-turbogg.yaml    # use_turbo_grouped_gemm: true
  qwen3_5_35B_A3B-BF16-nanrepro-legacygg.yaml   # moe_use_legacy_grouped_gemm: true
  qwen3_5_35B_A3B-BF16-nanrepro-nogg.yaml       # moe_grouped_gemm: false (SequentialMLP)
  qwen3_5_35B_A3B-BF16-nanrepro-alltoall.yaml   # use_turbo_deepep: false (alltoall)
  qwen3_5_35B_A3B-BF16-nanrepro-cp2.yaml        # context_parallel_size: 2 (unsupported)
  qwen3_5_35B_A3B-BF16-nanrepro-cp4.yaml        # context_parallel_size: 4 (unsupported)
  qwen3_5_35B_A3B-BF16-nanrepro-nocommstream.yaml  # turbo_deepep_use_comm_stream: false
```

## Appendix B — Env vars

Stock env vars used for the decisive controls (no code changes needed):

```bash
PYTORCH_NO_CUDA_MEMORY_CACHING=1   # bypass caching allocator -> OOB faults instead of stale-NaN
AMD_SERIALIZE_KERNEL=3             # serialize kernel execution (rule out races)
HIP_LAUNCH_BLOCKING=1              # synchronous launches
TORCH_USE_HIP_DSA=1                # device-side assertions
HIPBLASLT_LOG_LEVEL=2              # log every rocblaslt_matmul (last one before the fault = culprit)
```

`PRIMUS_NAN_*` env vars referenced above belonged to the temporary instrumentation
(§4), which has been reverted; recreate that instrumentation to use them.
