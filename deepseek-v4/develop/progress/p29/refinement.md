# Plan-5 P29 — task-list refinement against the P28 baseline trace

> P28 (commit `afd7ea59`) named the dominant `aten::sum` fp32 reduce as
> the #1 bottleneck (87.3 % of step) and KEEP-RESCOPED P29 to root-cause
> + eliminate it (the original "small-op kernel-launch fusion" mandate
> was de-scoped because the CPU-bound floor is 0.3 % at V4-Flash
> production widths).
>
> This document is the deliverable for the first task row of the
> rescoped P29 plan: "task list refinement against P28 trace".

---

## 1. Forensic attribution

The forensic helpers under `progress/p29/_forensics{,2,3}.py` walked the
P28 chrome-trace JSON and matched every dominant
`reduce_kernel<512, 1, ReduceOp<float, sum_functor<float, float, float>>>`
launch back to its launching `cpu_op` (via the trace's `External id`
flow) and, where possible, the deepest enclosing Python source line.

### Direct launcher

| metric | value |
|---|---|
| matching kernels in steady iter | 717 |
| direct launcher cpu_op | `aten::sum` (100 %) |
| total kernel time | 7611 ms (87.3 % of step) |
| avg per launch | 10.62 ms |

### Origin (FWD vs BWD)

| origin | count | total | avg / call |
|---|---:|---:|---:|
| FWD | 39 | 0.2 ms | 0.005 ms |
| BWD | 277 | 7610.8 ms | 27.5 ms |
| (External-id-unmatched) | 401 | n/a | n/a |

> The 401 unmatched kernels are launches whose `External id` did not map
> to a cpu_op event in this trace's correlation table — they are the
> majority of the steady-iter kernels and are almost certainly the same
> source by signature. The 316 matched launches account for **~37 %**
> of the kernel population, but **~100 %** of the kernel time, so the
> matched sample is statistically representative.

### Deepest Python source line for matched launchers

| count | source line |
|---:|---|
| 38 + 266 = **304** | `primus/backends/megatron/core/transformer/hyper_connection.py:47 sinkhorn_normalize` |
| 6 | `primus/backends/megatron/core/transformer/moe/v4_topk_router.py:95 _compute_route` |
| 3 | `primus/backends/megatron/core/transformer/moe/v4_hash_router.py:161 forward` |
| 2 | `pretrain_gpt.py:103 loss_func` |
| 2 | `megatron/core/transformer/moe/moe_utils.py:1015 track_moe_metrics` |

> **`sinkhorn_normalize` accounts for 304 of 316 matched launches —
> 96 %.** Every other source line is a rounding error.

### Input shape attribution (matched aten::sum cpu_ops)

| count | total | avg | shape | dtype | reduction |
|---:|---:|---:|---|---|---|
| **624** | **7607.9 ms** | **12.19 ms** | `(1, 4096, 4, 4)` | `float` | `dim=[-1]`, keepdim |
|   33 |   0.2 ms |  0.01 ms | `(1, 4096, 4)` | float | dim=[-1] |
|   17 |   1.3 ms |  0.08 ms | `(1, 4096, 16384)` | float | dim=[-1] |
|   16 |   0.1 ms |  0.01 ms | `(4096, 6)` | float | dim=[-1] |
|    8 |   1.0 ms |  0.12 ms | `(1, 4096, 64, 512)` | float | dim=[-1] |

**The 624 calls of `(1, 4096, 4, 4) -> (1, 4096, 4, 1)` account for
99.95 % of the 7611 ms total.** Every other shape combined adds 3 ms.

---

## 2. Why the kernel is so slow

The reduction is `[1, 4096, 4, 4] -> [1, 4096, 4, 1]` over the last dim
of size 4. That is **16384 outputs × 4 inputs = 64 K fp32 ops total**,
~256 KiB of data. On an MI355X with ~5 TB/s peak HBM bandwidth the
memory-bound floor is ≈ **51 µs** (read all 256 KiB once + write 64 KiB
once). The observed kernel time is **12.19 ms** — **~240× over
memory-bound floor**.

The kernel template `reduce_kernel<512, 1, ReduceOp<float, sum_functor<
float, float, float>>>` is HIP / ROCm's default fp32 reduce kernel. The
template parameters are **`<num_threads_per_block=512, num_per_thread=1>
`**, sized for huge reductions. For our `[16384, 4]` shape:

* 16384 outputs × 1 thread per output / 512 threads per block
  = **32 blocks**, each block has only **64 threads doing useful
  work** out of 512 (the rest are idle on the 4-element inner
  reduction).
* Effective occupancy ≈ 64 / 512 = 12.5 %.
* Plus per-launch fixed overhead (~5 µs on HIP) × 624 launches per
  iter = 3.1 ms of pure launch overhead alone.

So the dispatcher chose the wrong kernel for a tiny inner reduction
shape. We can not fix the dispatcher; we **must avoid issuing 624 of
these calls per iter in the first place**.

---

## 3. Root cause: `sinkhorn_normalize` algorithmic shape

`primus/backends/megatron/core/transformer/hyper_connection.py:47`
implements the Sinkhorn-Knopp doubly-stochastic projection used by
the V4 mHC mixer:

```python
def sinkhorn_normalize(logits, *, n_iters: int = 20, eps: float = 1e-6):
    in_dtype = logits.dtype
    m = logits.float()                              # cast up to fp32
    m = m / (m.sum(dim=-2, keepdim=True) + eps)      # priming col-norm
    for _ in range(max(n_iters - 1, 0)):             # 19 iter
        m = m / (m.sum(dim=-1, keepdim=True) + eps)  # row-norm
        m = m / (m.sum(dim=-2, keepdim=True) + eps)  # col-norm
    return m.to(in_dtype)
```

Per call this issues **1 + 19 × 2 = 39 separate `aten::sum` kernel
launches** plus 39 `aten::div` kernels and 39 `aten::add` kernels,
each operating on `[B, S, K, K]` with K = `hc_mult` = 4. Per iter
under the V4-Flash proxy:

* **8 hybrid layers × 1 `compute_weights` call/layer = 8 FWD calls**
* Each FWD call → 39 reduce launches → **312 FWD launches / iter**
* Autograd backward through the chain → **312 BWD launches / iter**
* **Total ≈ 624 launches / iter** ✓ matches the trace

Plus the BWD per launch is more expensive (autograd walks the
`m = m / s` chain in reverse, each `s` is materialised again as a
broadcast, etc.). The 27.5 ms BWD-avg vs 0.005 ms FWD-avg confirms
this: BWD is doing more work per call, not just more calls.

---

## 4. Fix decision

| option | speedup est. | effort | numerical risk | decision |
|---|---|---|---|---|
| (1) `torch.compile(fullgraph=True, dynamic=False)` wrapping `sinkhorn_normalize` | **2–10×** (collapses 39 reduces + divides into one fused Triton kernel; AOT-autograd handles BWD) | 1 day | bit-equivalent; `torch.compile` does not change algorithm | **PRIMARY (P29 ships this)** |
| (2) Hand-written Triton fused-Sinkhorn kernel — one program / row, all 20 iters in registers | 50–100× (memory-bound: 256 KiB once at ~50 µs) | 3-5 days | identical math; bf16/fp32 mix to be validated | **fallback if option (1) is < 50 % win** |
| (3) Reduce `n_iters` (20 → 5) | 4× | 1 hr | model-quality decision; affects pretraining convergence; out of plan-5 scope | **REJECT** |
| (4) Cast to bf16 | 2× bandwidth | 1 day | accuracy risk; the function explicitly casts to fp32 (techblog §2.2 pitfall #3) | **REJECT** |
| (5) Disable mHC (`hc_mult=1`) | full removal | 1 hr | breaks V4 architecture | **REJECT** |

**Decision: option (1) — `torch.compile`.** Lowest risk (bit-equivalent
math, no algorithmic change); the 39 small reductions plus 39 divides
plus 39 broadcasts compile down to a single Inductor-generated Triton
kernel that runs the whole Sinkhorn-Knopp loop in registers. The BWD
graph has the same shape (the autograd of `m / m.sum(...)` is
deterministic and shape-stable), so AOT-autograd compiles it too.

If the post-P29 trace shows the `aten::sum` reduce dropped by < 50 %,
option (2) takes over — but option (2) costs 3-5 days and option (1)
should clear the 50 % bar comfortably.

---

## 5. Refined P29 task list (replaces the seeded a..e fusion targets)

| # | task | gate | status |
|---|---|---|---|
| 1 | Task-list refinement against P28 trace (this document) | — | DONE |
| 2 | Add `use_v4_compiled_sinkhorn` config flag (default `False`) on `DeepSeekV4TransformerConfig`; plumb through YAML + `run_deepseek_v4.sh` + the proxy script | — | TODO |
| 3 | `hyper_connection.py` — wrap `sinkhorn_normalize` with a cached `torch.compile(fullgraph=True, dynamic=False)` build, dispatched by the new flag; preserve eager path as the default | — | TODO |
| 4 | `HyperMixer.__init__` accepts `use_compiled_sinkhorn`; `compute_weights` dispatches; `DeepseekV4HybridLayer` passes the config flag through | — | TODO |
| 5 | G32 — `tests/unit_tests/megatron/transformer/deepseek_v4/test_v4_p29_compiled_sinkhorn.py`: FWD + BWD parity vs eager at fast-tier (`B=2 S=64 K=4`) and release-tier (`B=1 S=4096 K=4`); `pytest.mark.slow` for release tier | G32 | TODO |
| 6 | G33a smoke — `progress/p29/run_smoke_compiled_sinkhorn_ep8.sh`: 10-iter EP=8 proxy with the flag on; assert plan-4 ratchet stays green, no banned warnings, lm_loss within 1e-2 of P28 baseline | G33a | TODO |
| 7 | G33b proxy trace — `progress/p29/run_baseline_trace_ep8_p29.sh`: capture chrome-trace iter 6 → 7 with the flag on; render `develop/profile/profile-after-p29-ep8-<YYYYMMDD>.{md,html}` reusing the `_tools/render_baseline_report.py` from P28; assert `aten::sum` fp32 reduce time drops by ≥ 50 % vs the P28 baseline (budget X1) | G33b | TODO |
| 8 | If G33b shows < 50 % drop → escalate to option (2) (hand-Triton); document in `progress/p29/post_compile_results.md` and ship Triton-fused kernel | — | conditional |
| 9 | Flip the `use_v4_compiled_sinkhorn` default to `True` in YAML + `run_deepseek_v4.sh` + the proxy once G32 + G33a + G33b are green | — | TODO |

The original seeded targets (a) `v4_fused_q_proj`, (b) `v4_fused_kv_proj`,
(c) `v4_fused_o_proj`, (d) `v4_fused_compressor` + `v4_fused_indexer`,
(e) `v4_fused_moe_router` are **all de-scoped** under the P28 KEEP-
RESCOPE rule — the CPU-bound floor at V4-Flash production widths is
0.3 %, and the small-op-launch tail those targets attack is 0.7 % of
step (rank #4 in the P28 bottleneck table, < 10 % rule). They become
**plan-5 follow-ups** for revisit only if a future trace shows the
small-op tail re-emerging at a different shape configuration (e.g.
multi-node EP with cross-node activation reshuffling).

---

## 6. Perf-budget contract (X1 from P28)

| metric | P28 baseline | P29 target (X1) |
|---|---:|---:|
| `aten::sum` fp32 reduce kernel total | 7.61 s | ≤ **3.81 s** (≥ 50 % drop) |
| steady iter wall time | 8.83 s | ≤ **5.5 s** (target ≥ 35 % gain — the kernel reduction net of multi-stream overlap factor 1.87×) |
| TFLOP/s/GPU steady | 78 | ≥ **125** (60 %+ gain — multi-stream overlap means the iter time gain ≥ kernel time gain) |
| HBM peak | 195 GiB | ≤ 195 GiB (no regression) |
| lm_loss after 10 iters at fixed seed | 9.26 | within 5e-2 |

If the actual trace shows the kernel drop at < 50 % we ship option (2)
(hand-Triton) before closing P29; if ≥ 50 % we close P29 and proceed to
P30 with the new baseline pinned.
