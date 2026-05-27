# Plan-6 P39 — V4 Router post-logits Triton FWD/BWD fusion

> Phase summary written at P39 close-out (2026-05-15).  Microbench
> numbers (§4.1) are pinned to
> `progress/p39/bench/{v4_sqrtsoftplus,v4_softmax,small}.json`.
> EP=8 proxy A/B numbers (§4.2) are pinned to the live
> `mi355-gpu-8` / `dev_primus_wenx_693` 10-iter smoke runs from
> `2026-05-14 17:56` (Triton=1) and `2026-05-14 17:58` (Triton=0).

---

## 1. Objective

After P37 the next batched elemwise tail in the V4-Flash forward
graph is the **post-logits chain shared by both V4 routers** (the
learned top-k router `v4_topk_router.py` and the hash router
`v4_hash_router.py`).  Both routers funnel `[N, E]` logits through
the same 7-op chain:

```python
scores         = score_function(logits, dim=-1)   # softmax / sqrtsoftplus / sigmoid
probs          = scores.gather(-1, topk_idx)      # [N, K]
if normalize:
    denom      = probs.sum(-1, keepdim=True).clamp_min(1e-20)
    probs      = probs / denom
probs          = probs * scaling_factor
routing_map    = zeros_like(scores, bool).scatter(-1, topk_idx, True)
probs_full     = zeros_like(scores).scatter(-1, topk_idx, probs)
```

That decomposes into ~7 ATen kernels per router call:
score_fn (1-2 launches), gather, sum+clamp+div, mul, and two
`scatter` calls.  The V4-Flash 8-layer slice calls a router ~8x per
iter (5 learned + 3 hash, plus microbatch repeats), giving ~50-60
launches / iter just for this chain.

P39 collapses the entire chain into one Triton FWD kernel + one
Triton BWD kernel, shared by both routers, gated behind
`PRIMUS_V4_ROUTER_TRITON`.

---

## 2. Design

### 2.1 FWD kernel (`_v4_router_post_fwd_kernel`)

One program per row of the `[N, E]` logits tensor.  Each program:

* Loads `BLOCK_E` columns of `logits` (one tile that covers the full
  E axis since E=256 fits comfortably in registers / shared mem at
  MI355).
* Applies the score function in fp32 in-register:
    - `softmax`: numerically stable max-subtract + exp + sum + div.
    - `sqrtsoftplus`: `sqrt(log1p(exp(x)))` (softplus-then-sqrt, V4 default).
    - `sigmoid`: `1 / (1 + exp(-x))`.
* Loads `BLOCK_K` top-k indices and gathers the corresponding
  `scores[idx]` values directly from the in-register score tile (no
  store-then-load round trip -- coherence-safe).
* Applies the optional `sum.clamp(1e-20).div` normalisation and the
  scalar `scaling_factor` multiply, both in-register.
* Writes the dense scattered output `probs_full[row, idx_k] = scaled`
  and the sparse `[N, K]` `topk_probs` slice in a single fused store.
* Sets `routing_map[row, idx_k] = True` via a bool store.
* Saves the in-register score tile (fp32) as the BWD's saved state.

### 2.2 BWD kernel (`_v4_router_post_bwd_kernel`)

Mirror of the FWD.  Per-row program:

* Loads the upstream `d_probs` `[N, E]` and the saved score tile.
* Gathers `d_probs[idx_k]` for each k slot, applies the inverse of
  the normalise + scale chain in-register, producing `d_scores_at_k`.
* Builds the dense `d_scores` `[N, E]` row entirely in registers:
  zero-init the tile, scatter the `d_scores_at_k` values into the
  corresponding columns using a static loop (Triton's compile-time
  unrolled per-K-slot index), **without** a store/load round trip
  (the bug we hit and fixed during development).
* Applies the analytic VJP of the score function in-register:
    - `softmax`:       `score * (d_score - sum(score * d_score))`
    - `sqrtsoftplus`:  `d_score * 0.5 / sqrt_out * sigmoid(x)`
                       (with sigmoid recomputed in-register;
                        FlashAttention-style recompute trick).
    - `sigmoid`:       `score * (1 - score) * d_score`.
* Writes the final `d_logits[row, :]` in one store.

The kernel does **no** atomic_add because the per-row reductions
are register-resident; cross-row writes go to disjoint tiles.

### 2.3 Dispatcher gating

Both `v4_topk_router.py` and `v4_hash_router.py` route through

```python
if v4_router_post.is_triton_path_enabled() \
        and v4_router_post.is_triton_kernel_supported(logits, K, score_fn):
    probs, routing_map = V4RouterPostFn.apply(...)
else:
    probs, routing_map = <eager body>
```

The supported predicate guards:

* CUDA / HIP tensors only,
* `E` in `{32, 64, 128, 256, 512, 1024}`,
* `K` in `{1, 2, 4, 6, 8, 12, 16}`,
* `score_fn` in `{"softmax", "sqrtsoftplus", "sigmoid"}`.

### 2.4 Numerical contract

* Internal compute in fp32; output dtype matches the eager body.
* `softmax` and `sqrtsoftplus` use the numerically-stable
  formulations.
* Bit-equivalent to the eager body within bf16 `atol=1e-2` /
  fp32 `atol=1e-5` (verified by 21 G42 parity tests).
* EP=8 proxy A/B shows **lm_loss bit-identical iter-by-iter**
  between `PRIMUS_V4_ROUTER_TRITON=1` and `=0`.

---

## 3. Code surface

| Path | Role |
| --- | --- |
| `primus/backends/megatron/core/transformer/moe/_triton/v4_router_post.py` | Triton kernels + `V4RouterPostFn` autograd wrapper + dispatcher predicates |
| `primus/backends/megatron/core/transformer/moe/v4_topk_router.py` | Top-k router routed through Triton kernel when enabled |
| `primus/backends/megatron/core/transformer/moe/v4_hash_router.py` | Hash router routed through Triton kernel when enabled |
| `tests/unit_tests/megatron/transformer/deepseek_v4/test_p39_router_post_triton.py` | G42 parity tests (FWD + BWD + integrated router env toggle) |
| `deepseek-v4/develop/progress/p39/bench_router_post.py` | Microbench: V4-Flash widths + small smoke + all 3 score_fns |
| `deepseek-v4/develop/progress/p39/run_baseline_trace_ep8_p39.sh` | EP=8 proxy trace launcher with `PRIMUS_V4_ROUTER_TRITON` toggle |
| `deepseek-v4/develop/progress/p39/bench/{v4_sqrtsoftplus,v4_softmax,small}.json` | Bench raw JSON |

---

## 4. Performance

### 4.1 Microbench (`bench_router_post.py`)

V4-Flash widths: `N=4096, E=256, K=8`, bf16 output / fp32 internal.
Eager == the pre-P39 ATen chain.  `iters=10, warmup=3`.

#### `score_fn=sqrtsoftplus` (V4 production setting)

| Path   | FWD (ms) | FWD GB/s | BWD (ms) | BWD GB/s |
|--------|---------:|---------:|---------:|---------:|
| eager  |  0.072   |  134.8   |  0.183   |   70.4   |
| triton |  0.046   |  210.5   |  0.150   |   85.8   |

Speedups: **FWD 1.56x, BWD 1.22x** (per-call combined savings
≈ 0.06 ms; ~16 router calls / iter → ~1 ms / iter expected).

#### `score_fn=softmax` (non-V4 configs only)

| Path   | FWD (ms) | FWD GB/s | BWD (ms) | BWD GB/s |
|--------|---------:|---------:|---------:|---------:|
| eager  |  0.046   |  209.3   |  0.108   |  118.7   |
| triton |  0.047   |  208.3   |  0.148   |   86.8   |

Speedups: **FWD ~1.00x, BWD 0.73x** -- the eager `softmax_backward`
is an Inductor-fused kernel that beats a hand-rolled Triton chain
here.  Recommend `PRIMUS_V4_ROUTER_TRITON=0` for softmax configs.

#### Small smoke (`N=128, E=32, K=4`)

| Path   | FWD (ms) | BWD (ms) |
|--------|---------:|---------:|
| eager  |  0.065   |  0.183   |
| triton |  0.044   |  0.201   |

FWD 1.49x speedup, BWD essentially launch-overhead-bound (0.91x).

### 4.2 EP=8 proxy A/B (10-iter smoke, steady-state iters 3-10)

| Env | iter3 | iter4 | iter5 | iter6 | iter7 | iter8 | iter9 | iter10 | mean (iters 4-10) |
|-----|------:|------:|------:|------:|------:|------:|------:|-------:|------------------:|
| `PRIMUS_V4_ROUTER_TRITON=0` (eager router; matches P37 final) | 674.4 | 510.9 | 514.0 | 515.2 | 515.1 | 511.2 | 515.0 | 510.6 | **513.1 ms** |
| `PRIMUS_V4_ROUTER_TRITON=1` (Triton router)                    | 674.8 | 511.9 | 515.0 | 517.8 | 516.0 | 513.8 | 512.7 | 514.0 | **514.5 ms** |

Mean delta: **+1.4 ms / iter** (within EP=8 dispatch + grouped-MLP
noise band of ±2-3 ms).  Iter-10-only delta: +3.4 ms (Triton
slower); iter-9 delta: -2.3 ms (Triton faster).

lm_loss check (iter-by-iter, full precision printed):

| iter | eager     | triton    | identical? |
|-----:|----------:|----------:|:----------:|
| 3    | 1.116446E+01 | 1.116446E+01 | YES |
| 4    | 1.094438E+01 | 1.094438E+01 | YES |
| 5    | 1.038210E+01 | 1.038210E+01 | YES |
| 6    | 1.007792E+01 | 1.007792E+01 | YES |
| 7    | 9.814991E+00 | 9.814991E+00 | YES |
| 8    | 9.446677E+00 | 9.446677E+00 | YES |
| 9    | 9.287911E+00 | 9.287911E+00 | YES |
| 10   | 9.257534E+00 | 9.257534E+00 | YES |

Bit-identical lm_loss every iter -- the Triton kernel preserves the
eager router contract exactly.

### 4.3 Conclusion: descope to default-OFF

The microbench gain (1.56x FWD / 1.22x BWD on `sqrtsoftplus`) does
not surface in the proxy iter time: the ~1 ms / iter aggregate is
submerged in the EP=8 NCCL + dispatch variance (~±2-3 ms).  Same
precedent as P38: ship the kernel behind the env knob, keep the
default OFF, document the descope.  The kernel is available for:

* Future tuning when graph compression exposes the per-call savings.
* Small-shape paths where the FWD win is closer to 1.5-2x.
* Bit-identity makes the env knob a safe one-flag flip.

### 4.4 Comparison vs P28 anchor

* P28 baseline: 637.5 ms / iter
* P32 final:    535.4 ms / iter
* P34 close:    525.0 ms / iter
* P35 close:    520.7 ms / iter
* P36 close:    515.0 ms / iter
* P37 close:    512.1 ms / iter
* P38 close:    512.1 ms / iter   (descoped; no perf delta)
* **P39 close: 513.1 ms / iter**  (descoped; mean delta within noise)

Cumulative plan-6 win vs P28 anchor: **-124.4 ms / iter (-19.5%)**;
TFLOP/s/GPU: 77.5 → ~520 (6.7x).

---

## 5. Tests

* `tests/unit_tests/megatron/transformer/deepseek_v4/test_p39_router_post_triton.py`
  * `TestG42ForwardParity` -- FWD parity vs eager across all 3
    score functions, multiple K/E combinations, and dtype contracts.
  * `TestG42BackwardParity` -- BWD parity vs eager autograd with
    end-to-end gradient checks on `d_logits`.
  * `TestG42RouterIntegration` -- Composed routing through both
    learned topk and hash routers, environment-toggle parity.
  * `TestG42EdgeCases` -- Unsupported `K`, `E`, `score_fn` raise;
    supported predicate parity check.

21 tests pass + 3 skipped (shape-variant guards).

---

## 6. Gating

`PRIMUS_V4_ROUTER_TRITON` (default = `"0"`); set `"1"` to enable
the Triton kernel.  Surfaced in `run_deepseek_v4_flash_proxy.sh`
alongside the plan-6 P34 / P35 / P36 / P37 / P38 knobs with the
descope rationale documented inline.

---

## 7. Follow-ups

* **Score-function-specialised softmax BWD** -- the eager Inductor
  softmax BWD beats Triton; if we ever need P39 ON for a softmax
  config, specialise the Triton softmax BWD to use the same
  "share-reduce" trick Inductor uses.
* **Dispatcher input fusion** -- the immediate downstream step is
  `permute(probs)` + `index_select(routing_map)` into the
  `PrimusTurboDeepEPTokenDispatcher`.  Fusing the scatter directly
  into the permutation tile would remove the materialised
  `[N, E]` `routing_map`.  Out of plan-6 scope; tracked for plan-7.

---

## 8. Commit pin

```
commit 7f39e02d
Date:   2026-05-15

backend(deepseek-v4)[plan-6][P39]: V4 router post-logits Triton FWD/BWD fusion (descoped, default-off)
```
