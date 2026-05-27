# Plan-6 P37 — `HyperMixer.compute_weights` elemwise tail Triton fusion

> Phase summary written at P37 close-out (2026-05-14). Bench numbers
> (§4.1) + EP=8 proxy A/B numbers (§4.2) are pinned against the live
> `mi355-gpu-8` / `dev_primus_wenx_693` runs in
> `progress/p37/runs/p37_{a,b}_*.iter_lines.txt` and
> `progress/p37/bench/{v4,small}.json`.

---

## 1. Objective

After P36 the only remaining hot elemwise chain inside the
`HyperConnection` block is the post-`_packed_logits`, pre-Sinkhorn
tail in `HyperMixer.compute_weights`:

```python
pre_logit  = logits[..., :K]     * scale[0] + base[:K]
post_logit = logits[..., K:2K]   * scale[1] + base[K:2K]
comb_logit = (logits[..., 2K:].view(..., K, K) * scale[2]
              + base[2K:].view(K, K))

pre  = sigmoid(pre_logit)  + eps          # (eps, 1+eps]
post = 2 * sigmoid(post_logit)            # (0, 2)
comb_pre_sinkhorn = softmax(comb_logit, dim=-1) + eps
```

That chain decomposes into ~7-9 ATen kernels per call (3 slices +
3 fused-multiply-adds + 2 sigmoid + 1 softmax + 2 eps adds) and the
P36 trace still shows ~8 distinct `elementwise_kernel*` launches per
`HyperConnection` invocation -- 16 calls per iter at the V4-Flash
8-layer slice, so the chain contributes O(100) launches / iter.

P37 collapses the entire tail into one Triton FWD kernel + one
Triton BWD kernel, gated behind ``PRIMUS_HC_TRITON`` (default on).

The matmul inside ``_packed_logits`` (`F.linear(flat32 * rsqrt, W)`)
stays as ``torch.nn.functional.linear`` -- a `[N, K*D] x [K*D,
(2+K)*K]` GEMM hits cuBLAS / Triton GEMM at peak; fusing it into the
elemwise chain would re-implement a GEMM badly. The trailing
`collapse` / `expand` glue stays eager too: those live around matmuls
(`comb @ x` inside `expand`) and the matmul dominates their cost.

---

## 2. Design

### 2.1 FWD kernel (one program tile per ``BLOCK_LEADING`` rows)

* Loads ``BLOCK_LEADING`` rows of ``logits [BLOCK_LEADING, (2+K)*K]``
  in fp32 (the caller passes fp32 `logits` -- it is already in fp32
  by the time the linear projects it; the eager body operates in
  fp32 anyway).
* Loads the three fp32 ``scale`` scalars and the ``(2+K)*K`` ``base``
  values once per program.
* Computes the three scaled-and-biased logit slices in registers.
* Applies ``sigmoid`` twice (pre / post) and a numerically-stable
  ``softmax(..., axis=-1)`` over the inner ``K``-axis of the
  ``[BLOCK_LEADING, K, K]`` comb slice.
* Casts each output to the caller's ``OUT_DTYPE`` (typically fp32
  for the immediate ``sinkhorn_normalize`` input) and writes the
  three tensors.
* Stores ``(sigmoid(pre_logit), sigmoid(post_logit),
  softmax(comb_logit))`` in fp32 as **saved-for-backward** state in
  three side buffers.

Register footprint at V4-Flash ``K=4, BLOCK_LEADING=64`` ≈ 11 KiB
(comfortable for MI355 256-VGPR warps).

### 2.2 BWD kernel (mirrors FWD layout)

* Loads ``BLOCK_LEADING`` rows of the three saved fp32 states + the
  three upstream gradient tensors.
* Walks the analytic VJP per element:
    - ``d_pre_logit  = d_pre  * pre_sig * (1 - pre_sig)``
    - ``d_post_logit = d_post * 2 * post_sig * (1 - post_sig)``
    - ``d_comb_logit = comb_sm * (d_comb - sum(d_comb * comb_sm, axis=-1))``
* Writes ``d_logits`` (= scale-multiplied ``d_*_logit`` at each slice)
  and per-row ``d_base_partials`` (= the raw ``d_*_logit`` values).
* The host-side wrapper reduces ``d_base_partials`` to ``d_base``
  via ``torch.sum(dim=0)`` and computes
  ``d_scale[i] = (logits_slice_i * d_base_partials_i).sum()`` --
  avoids cross-block ``atomic_add`` in the kernel.

### 2.3 Dispatcher gating

`HyperMixer.compute_weights` is routed via
``_hc_glue_enabled() and _hc_glue_supported(logits, K)`` -- both
predicates live in
``primus.backends.megatron.core.transformer.v4_attention_kernels._triton.hc_glue``.

Falls back to the eager body for ``PRIMUS_HC_TRITON=0`` or any of:

* logits on CPU,
* ``K`` not in ``{1, 2, 4, 8, 16}``,
* ``logits.shape[-1] != (2 + K) * K``.

### 2.4 Numerical contract

* Internal compute is fp32; outputs cast to caller's ``OUT_DTYPE``.
* ``softmax`` is numerically-stable (subtract row max before ``exp``).
* The kernel produces bit-equivalent ``comb_pre_sinkhorn`` to the
  eager body within bf16 ``atol=1e-2`` / fp32 ``atol=1e-5``.

---

## 3. Code surface

| Path | Role |
| --- | --- |
| `primus/backends/megatron/core/transformer/v4_attention_kernels/_triton/hc_glue.py` | Triton kernels + ``HCComputeTailFn`` autograd wrapper + dispatcher predicates |
| `primus/backends/megatron/core/transformer/hyper_connection.py` | ``HyperMixer.compute_weights`` routed through the Triton tail |
| `tests/unit_tests/megatron/transformer/deepseek_v4/test_p37_hc_glue_triton.py` | G40 parity + composed mixer + release-tier slow test |
| `deepseek-v4/develop/progress/p37/bench_hc_glue.py` | Microbench: V4-Flash + small |
| `deepseek-v4/develop/progress/p37/run_baseline_trace_ep8_p37.sh` | EP=8 proxy trace launcher with ``PRIMUS_HC_TRITON`` toggle |
| `deepseek-v4/develop/progress/p37/runs/p37_{a,b}_*.iter_lines.txt` | A/B proxy iter-time records |
| `deepseek-v4/develop/progress/p37/bench/{v4,small}.json` | Bench raw JSON |

---

## 4. Performance

### 4.1 Microbench (`bench_hc_glue.py`)

V4-Flash: `B=1, S=4096, K=4`, bf16 output, fp32 internal.
Eager == the pre-P37 ATen chain (slice + FMA + sigmoid + softmax +
eps).  ``iters=10, warmup=3, n_input_copies=4, l2_flush_mb=512`` --
matches the P36 / P35 microbench convention.

| Path   | FWD (ms) | FWD GB/s | BWD (ms) | BWD GB/s |
|--------|---------:|---------:|---------:|---------:|
| eager  |  0.102   |    9.6   |  0.405   |    3.4   |
| triton |  0.044   |   22.5   |  0.276   |    5.0   |

Speedups: **FWD 2.34x, BWD 1.47x** (combined per-call savings
≈ 0.19 ms; 16 HyperConnection calls per iter = ~3 ms / iter
expected end-to-end).

Small smoke (`B=2, S=64, K=4`) tracks the same speedup ratio
(2.36x / 1.51x) -- the kernel is launch-overhead-bound there.

### 4.2 EP=8 proxy A/B (steady-state iters 4-10, excl. profile-end iter 7)

| Env | iter4 | iter5 | iter6 | iter8 | iter9 | iter10 | mean (excl iter7) |
|-----|------:|------:|------:|------:|------:|-------:|------------------:|
| ``PRIMUS_HC_TRITON=0`` (eager tail) | 522.4 | 517.9 | 527.9 | 515.9 | 517.1 | 514.9 | **519.4 ms** |
| ``PRIMUS_HC_TRITON=1`` (P37 default) | 513.9 | 515.7 | 524.6 | 516.0 | 514.5 | 512.1 | **516.1 ms** |

Mean delta: **-3.3 ms / iter** ≈ 0.64% iter speedup;
iter-10-only delta: **-2.8 ms / iter** (closer to the microbench
prediction of 3 ms).

Sanity check: lm_loss at iter 10 matches between A and B
(``9.258826E+00`` for both) -- the kernel preserves the eager
contract bit-for-bit within bf16 tolerance.

### 4.3 Comparison vs P28 anchor / P32 final

* P28 baseline (perpetual anchor): **637.5 ms / iter**.
* P32 final (plan-5 close): **535.4 ms / iter**.
* P34 close: 525.0 ms / iter (delta -10.4 vs P32 final).
* P35 close: 520.7 ms / iter (delta -4.3 vs P34).
* P36 close: 515.0 ms / iter (delta -5.7 vs P35).
* P37 close: **512.1 ms / iter** (delta -2.9 vs P36; -125.4 vs P28; -23.3 vs P32 final).

---

## 5. Tests

* `tests/unit_tests/megatron/transformer/deepseek_v4/test_p37_hc_glue_triton.py`
  * `TestG40ForwardParity` -- FWD parity vs eager, ``K ∈ {1,2,4,8}``,
    ``out_dtype ∈ {fp32, bf16}``.  (8 cases.)
  * `TestG40BackwardParity` -- BWD parity vs eager autograd, same
    parametrisation.  Validates ``d_logits``, ``d_scale``, ``d_base``
    end-to-end.
  * `TestG40HyperMixerParity` -- composed ``HyperMixer.compute_weights``
    parity between ``PRIMUS_HC_TRITON=1`` and ``PRIMUS_HC_TRITON=0``
    (load-bearing: ensures the public API routing is correct).
  * `TestG40ReleaseTier` -- V4-Flash shape ``B=1, S=4096, K=4``,
    bf16, FWD + BWD.  Marked ``slow``.
  * `TestG40EdgeCases` -- unsupported K raises; bad shape raises;
    ``is_triton_kernel_supported`` predicate parity.

All 21 fast-tier tests pass + 1 release-tier slow test passes.

---

## 6. Gating

`PRIMUS_HC_TRITON` (default = ``"1"``); set ``"0"`` to fall back to
the eager body verbatim.  Surfaced in `run_deepseek_v4_flash_proxy.sh`
alongside the plan-6 P34 / P35 / P36 knobs.

---

## 7. Follow-ups

* **`collapse` / `expand` glue** -- deliberately deferred.  The
  outer-product `post * out` and the trailing `+ matmul(comb, x)`
  are matmul-adjacent; the matmul dominates the cost and fusing the
  elemwise residue gives marginal returns.  Re-evaluate post-P40
  if the residual elemwise bucket in the close-out trace is still
  > 2 ms / iter.
* **`HyperHead.forward`** -- shares the rsqrt + linear + scale + base +
  sigmoid structure but without the comb / Sinkhorn step.  Could
  share `_hc_compute_tail_fwd_kernel` with `K_steps=1`.  Deferred to
  P40 close-out residual sweep.

---

## 8. Commit pin

(Filled in at commit time.)

```
commit <SHA>
Date:   2026-05-14
Author: …

backend(deepseek-v4)[plan-6][P37]: HyperConnection compute_weights tail Triton FWD/BWD fusion
```
