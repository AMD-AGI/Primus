# Plan-6 P36 — `sinkhorn_normalize` Triton FWD/BWD fusion

> Phase summary written at P36 close-out (2026-05-14). Bench numbers
> (§4.1) + EP=8 proxy A/B numbers (§4.2) are pinned against the live
> `mi355-gpu-8` / `dev_primus_wenx_693` runs in
> `progress/p36/runs/{triton_on,triton_off}.iter_lines.txt` and
> `progress/p36/bench/{k4,k4_small,k8}.json`.

---

## 1. Objective

The plan-5 P32 final EP=8 proxy trace attributes:

* `Torch-Compiled Region` ≈ **21 ms / 16 calls** to the FWD side of the
  cached `torch.compile` Sinkhorn artefact (built in
  `_build_compiled_sinkhorn` in `hyper_connection.py`);
* `CompiledFunctionBackward`  ≈ **41 ms / 16 calls** to the BWD side
  of the same artefact (AOT-autograd-emitted Inductor graph).

Combined per-call cost is **~3.9 ms** (1.3 FWD + 2.6 BWD).
`torch.compile` collapses the `1 + 2 * (n_iters - 1) = 39` fp32
reductions / divides into one Inductor-generated Triton kernel, but
its Dynamo-side bookkeeping (the `Torch-Compiled Region` event itself
counts as overhead) plus the Inductor BWD graph are still a
non-trivial slice of per-iter time.

P36 replaces this with a hand-rolled Triton FWD + BWD kernel pair that:

1. Holds the `[K, K]` matrix per row of the leading axis in registers
   throughout the full normalize trajectory (V4-Flash `K=4`: `K*K=16`
   fp32 elements per row, comfortably within register budget);
2. Runs the priming col-normalize plus `n_iters - 1` `(row, col)`
   pairs in a single kernel launch (no Dynamo entry per call);
3. Writes the FWD trajectory `m_0, m_1, ..., m_{2*n_iters - 1}` to a
   small `[N, 2 * n_iters, K, K]` fp32 buffer in HBM during FWD; the
   BWD reads that buffer and walks the analytic VJP backward
   step-by-step.

Storing the FWD trajectory to HBM (~10 MiB / call at V4-Flash widths;
negligible vs the ~170 GiB rank-peak footprint) sidesteps a Triton
AST-visitor restriction in our toolchain that rejects runtime
indexing of Python lists holding `tl.tensor` bundles -- which would
otherwise have let us recompute the trajectory in registers during
BWD.

Per-step closed-form VJP:

```python
# forward:  y = x / s, where s = sum(x, axis=axis) + eps
# backward: dx = (dy - sum(dy * y, axis=axis, keep_dims=True)) / s
```

Both axes (row / col) have the same shape of VJP (Triton's `tl.sum`
handles either with the right `axis=` argument). The BWD kernel is
just the cached FWD-trajectory read followed by 39 of these VJP
steps in reverse order, pair-folded into a `col → row → col → row →
... → col_priming` walk so the `axis=` argument stays compile-time
Python int.

---

## 2. What changed

| component | path | change |
|---|---|---|
| Triton kernel | `primus/backends/megatron/core/transformer/v4_attention_kernels/_triton/sinkhorn.py` (new) | `_sinkhorn_fwd_kernel` (forward + state-trajectory cache) + `_sinkhorn_bwd_kernel` (cached-state VJP walk) + `SinkhornNormalizeFn(autograd.Function)` + `sinkhorn_normalize_triton(...)` dispatcher gated on `PRIMUS_SINKHORN_TRITON` env (default `"1"`). Block heuristic: `BLOCK_LEADING ∈ {8, 32, 128}` by `K`. Supports `K ∈ {1, 2, 4, 8, 16}` and `{fp64, fp32, fp16, bf16}` inputs (internal compute is always fp32 -- matches the eager `m = logits.float()` contract). |
| Package docstring | `primus/backends/megatron/core/transformer/v4_attention_kernels/_triton/__init__.py` | Adds a new bullet for `sinkhorn`. |
| Wiring | `primus/backends/megatron/core/transformer/hyper_connection.py` | `sinkhorn_normalize` gains a `use_triton: bool = False` kwarg. Routing precedence: `use_triton or PRIMUS_SINKHORN_TRITON != "0" > use_compiled > eager`. Eager body kept in tree as the secondary fallback and as the G39 reference; the plan-5 P29 compiled body is still reachable via `use_compiled=True` AND `PRIMUS_SINKHORN_TRITON=0`. |
| Unit tests (G39) | `tests/unit_tests/megatron/transformer/deepseek_v4/test_p36_sinkhorn_triton.py` (new) | Fast tier (`B=2, S=64, K=4` × `{fp32, fp16, bf16}` × `n_iters ∈ {5, 20}`) FWD + BWD parity vs the eager body AND vs the plan-5 P29 compiled path, plus fp64 BWD parity via `torch.autograd` of the eager body (gradcheck at fp64 is incompatible with the fp32-internal compute contract -- the lossy fp64→fp32 cast wipes out the finite-difference delta). Release tier (`pytest.mark.slow`) covers V4-Flash production shape `B=1, S=4096, K=4` bf16. Doubly-stochastic property check (row / col sums equal 1 within `eps * K`) pinned independently of the eager path. Edge cases: non-square / unsupported-K / n_iters=0 raise `ValueError`. Env-flag dispatch test pins all three code paths through `hyper_connection.sinkhorn_normalize`. |
| G32 fixture | `tests/unit_tests/megatron/transformer/deepseek_v4/test_v4_p29_compiled_sinkhorn.py` | Adds an `autouse=True` `monkeypatch.setenv("PRIMUS_SINKHORN_TRITON", "0")` fixture so the file keeps testing the plan-5 P29 compiled-vs-eager boundary it was authored to test (otherwise the default-on Triton path silently hijacks all `use_compiled=True` calls and the compiled cache stays empty). |
| Microbench | `deepseek-v4/develop/progress/p36/bench_sinkhorn.py` (new) | `--mode {k4, k4_small, k8}` with V4-Flash EP=8 default widths (production `B=1, S=4096, K=4`; coverage `B=2, S=64, K=4`; forward-compat `B=1, S=4096, K=8`), proxy-mode flags (`--n-input-copies`, `--l2-flush-mb`) mirroring P34 / P35. |
| Trace launcher | `deepseek-v4/develop/progress/p36/run_baseline_trace_ep8_p36.sh` (new) | Mirrors P35's trace launcher; carries forward `PRIMUS_STACK_GROUPED_WEIGHT_TRITON=1`, `PRIMUS_ROPE_TRITON=1`, and adds `PRIMUS_SINKHORN_TRITON=1`. Override to `0` for the A side. |
| Proxy script | `run_deepseek_v4_flash_proxy.sh` | `PRIMUS_SINKHORN_TRITON` added under the plan-6 elemwise-fusion block as default-on (matches the plan-5 P32 / plan-6 P34 / P35 precedent of surfacing every shipped fusion knob). |
| Elem-fusion tracker | `deepseek-v4/develop/perf/elem_fusion.md` | P36 row populated with the per-shape bench delta + EP=8 proxy iter time. |
| Proxy summary | `deepseek-v4/develop/perf/proxy_ep8.md` | P36 row added with iter time + delta vs P35. |

---

## 3. Gates

| gate | status | numbers |
|---|---|---|
| **G39** unit tests (FWD/BWD parity vs eager + compiled + edge cases + env-flag) | **GREEN** | `pytest tests/unit_tests/megatron/transformer/deepseek_v4/test_p36_sinkhorn_triton.py` on `mi355-gpu-8` / `dev_primus_wenx_693`: **26 passed, 1 deselected** at fast tier and **1 passed** at release tier (V4-Flash B=1, S=4096, K=4). |
| **G39a** EP8 proxy smoke; `lm_loss` within bf16 tolerance of compiled fallback at fixed seed | **GREEN** | `lm_loss[10] = 9.258826` (Triton) vs `9.258817` (compiled fallback), diff `9e-6` -- bit-identical to bf16 precision. |
| **G39b** EP8 proxy A/B; iter time drops vs compiled fallback | **GREEN** | Iter-10 instantaneous **515.0 ms (Triton) vs 526.2 ms (compiled)**, **-11.2 ms (-2.1 %)**.  TFLOP/s/GPU **520.4 (Triton) vs 509.3 (compiled), +11.1 (+2.2 %)**. Matches microbench prediction of 16 calls × (0.893 − 0.144) = **12.0 ms** within profiler noise. |
| Plan-5 G32 (P29 `use_compiled` Sinkhorn boundary) stays green | **GREEN** | `pytest -m slow tests/unit_tests/.../test_v4_p29_compiled_sinkhorn.py`: 4/4 release-tier tests pass after the `autouse=True` PRIMUS_SINKHORN_TRITON=0 fixture is in place. |
| Plan-4 / plan-5 release-tier `pytest.mark.slow` ratchet stays green | **GREEN** | `pytest -m slow tests/unit_tests/megatron/transformer/deepseek_v4/` returned **95 passed, 357 deselected** in 73.27 s (vs the 94-passed P35 baseline; the +1 is the G39 release-tier V4-Flash test). |

---

## 4. Performance delta

### 4.1 Standalone microbench

Reproduce with `deepseek-v4/develop/progress/p36/bench_sinkhorn.py`:

```bash
python3 deepseek-v4/develop/progress/p36/bench_sinkhorn.py \
  --mode k4 --json-out deepseek-v4/develop/progress/p36/bench/k4.json
python3 deepseek-v4/develop/progress/p36/bench_sinkhorn.py \
  --mode k4_small --json-out deepseek-v4/develop/progress/p36/bench/k4_small.json
python3 deepseek-v4/develop/progress/p36/bench_sinkhorn.py \
  --mode k8 --json-out deepseek-v4/develop/progress/p36/bench/k8.json
```

Measured on `mi355-gpu-8` inside container `dev_primus_wenx_693`
(2026-05-14, `iters=20`, `warmup=5`, `n_input_copies=4`,
`l2_flush_mb=512`, bf16, `n_iters=20`) -- raw JSON at
`deepseek-v4/develop/progress/p36/bench/{k4,k4_small,k8}.json`:

| shape | path | FWD median (ms) | BWD median (ms) | FWD GB/s | BWD GB/s |
|---|---|---:|---:|---:|---:|
| k4 (V4-Flash B=1, S=4096, K=4; 128 KiB x / call)        | eager    | 0.587 | 1.502 |  18.3 |   7.2 |
| k4                                                       | compiled | 0.270 | 0.623 |  39.8 |  17.2 |
| k4                                                       | triton   | 0.043 | 0.101 | 249.4 | 106.0 |
| **k4 speedup vs eager**                                  |          | **13.62x** | **14.81x** | **+1265 %** | **+1372 %** |
| **k4 speedup vs P29 compiled**                           |          | **6.26x** | **6.15x** | **+528 %** | **+515 %** |
| k4_small (coverage B=2, S=64, K=4; 4 KiB x / call)       | eager    | 0.586 | 1.501 |   0.6 |   0.2 |
| k4_small                                                 | compiled | 0.291 | 0.627 |   1.2 |   0.5 |
| k4_small                                                 | triton   | 0.042 | 0.097 |   8.0 |   3.5 |
| **k4_small speedup vs eager**                            |          | **13.98x** | **15.47x** | **+1297 %** | **+1421 %** |
| k8 (forward-compat B=1, S=4096, K=8; 512 KiB x / call)   | eager    | 0.501 | 1.450 |  85.8 |  29.6 |
| k8                                                       | compiled | 0.288 | 0.623 | 149.4 |  69.0 |
| k8                                                       | triton   | 0.072 | 0.146 | 600.6 | 294.7 |
| **k8 speedup vs eager**                                  |          | **7.00x**  | **9.94x**  | **+599 %**  | **+894 %**  |
| **k8 speedup vs P29 compiled**                           |          | **4.02x**  | **4.27x**  | **+302 %**  | **+327 %**  |

Notes:

* At V4-Flash production widths (`k4`), the Triton FWD finishes in
  **43 µs** -- close to the HIP kernel launch granularity floor.
  The kernel's grid is `cdiv(4096, 128) = 32` blocks of
  `[BLOCK_LEADING=128, K=4, K=4]` and 39 iterations of in-register
  reductions; effectively a launch-latency-bound 4 KiB-state-per-row
  kernel.
* BWD is `~2.3x` slower than FWD (101 µs vs 43 µs) because it reads
  the 10 MiB state cache from HBM, which is bandwidth-bound at MI355
  ~3 TB/s peak (10 MiB / 3 TB/s ≈ 3 µs alone; the rest is the 39 VJP
  steps in registers + write of the 128 KiB dx).
* Per-call savings vs the plan-5 P29 compiled path (FWD + BWD):
  k4 **0.750 ms**; k4_small **0.779 ms**; k8 **0.693 ms**.
* Per-iter savings at V4-Flash (8 layers × 2 `HyperConnection`
  calls = 16 sinkhorn calls): **k4 = 12.0 ms / iter** predicted.

### 4.2 EP=8 proxy A/B

Measured by running `./run_deepseek_v4_flash_proxy.sh` with
`PRIMUS_SINKHORN_TRITON` toggled (filtered iter lines in
`deepseek-v4/develop/progress/p36/runs/{triton_on,triton_off}.iter_lines.txt`).
Iter time is the iter-10 instantaneous value (steady-state, post
iter-7 profile window):

| metric | P36 compiled fallback (`PRIMUS_SINKHORN_TRITON=0`) | P36 Triton (env on, default) | delta |
|---|---:|---:|---:|
| iter time (iter-10 instantaneous, ms) | **526.2** | **515.0** | **-11.2 ms (-2.1 %)** |
| iter-10 throughput TFLOP/s/GPU | 509.3 | 520.4 | **+11.1 (+2.2 %)** |
| `lm_loss[10]` | 9.258817 | 9.258826 | diff **9e-6** (bit-identical to bf16 precision) |
| ratchet vs plan-3 P28 baseline (8837.4 ms / iter) | 16.79× | **17.16×** | +0.37× |

Pass criteria:

* iter time drops by **≥ 8 ms** vs compiled fallback (the microbench
  predicted 12 ms / iter savings, target half that to allow for
  proxy noise / overlap) -- **MET** at 11.2 ms / iter (≈ 93 % of
  predicted, within profiler noise);
* `lm_loss[10]` matches the fallback to bf16 precision -- **MET**
  (diff `9e-6`, well below the `1e-3` bf16 rounding floor).

The microbench predicted 16 calls × 0.75 ms = **12.0 ms / iter**
savings; the observed proxy delta of 11.2 ms / iter matches within
**1 ms**.  Unlike P35 (where overlapping with prior BWD ate ~60 %
of the predicted saving), Sinkhorn lives on a serial path in
`HyperConnection.compute_weights` and its savings surface ~1:1 as
wall-clock.

---

## 5. Hand-off to P37

P36 removes the `Torch-Compiled Region` + `CompiledFunctionBackward`
small-op buckets.  The post-P36 trace will surface the **next-largest**
small-op residual for P37 (`HyperConnection` elemwise glue:
`slice → scale → base → sigmoid → softmax` chain after `_packed_logits`,
plus `collapse`'s `(pre * x).sum` and `expand`'s `post * out` outer
product).  P37's success metric will be recomputed against the
post-P36 trace.

---

## 6. Failed / negative probes

* **Initial gradcheck at fp64 input.** `torch.autograd.gradcheck` with
  `eps=1e-6` was added in an early G39 draft to catch any analytic
  VJP bug in `_sinkhorn_bwd_kernel`.  It failed with `max_abs ≈ 0.01`
  -- the kernel does internal fp32 compute (matches the eager
  `m = logits.float()` contract bit-for-bit), and the fp32 cast at
  the kernel boundary is lossy at the fp64 finite-difference step
  size, so gradcheck's numerical gradient and the kernel's
  analytical gradient legitimately disagree.  Replaced with a direct
  parity check against `torch.autograd` of the eager body in fp64
  (where the gradient flows correctly through the cast chain by
  PyTorch's standard autograd rules).
* **In-register FWD trajectory recompute during BWD.** First BWD
  draft tried to recompute `m_0 ... m_{2*n_iters - 1}` in registers
  during the backward pass to avoid the FWD-side state buffer.  This
  required holding the trajectory in a Python list indexed at
  runtime, which the Triton AST visitor in our toolchain rejects
  (it disallows runtime indexing into Python lists of `tl.tensor`s,
  and also disallows `zip()` / `reversed()` over them).  Restructured
  to materialise the FWD trajectory once into a `[N, 2*n_iters, K, K]`
  fp32 HBM buffer (~10 MiB / call at V4-Flash widths); BWD reads
  the buffer once and walks the analytic VJP forward-to-backward.
  The HBM round-trip for the cache is ~3 µs at MI355 peak HBM BW --
  negligible vs the 39 VJP steps' arithmetic.
* **`step % 2` runtime branch in the BWD loop.** First BWD draft
  used a Python `for step in range(...): if step % 2 == 0: axis=2
  else: axis=1` to alternate between row and col VJPs.  Triton
  rejected it with a shape mismatch in `tl.sum(..., keep_dims=True)`
  output across the two arms of the runtime conditional (axis=1 and
  axis=2 produce different `keep_dims` shapes).  Refactored to a
  Python `for i in range(n_iters - 1):` loop with explicit col then
  row steps inside; the priming col step lives outside the loop.
  Both axes are now compile-time `tl.constexpr` ints from Triton's
  perspective.

---

## 7. Open questions

* **HBM state buffer vs in-register recompute.** Our toolchain's
  Triton AST visitor forced the HBM-cached-state approach; a newer
  Triton release with proper support for runtime-indexed
  `tl.tensor` lists would allow an in-register recompute and save
  the ~10 MiB / call HBM round-trip.  At V4-Flash widths the state
  buffer is < 0.1 % of the rank footprint and the BWD is already
  bound by 39 VJP steps + 128 KiB dx write, so the saving would be
  marginal -- deferred unless a future Triton drop changes the
  budget.
* **Larger `K` shapes.** The current heuristic supports `K ∈
  {1, 2, 4, 8, 16}` with `BLOCK_LEADING` dropping from 128 (K=4)
  to 8 (K=16). V4-Flash uses K=4 today; `K=8` is forward-compat
  for plan-7 / V5 architectures and shows similar speedup ratios
  (7-10x vs eager, 4x vs compiled). `K > 16` would require a
  different kernel layout (tile the K×K matrix across program IDs).

---

## 8. References

* Plan-6 P36 design: `deepseek-v4/develop/plan-6/02-phase-details.md#phase-36--sinkhorn_normalize-triton-fwdbwd-replaces-plan-5-p29-torchcompile`
* Plan-5 P29 compiled body: `primus.backends.megatron.core.transformer.hyper_connection._build_compiled_sinkhorn`
* Plan-5 P32 final trace baseline: `develop/perf/proxy_ep8.md` "P32 final" row
* Eager body reference: `primus.backends.megatron.core.transformer.hyper_connection.sinkhorn_normalize`
