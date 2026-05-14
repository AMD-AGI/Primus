# Plan-6 P35 — `apply_interleaved_partial_rope` Triton FWD/BWD fusion

> Phase summary written at P35 close-out (2026-05-14). Bench numbers
> (§4.1) + EP=8 proxy A/B numbers (§4.2) are pinned against the live
> `mi355-gpu-8` / `dev_primus_wenx_693` runs in
> `progress/p35/runs/{triton_on,triton_off}.iter_lines.txt` and
> `progress/p35/bench/{q,k}.json`.

---

## 1. Objective

The plan-5 P32 final EP=8 proxy trace attributes:

* `CatArrayBatchedCopy_contig` ≈ **10.0 ms / 24 calls** to the closing
  `torch.cat([x_nope, rotated], -1)` of `apply_interleaved_partial_rope`;
* a non-trivial share of `elementwise_kernel_manual_unroll<128, 8>`
  (~61 ms / 693 calls) to the four broadcast muls inside the rotation.

The function is called **16 times per iter** (q + k per `DualRoPE` call
× 8 layers), so the per-call cost is **~3-5 ms** on the EP=8 proxy at
V4-Flash widths.  Removing the closing `cat` alone is worth ≥ 10 ms
per iter; collapsing the 9-op eager chain into a single kernel should
land in the 20-40 ms per-iter range based on the trace's per-call
attribution.

P35 fuses the 9-op chain (`slice → slice → reshape → 4× mul + add/sub
→ stack → reshape → cat`) into one Triton kernel that:

1. Reads `x [N, H, head_dim]` (any leading shape flattened);
2. Loads cos / sin once per position (shared across `BLOCK_H` heads);
3. Writes `out [N, H, head_dim]` in **one** contiguous pass — nope
   prefix copied verbatim, rotary suffix rotated using the interleaved
   `(2k, 2k+1)` pairing.

The BWD kernel applies the transpose rotation (analytic VJP), with no
saved input and only `cos / sin` saved for the backward.

---

## 2. What changed

| component | path | change |
|---|---|---|
| Triton kernel | `primus/backends/megatron/core/transformer/v4_attention_kernels/_triton/rope_interleaved_partial.py` (new) | `_apply_rope_fwd_kernel` + `_apply_rope_bwd_kernel` + `RoPEInterleavedPartialFn(autograd.Function)` + `apply_rope_interleaved_partial(...)` dispatcher gated on `PRIMUS_ROPE_TRITON` env (default `"1"`). Block heuristic: `BLOCK_H = max(1, min(8, H))`; `BLOCK_NOPE = next_pow2(nope)`; `BLOCK_RD_HALF = next_pow2(rotary_dim // 2)`. Supports `{fp64, fp32, fp16, bf16}` (fp64 added for `gradcheck`). |
| Package docstring | `primus/backends/megatron/core/transformer/v4_attention_kernels/_triton/__init__.py` | Adds a new bullet for `rope_interleaved_partial`. |
| Wiring | `primus/backends/megatron/core/transformer/dual_rope.py` | `apply_interleaved_partial_rope` now routes through `RoPEInterleavedPartialFn` when `x.is_cuda` and `PRIMUS_ROPE_TRITON != "0"`. Eager body kept in tree as the `=0` fallback and as the G38 reference. |
| Unit tests (G38) | `tests/unit_tests/megatron/transformer/deepseek_v4/test_p35_rope_triton.py` (new) | Fast tier (`B=2, S=8, H=4, head_dim=16, rd ∈ {0, 4, 8, 16}` × `{fp32, fp16, bf16}`) FWD + BWD parity vs the eager body, plus fp64 `gradcheck`. Release tier (`pytest.mark.slow`) covers Q (`H=64, head_dim=512, rd=64`) and K (`H=1, head_dim=64, rd=64`) at bf16. Edge cases: `rd=0` early-return; odd / over-sized rotary_dim raise `ValueError`; cos/sin shape mismatch raises. Env-flag dispatch test pins both code paths through `dual_rope.apply_interleaved_partial_rope`. |
| Microbench | `deepseek-v4/develop/progress/p35/bench_rope_triton.py` (new) | `--mode {q, k}` with V4-Flash EP=8 default widths (Q: `B=1, S=4096, H=64, head_dim=512, rd=64`; K: same but `H=1, head_dim=64`), proxy-mode flags (`--n-input-copies`, `--l2-flush-mb`) mirroring P34. |
| Trace launcher | `deepseek-v4/develop/progress/p35/run_baseline_trace_ep8_p35.sh` (new) | Mirrors P34's trace launcher; carries forward `PRIMUS_STACK_GROUPED_WEIGHT_TRITON=1` and adds `PRIMUS_ROPE_TRITON=1`. Override to `0` for the A side. |
| Proxy script | `run_deepseek_v4_flash_proxy.sh` | `PRIMUS_ROPE_TRITON` added under the plan-6 elemwise-fusion block as default-on (matches the plan-5 P32 / plan-6 P34 precedent of surfacing every shipped fusion knob). |
| Elem-fusion tracker | `deepseek-v4/develop/perf/elem_fusion.md` | P35 row populated with the per-shape bench delta + EP=8 proxy iter time. |
| Proxy summary | `deepseek-v4/develop/perf/proxy_ep8.md` | P35 row added with iter time + delta vs P34. |

---

## 3. Gates

| gate | status | numbers |
|---|---|---|
| **G38** unit tests (FWD/BWD parity + gradcheck + edge cases + env-flag) | **GREEN** | `pytest tests/unit_tests/megatron/transformer/deepseek_v4/test_p35_rope_triton.py` on `mi355-gpu-8` / `dev_primus_wenx_693`: **27 passed, 2 deselected** at fast tier and **2 passed** at release tier (Q + K shapes). |
| **G38a** EP8 proxy smoke; `lm_loss` within bf16 tolerance of eager fallback at fixed seed | **GREEN** | `lm_loss[10] = 9.258817` bit-identical between `PRIMUS_ROPE_TRITON=1` (Triton) and `=0` (eager) paths. |
| **G38b** EP8 proxy A/B; iter time drops vs eager | **GREEN (small)** | Steady-state iter time **531.7 ms (eager) -> 526.7 ms (Triton)**, **-5.0 ms (-0.94%)**.  TFLOP/s/GPU **507.1 -> 513.3 (+1.2%)**.  Lower than the initially-projected 20-40 ms because the P32 final trace's 10 ms / 24 calls `CatArrayBatchedCopy_contig` bucket maps to roughly half that as steady-state per-iter cost (the rest overlaps with prior work).  Result is consistent with the microbench prediction (8 Q calls × 0.29 ms + 8 K calls × 0.04 ms + corresponding BWD saving ≈ 5.7 ms / iter). |
| Plan-4 / plan-5 release-tier `pytest.mark.slow` ratchet stays green | **GREEN** | `pytest -m slow tests/unit_tests/megatron/transformer/deepseek_v4/` returned **94 passed, 331 deselected** in 72.90 s (vs the 92-passed P34 baseline; the +2 are G38 release-tier tests). |

---

## 4. Performance delta

### 4.1 Standalone microbench

Reproduce with `deepseek-v4/develop/progress/p35/bench_rope_triton.py`:

```bash
python3 deepseek-v4/develop/progress/p35/bench_rope_triton.py \
  --mode q --json-out deepseek-v4/develop/progress/p35/bench/q.json
python3 deepseek-v4/develop/progress/p35/bench_rope_triton.py \
  --mode k --json-out deepseek-v4/develop/progress/p35/bench/k.json
```

Measured on `mi355-gpu-8` inside container `dev_primus_wenx_693`
(2026-05-14, `iters=20`, `warmup=5`, `n_input_copies=4`,
`l2_flush_mb=512`, bf16) — raw JSON at
`deepseek-v4/develop/progress/p35/bench/{q,k}.json`:

| shape | path | FWD median (ms) | BWD median (ms) | FWD GB/s | BWD GB/s |
|---|---|---:|---:|---:|---:|
| Q (B=1, S=4096, H=64, head_dim=512, rd=64; 256 MiB / call) | eager  | 0.437 | 0.524 | 1230.6 | 1024.6 |
| Q                                                          | triton | 0.148 | 0.187 | 3637.9 | 2878.1 |
| **Q speedup**                                              |        | **2.96x** | **2.81x** | **+196 %** | **+181 %** |
| K (B=1, S=4096, H=1,  head_dim=64,  rd=64; 0.5 MiB / call) | eager  | 0.064 | 0.140 |   24.6 |   11.2 |
| K                                                          | triton | 0.027 | 0.093 |   57.2 |   17.0 |
| **K speedup**                                              |        | **2.33x** | **1.51x** | **+133 %** | **+52 %** |

Notes:

* The Q shape hits **3.6 TB/s** effective HBM read+write bandwidth
  (vs MI355X ~3.2 TB/s peak per read+write pass; the kernel does one
  read + one write so the measurement is well above 50% HBM peak —
  validating the single-pass fused design).
* The K shape is **tiny** (0.5 MiB / call, ~1.5 MiB total traffic);
  effective bandwidth there is dominated by kernel launch latency and
  the fp32 cos/sin reads (eager + triton both look small in absolute
  ms terms — 0.027 ms / call is below the ~0.05 ms HIP launch
  granularity on MI355X).  The relative 2.3x speedup still
  translates 1:1 to wall-clock saving in the proxy because the launch
  cost is incurred 16 times per iter regardless.

Per-iter savings from microbench (8 layers × 1 RoPE call per Q + K,
× 1 FWD + 1 BWD):

* Q FWD: 8 × (0.437 − 0.148) = **2.31 ms**
* Q BWD: 8 × (0.524 − 0.187) = **2.70 ms**
* K FWD: 8 × (0.064 − 0.027) = **0.30 ms**
* K BWD: 8 × (0.140 − 0.093) = **0.38 ms**

Total predicted per-iter save: **5.69 ms / iter** — matches the
observed EP=8 proxy delta of 5.0 ms / iter to within profiler noise.

### 4.2 EP=8 proxy A/B

Measured by running
`deepseek-v4/develop/progress/p35/run_baseline_trace_ep8_p35.sh` with
`PRIMUS_ROPE_TRITON` toggled (filtered iter lines in
`deepseek-v4/develop/progress/p35/runs/{triton_on,triton_off}.iter_lines.txt`).
Iter time is the steady-state median across iters 4–6, 8–10 (skipping
the iter-7 profiled spike and iters 1–3 warmup):

| metric | P35 eager (`PRIMUS_ROPE_TRITON=0`) | P35 Triton (env on, default) | delta |
|---|---:|---:|---:|
| iter time (steady-state median, ms) | **531.7** | **526.7** | **-5.0 ms (-0.94 %)** |
| iter-10 throughput TFLOP/s/GPU | 507.1 | 513.3 | **+6.2 (+1.2 %)** |
| `lm_loss[10]` | 9.258817 | 9.258817 | bit-identical |
| ratchet vs plan-3 P28 baseline (8837.4 ms / iter) | 16.62× | **16.78×** | +0.16× |

Pass criteria:

* iter time drops by **≥ 4 ms** vs eager — **MET** at 5.0 ms (≈ 1.25× the bar);
* `lm_loss[10]` matches the eager fallback bit-for-bit — **MET**.

The original "iter time ≤ 420 ms" projection in the plan-6 P35 row was
over-optimistic: it extrapolated the trace's 10.0 ms / 24-call
`CatArrayBatchedCopy_contig` bucket as ~30 ms per-iter saving, but in
steady state the cat / mul cost overlaps materially with prior bwd
GPU work, so only ~5 ms per iter actually surfaces as wall-clock
saving.  This is consistent with the microbench (5.7 ms predicted)
and confirms that RoPE was a **moderate** small-op fusion target,
not a hot bottleneck.  The win is still positive at zero numerical
cost so the Triton path ships default-on.

---

## 5. Hand-off to P36

P35 removes the RoPE small-op chain.  The post-P35 trace will surface
the **next-largest** small-op bucket for P36 (`sinkhorn_normalize` —
`Torch-Compiled Region` + `CompiledFunctionBackward` ~62 ms / 32 calls
in the P32 final trace).  P36's success metric will be recomputed
against the post-P35 trace.

---

## 6. Failed / negative probes

None at landing — the operation is a pure layout transform + simple
rotation with no dtype trap (the plan-5 P32 RoPE bf16 cast contract
is carried forward inside the kernel via a `tl.cast` on cos / sin).
If the EP=8 proxy A/B regresses unexpectedly, the default flips to
`PRIMUS_ROPE_TRITON=0` and the reason goes here.

---

## 7. Open questions

* **Optimal `BLOCK_H` for Q (H=64).** Current heuristic picks `BLOCK_H=8`
  uniformly when `H >= 16`.  A real autotune sweep over
  `{4, 8, 16, 32}` may surface a 5-10 % win on the Q shape.  Deferred
  unless the bench shows the kernel under 50 % HBM peak.
* **Fused dropout (V4 has none on RoPE today).** Not relevant for V4;
  noted for potential plan-7 re-use.

---

## 8. References

* Plan-6 P35 design: `deepseek-v4/develop/plan-6/02-phase-details.md#phase-35--apply_interleaved_partial_rope-triton-fwdbwd-fusion`
* Plan-5 P32 RoPE bf16 cast contract: `deepseek-v4/develop/progress/p32/p32-summary.md`
* Plan-5 P32 final trace baseline: `develop/perf/proxy_ep8.md` "P32 final" row
* Eager body reference: `primus/backends/megatron/core/transformer/dual_rope.py::apply_interleaved_partial_rope`
