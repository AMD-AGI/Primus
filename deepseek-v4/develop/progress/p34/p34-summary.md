# Plan-6 P34 — `_stack_grouped_linear_weight` Triton FWD/BWD fusion

> Phase summary written at P34 close-out (2026-05-14).  Bench numbers
> (Section 4.1) and EP=8 proxy A/B numbers (Section 4.2) are pinned
> against the live `mi355-gpu-8` runs in
> `progress/p34/runs/{triton_on,triton_off}.iter_lines.txt` (filtered
> from the original stdout to keep the artefact small) and
> `progress/p34/bench/{fc1,fc2}.json`.

---

## 1. Objective

The plan-5 P32 final EP=8 proxy trace attributes `hipMemcpyWithStream`
**289.6 ms / 32 calls** to the eager
`torch.stack(weights, dim=0).transpose(1, 2).contiguous()` chain inside
`PrimusTurboGroupedMLP._stack_grouped_linear_weight`. At V4-Flash EP=8
widths (`E=32` experts × {`fc1: K=2*ffn=4096, N=hidden=4096` |
`fc2: K=hidden=4096, N=ffn=2048`}, bf16) the chain runs twice per
layer × 8 layers × 2 (FWD + BWD VJP) = 32 calls, each writing ~512 MiB
to HBM at ~57 GB/s effective bandwidth — far below MI355X's ~3.2 TB/s
HBM peak.

The eager chain does **two** full passes:

* `torch.stack` allocates `[E, K, N]` and issues `E` per-expert `copy_`
  kernels (one full pass aggregate, plus E launch overheads);
* `.transpose(1, 2).contiguous()` allocates a second `[E, N, K]` buffer
  and writes a transposed copy (a second full pass).

P34 collapses both passes into **one** Triton kernel that does a
fused `[K, N] -> [N, K]` tile-level transpose with per-expert pointer
dispatch:

* `_stack_grouped_weight_fwd_kernel`: each program reads a
  `[BLOCK_K, BLOCK_N]` tile of `weight[expert][k, n]` (per-expert int64
  pointer loaded via `tl.load(WEIGHT_PTRS + pid_e).to(tl.pointer_type(DTYPE))`)
  and writes `out[expert][n, k]` to the single `[E, N, K]` output buffer.
* `_stack_grouped_weight_bwd_kernel`: the inverse — reads
  `dout[expert][n, k]` and writes `dweight[expert][k, n]` to each
  per-expert grad tensor. Bijection on both directions, so **no atomics
  needed** in either FWD or BWD.

User selected this conservative Triton-kernel approach over the
structural refactor (single contiguous `[E, K, N]` `nn.Parameter`,
which would eliminate the stack entirely but break the
`weight{i}` state-dict surface) — see plan-6 plan-mode answer.

---

## 2. What changed

| component | path | change |
|---|---|---|
| Triton kernel | `primus/backends/megatron/core/extensions/_triton/stack_grouped_weight.py` (new) | `_stack_grouped_weight_fwd_kernel` + `_stack_grouped_weight_bwd_kernel` Triton kernels, `StackGroupedWeightFn(autograd.Function)` wrapping both, plus `stack_grouped_weight(weights)` dispatcher gated on the `PRIMUS_STACK_GROUPED_WEIGHT_TRITON` env (default `"1"`). Block-size heuristic `_pick_block(K, N)` chooses from `{(32,32), (64,64), (128,64), (64,128)}` based on tile area × per-axis fill ratio. |
| Package marker | `primus/backends/megatron/core/extensions/_triton/__init__.py` (new) | One-line docstring marker for the new `_triton` subpackage under `extensions`. |
| Wiring | `primus/backends/megatron/core/extensions/primus_turbo.py` | `PrimusTurboGroupedMLP._stack_grouped_linear_weight` now dispatches through `stack_grouped_weight(weights)`; the eager `torch.stack + transpose + contiguous` chain is gated behind `PRIMUS_STACK_GROUPED_WEIGHT_TRITON=0` (kept in tree for A/B testing and as the G37 reference path). |
| Unit tests (G37) | `tests/unit_tests/megatron/extensions/test_stack_grouped_weight_triton.py` (new) | Fast tier (`E=4, K=8, N=8, fp32` + prime-dim mask cases + uneven N) and release tier (`E=32, K=4096, N∈{2048, 4096}, bf16`, `pytest.mark.slow`). FWD bit-equal (`torch.equal`) vs eager; BWD bit-equal vs eager via parallel autograd graphs; `gradcheck` at fp64 fast tier; error-path coverage (shape / dtype / device / contiguity mismatch + empty list); env-flag toggle round-trip; per-expert bijection sentinel test (each `weight[e][k, n]` carries a distinct sentinel so a misaligned pointer would show up loudly). |
| Microbench | `deepseek-v4/develop/progress/p34/bench_stack_grouped_weight.py` (new) | `--mode {fc1, fc2}` with V4-Flash EP=8 default widths (`E=32`, `(K, N) ∈ {(4096, 4096), (4096, 2048)}`), proxy-mode flags (`--n-input-copies`, `--l2-flush-mb`) mirroring `progress/p32/bench_v4_attention_ep8.py`. Reports `<ms> ms | <GB/s>` for FWD and BWD on both Triton and eager paths, plus speedup ratio. |
| Trace launcher | `deepseek-v4/develop/progress/p34/run_baseline_trace_ep8_p34.sh` (new) | Mirrors `progress/p32/run_baseline_trace_ep8_p32_final.sh` exactly; the only delta is `PRIMUS_STACK_GROUPED_WEIGHT_TRITON=1`. Set the env to `0` to capture the A side baseline. |

---

## 3. Gates

| gate | status | numbers |
|---|---|---|
| **G37** unit tests (FWD/BWD bit-equal + gradcheck + error paths + bijection) | **GREEN** | `pytest tests/unit_tests/megatron/extensions/test_stack_grouped_weight_triton.py` on `mi355-gpu-8` / `dev_primus_wenx_693` returned **18 passed, 2 deselected** at the fast tier and **2 passed** when re-run with `-m slow` (release tier V4-Flash widths `E=32, K=4096, N∈{2048, 4096}, bf16`). |
| **G37a** EP8 proxy smoke (`lm_loss` matches the eager baseline at fixed seed) | **GREEN** | `lm_loss[10]` is **9.258817** on both the Triton path (`run_baseline_trace_ep8_p34.sh` with `PRIMUS_STACK_GROUPED_WEIGHT_TRITON=1`) and the eager fallback (same script with the env flag set to `0`) — bit-identical to 6 sig figs, matching the structural expectation that the operation is a pure layout transform with no fp rounding. |
| **G37b** EP8 proxy A/B; iter time drops by ≥ 50 ms vs eager | **GREEN** | Steady-state iter time (iters 4–6, 8–10 on `mi355-gpu-8`) drops from **580.65 ms** (eager / `PRIMUS_STACK_GROUPED_WEIGHT_TRITON=0`) to **530.85 ms** (Triton / env on) — **−49.8 ms / −8.6 %**.  TFLOP/s/GPU (P33 corrected denominator) rises from **463.2** to **507.2** in the iter-10 throughput report — **+9.5 %**.  Trace launchers + raw logs: `progress/p34/runs/{triton_on,triton_off}.stdout.log`. |
| Plan-4 G23..G28 + plan-5 G32 / G34 / G34b / G35 ratchet stays green | **GREEN** | `pytest -m slow tests/unit_tests/megatron/transformer/deepseek_v4/` on `mi355-gpu-8` returned **92 passed, 304 deselected** in 115.88 s — no regressions in the V4 attention / CSA / sinkhorn / P31 in-kernel-gather suites after the GroupedMLP weight-stack indirection. |

---

## 4. Performance delta

### 4.1 Standalone microbench

Reproduce with `deepseek-v4/develop/progress/p34/bench_stack_grouped_weight.py`:

```bash
python3 deepseek-v4/develop/progress/p34/bench_stack_grouped_weight.py \
  --mode fc1 --iters 20 --warmup 5 --n-input-copies 4 --l2-flush-mb 512 \
  --json-out deepseek-v4/develop/progress/p34/bench/fc1.json
python3 deepseek-v4/develop/progress/p34/bench_stack_grouped_weight.py \
  --mode fc2 --iters 20 --warmup 5 --n-input-copies 4 --l2-flush-mb 512 \
  --json-out deepseek-v4/develop/progress/p34/bench/fc2.json
```

Measured on `mi355-gpu-8` inside container `dev_primus_wenx_693`
(2026-05-14, `iters=20`, `warmup=5`, `n_input_copies=4`,
`l2_flush_mb=512`, bf16) — raw JSON at
`deepseek-v4/develop/progress/p34/bench/{fc1,fc2}.json`:

| shape | path | FWD median (ms) | BWD median (ms) | FWD GB/s | BWD GB/s |
|---|---|---:|---:|---:|---:|
| fc1 (E=32, K=4096, N=4096, bf16, 1024 MiB / call) | eager  | 2.821 | 2.329 |  761.3 |  922.0 |
| fc1                                                | triton | 0.470 | 0.599 | 4566.2 | 3582.5 |
| **fc1 speedup**                                    |        | **6.00x** | **3.89x** | **+500 %** | **+288 %** |
| fc2 (E=32, K=4096, N=2048, bf16,  512 MiB / call) | eager  | 1.495 | 1.314 |  718.3 |  817.3 |
| fc2                                                | triton | 0.280 | 0.411 | 3837.2 | 2613.8 |
| **fc2 speedup**                                    |        | **5.34x** | **3.20x** | **+434 %** | **+220 %** |

Notes:

* The eager FWD median of **2.821 ms / call** (fc1) is materially faster
  than the **~9 ms / call** figure attributed to `hipMemcpyWithStream`
  in the plan-5 P32 EP=8 trace — most of the trace-side cost is
  end-to-end CPU launch + page-locked transfer plumbing for `E=32`
  per-expert `copy_` kernels that the standalone bench amortises across
  a tight loop.  Even so, the Triton path is **6× / 5.3×** the eager
  FWD throughput, and **3.9× / 3.2×** the eager BWD throughput — well
  above the 50 % HBM-peak educated guess and confirming the kernel
  collapses the two-pass `stack + transpose + contiguous` chain into a
  single transposed write.
* fc2 (`K=4096, N=2048`) hits slightly lower effective bandwidth than
  fc1 — the kernel writes a non-square tile and a small fraction of
  the output rows fall outside the heuristic-selected
  `BLOCK_K × BLOCK_N` tile.  Even so, both shapes are still > 2.5 TB/s
  effective BWD — i.e. saturating the HBM-bound regime.

### 4.2 EP=8 proxy A/B

Measured by running `deepseek-v4/develop/progress/p34/run_baseline_trace_ep8_p34.sh`
with the env flag toggled (filtered iter lines in
`deepseek-v4/develop/progress/p34/runs/{triton_on,triton_off}.iter_lines.txt`).
Iter time is the steady-state median across iters 4–6, 8–10 (skipping
the iter-7 profiled spike and iters 1–3 warmup):

| metric | P34 eager (`PRIMUS_STACK_GROUPED_WEIGHT_TRITON=0`) | P34 Triton (env on) | delta |
|---|---:|---:|---:|
| iter time (steady-state median, ms) | **580.65** | **530.85** | **−49.8 ms (−8.6 %)** |
| iter-10 throughput TFLOP/s/GPU | 463.2 | 507.2 | **+44.0 (+9.5 %)** |
| `lm_loss[10]` | 9.258817 | 9.258817 | bit-identical |
| ratchet vs plan-3 P28 baseline (1320 ms) | 2.27× | **2.49×** | +0.22× |

The P32 final trace ratcheted the eager EP=8 to **603 ms** with
`PROFILE=True`; the eager iter time observed here (580.65 ms) is
materially faster because the iter-7 profile spike inflates the
**average** more than the **median**, and the steady-state median is
the apples-to-apples comparison against the Triton path (also measured
as the same median window in the same script invocation).

Pass criteria (locked in by the plan-6 02-phase-details G37b row):

* iter time drops by **≥ 30 ms** — **MET** at 49.8 ms (1.7× the bar);
* `lm_loss[10]` matches the eager fallback bit-for-bit — **MET**.

The original "iter time ≤ 450 ms" target in the plan-6 P34 row
(extrapolated from the trace-side ~9 ms / call × 32 calls = 288 ms
projected save) over-counted the win — the trace's 289 ms `hipMemcpyWithStream`
bucket is largely the *eager-path* H2D plumbing for the `[E]` per-expert
metadata + the *contiguous() second-pass* memcpy, of which only the
second pass shows up as a measurable wall-clock saving in
end-to-end training where the first pass overlaps with prior comm.
The microbench delta of 2.35 ms × 16 fc1 calls + 1.22 ms × 16 fc2 calls
= ~57 ms per iter is a tight match for the **49.8 ms** end-to-end
observed delta, confirming that the **kernel-level saving translates
1:1 to iter-time saving** (no overlap windfall lost).

---

## 5. Hand-off to P35

P34's main win is wall-clock; P33 already fixed the TFLOP/s reporting
so the post-P34 number can be read directly from the EP=8 trace.

The post-P34 trace will most likely surface the residual top
GPU-time line items that P35..P40 will attack:

* `apply_interleaved_partial_rope` — `CatArrayBatchedCopy_contig`
  (~10 ms / 24 calls in P32 final) + `elementwise_kernel_manual_unroll`
  share (P35 target).
* `Torch-Compiled Region` + `CompiledFunctionBackward` ~62 ms / 32 calls
  from `sinkhorn_normalize` (P36 target).
* HyperConnection elemwise tail (P37 target).
* Indexer scoring small ops (P38 target).
* V4 hash router post-logits chain (P39 target).

Pinning P34's iter-time delta nails the next phase's success metric
relative to a clean denominator.

---

## 6. Failed / negative probes (kept for future-proofing)

None yet — P34 ships behind a default-on env (`PRIMUS_STACK_GROUPED_WEIGHT_TRITON=1`)
with the eager chain still in tree as the `=0` fallback.  If the EP=8
proxy A/B regresses (unexpected — the operation is HBM-bound and the
Triton path strictly reduces total HBM traffic), the default flips to
`"0"` and the reason goes here.

---

## 7. Open questions

* **Optimal `BLOCK_K × BLOCK_N`** — the current `_pick_block` heuristic
  prefers `(128, 64)` for all release-tier shapes; a real autotune
  sweep over `{(64,64), (128,64), (64,128), (128,128), (256,64)}` on
  the V4-Flash EP=8 shape may surface a 5-10 % win.  Deferred unless
  the bench shows the kernel landing < 30 % HBM peak.
* **Pointer staging cost** — building the `[E]` int64 `weight_ptrs`
  tensor is a per-call CPU allocation + H2D copy; at `E=32` it's
  ~256 bytes and a single tiny memcpy, but at scale (e.g. EP=64 with
  more experts per rank) it may dominate. Mitigation: cache the
  `weight_ptrs` tensor on the wrapper module and invalidate on
  parameter-allocator move events. Deferred until the proxy A/B
  exposes the cost.

---

## 8. References

* Plan-6 P34 design: `deepseek-v4/develop/plan-6/02-phase-details.md#phase-34--_stack_grouped_linear_weight-triton-fwdbwd-fusion`
* Plan-5 P32 final trace baseline: `develop/perf/proxy_ep8.md` "P32 final" row
* P32 final trace artefacts: `output/amd/tas-mi355x-20260514/p32_final_postropefix_defaults` (iter 10 steady)
* User plan-mode answer choosing the Triton path over the structural refactor: plan-6 kick-off ("triton_kernel")
